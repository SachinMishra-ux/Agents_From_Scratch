import json
import asyncio
import os
import warnings
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv

# Suppress python warnings for additionalProperties in JSON schema
warnings.filterwarnings("ignore", message=".*additionalProperties.*")

# Suppress library warnings logging via standard logging
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)


# Load env variables robustly
def load_env_robust():
    path = os.getcwd()
    for _ in range(5):
        dotenv_path = os.path.join(path, ".env")
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            return
        path = os.path.dirname(path)
    load_dotenv()

load_env_robust()

# Remap Google API key if GEMINI key is not explicitly set
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

# Path for conversation memory database
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_chat_history.db")

app = FastAPI(title="MCP Agent UI Server")

# Enable CORS for local development (allowing separate frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory server config registry
mcp_servers = {}

class ConnectRequest(BaseModel):
    name: str
    url: str
    transport: str
    headers: dict | None = None

class ChatRequest(BaseModel):
    message: str
    model: str = "gemini-3.1-flash-lite"
    thread_id: str = "default_thread"

@app.post("/api/mcp/connect")
async def connect_mcp(req: ConnectRequest):
    try:
        # Validate connection parameters by loading tools temporarily
        client = MultiServerMCPClient({
            req.name: {
                "url": req.url,
                "transport": req.transport,
                "headers": req.headers or {}
            }
        })
        tools = await client.get_tools(server_name=req.name)
        
        # Save connection configuration to registry
        mcp_servers[req.name] = {
            "url": req.url,
            "transport": req.transport,
            "headers": req.headers or {}
        }
        
        tools_list = [{"name": t.name, "description": t.description} for t in tools]
        return {"status": "success", "server": req.name, "tools": tools_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/mcp/list")
async def list_mcp():
    return {"servers": mcp_servers}

@app.delete("/api/mcp/disconnect/{name}")
async def disconnect_mcp(name: str):
    if name in mcp_servers:
        del mcp_servers[name]
        return {"status": "success", "message": f"Server {name} disconnected."}
    raise HTTPException(status_code=404, detail="Server not found")

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not mcp_servers:
        raise HTTPException(
            status_code=400, 
            detail="No MCP servers connected. Please connect at least one server using the '+' button in the sidebar."
        )
        
    # Instantiate client with all active servers
    client = MultiServerMCPClient(mcp_servers)
    
    try:
        tools = await client.get_tools()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tools from servers: {str(e)}")

    # Instantiate LLM and LangChain agent
    llm = ChatGoogleGenerativeAI(model=req.model, temperature=0)
    
    system_prompt = (
        "You are an expert Career and Compensation Counselor. Your goal is to help users understand their market value "
        "by utilizing the salary prediction, explanation, and gap analysis tools.\n\n"
        "CRITICAL: Only invoke the tool(s) that are directly relevant to the user's explicit request. Do not run all tools unless requested:\n"
        "- If the user only asks to predict their salary, only run the `predict_salary` tool.\n"
        "- If the user only asks to explain their prediction or factors, only run the `explain_salary_prediction` tool.\n"
        "- If the user only asks for a gap analysis or to see if they are underpaid/overpaid, only run the `salary_gap_analysis` tool.\n"
        "- If the user asks for multiple things or a complete analysis, run the corresponding combination of tools.\n\n"
        "Present a professional, encouraging, and clear analysis with exact figures from the tools. "
        "Be sure to map user attributes correctly:\n"
        "- experience_years: Years of experience (integer)\n"
        "- education_level: Level of education (integer, e.g. 1 for Bachelors, 2 for Masters, 3 for PhD)\n"
        "- num_skills: Number of skills (integer)\n"
        "- location_index: Location index (integer, e.g. 1 for Tier-1 city, 2 for Tier-2 city)\n"
        "- current_salary_lpa: Current salary in LPA (number)"
    )

    async def event_generator():
        try:
            async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
                agent = create_agent(
                    model=llm,
                    tools=tools,
                    system_prompt=system_prompt,
                    checkpointer=memory
                )
                config = {"configurable": {"thread_id": req.thread_id}}
                async for event in agent.astream_events(
                    {"messages": [HumanMessage(content=req.message)]},
                    config=config,
                    version="v2"
                ):
                    event_type = event.get("event")
                    name = event.get("name")
                    
                    if event_type == "on_tool_start":
                        inputs = event.get("data", {}).get("input")
                        payload = {"event": "tool_start", "name": name, "input": inputs, "run_id": event.get("run_id")}
                        yield f"data: {json.dumps(payload)}\n\n"
                        
                    elif event_type == "on_tool_end":
                        output = event.get("data", {}).get("output")
                        output_str = str(output)
                        payload = {"event": "tool_end", "name": name, "output": output_str, "run_id": event.get("run_id")}
                        yield f"data: {json.dumps(payload)}\n\n"
                        
                    elif event_type == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        
                        if isinstance(content, list):
                            text_content = "".join([c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"])
                        else:
                            text_content = str(content)
                            
                        if text_content:
                            payload = {"event": "chunk", "text": text_content}
                            yield f"data: {json.dumps(payload)}\n\n"
                            
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
        except Exception as e:
            payload = {"event": "error", "message": str(e)}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)