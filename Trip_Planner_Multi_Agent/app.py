"""
FastAPI Application for the Multi-Agent Trip Planner System
===========================================================
Wraps the Supervisor (Main Orchestrator + Hotel Sub-Agent + Flight Sub-Agent),
LangChain `HumanInTheLoopMiddleware`, and `SqliteSaver` checkpointer into REST endpoints.

Run locally with:
    uvicorn Trip_Planner_Multi_Agent.app:app --reload --port 8000
"""

from typing import Any, Dict, Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

try:
    from Trip_Planner_Multi_Agent.trip_planner_agent import (
        BOOKINGS_DB,
        create_trip_planner_system,
        format_agent_result,
    )
except ImportError:
    from trip_planner_agent import (
        BOOKINGS_DB,
        create_trip_planner_system,
        format_agent_result,
    )

app = FastAPI(
    title="Multi-Agent Trip Planner API (Subagents + HITL Middleware + SqliteSaver)",
    description=(
        "Hierarchical Multi-Agent Trip Planner using LangChain `create_agent`, "
        "`HumanInTheLoopMiddleware` for every approval gate, and persistent `SqliteSaver`."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Multi-Agent system backed by SQLite
main_agent, sqlite_checkpointer, sqlite_conn = create_trip_planner_system()


# =====================================================================
# PYDANTIC REQUEST & RESPONSE SCHEMAS
# =====================================================================

class ChatRequest(BaseModel):
    thread_id: str = Field(
        default="trip-session-1",
        description="Unique conversation thread ID stored in SQLite.",
    )
    message: str = Field(
        ...,
        description="User message (e.g., 'Hi, I am Sachin. I want to travel from Delhi to Goa with a $1500 budget').",
    )


class ApprovalRequest(BaseModel):
    thread_id: str = Field(
        default="trip-session-1",
        description="Conversation thread ID currently paused at a HumanInTheLoopMiddleware interrupt.",
    )
    decision_type: Literal["approve", "edit", "reject"] = Field(
        default="approve",
        description="HITL decision type: 'approve', 'edit', or 'reject'.",
    )
    edited_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of modified tool arguments when decision_type is 'edit'.",
    )
    reject_message: Optional[str] = Field(
        default=None,
        description="Optional reason message when decision_type is 'reject'.",
    )


def _get_pending_interrupt_info(thread_id: str) -> Optional[Dict[str, Any]]:
    """Inspect the current SQLite checkpoint state to find any pending HITL interrupt."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = main_agent.get_state(config)
    if not snapshot or not snapshot.tasks:
        return None

    for task in snapshot.tasks:
        if getattr(task, "interrupts", None):
            raw_int = task.interrupts[0].value
            action_requests = raw_int.get("action_requests", [])
            review_configs = raw_int.get("review_configs", [])
            if action_requests:
                act = action_requests[0]
                rev = review_configs[0] if review_configs else {}
                return {
                    "tool_name": act.get("name"),
                    "proposed_args": act.get("args", {}),
                    "description": act.get("description", ""),
                    "allowed_decisions": rev.get("allowed_decisions", ["approve", "edit", "reject"]),
                }
    return None


# =====================================================================
# FASTAPI ENDPOINTS
# =====================================================================

@app.post("/chat", summary="Send a user message to the Main Orchestrator Agent")
def chat_endpoint(req: ChatRequest) -> Dict[str, Any]:
    """
    Sends a user message into the conversation thread persisted in SQLite.
    If the Main Agent calls an approval-gated tool (`find_hotels_subagent`,
    `book_hotel_subagent`, or `book_flight_subagent`), `HumanInTheLoopMiddleware`
    pauses execution and returns the `interrupt` payload for user approval.
    """
    config = {"configurable": {"thread_id": req.thread_id}}

    # Check if thread is currently paused on an unapproved HITL interrupt
    pending = _get_pending_interrupt_info(req.thread_id)
    if pending:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Thread is currently paused waiting for Human-in-the-Loop approval. Call POST /approve first.",
                "pending_interrupt": pending,
            },
        )

    result = main_agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
    )
    formatted = format_agent_result(result)
    formatted["thread_id"] = req.thread_id
    formatted["confirmed_bookings"] = BOOKINGS_DB
    return formatted


@app.post("/approve", summary="Submit Human-in-the-Loop Decision (Approve / Edit / Reject)")
def approve_endpoint(req: ApprovalRequest) -> Dict[str, Any]:
    """
    Resumes a paused thread after a `HumanInTheLoopMiddleware` interrupt using
    `Command(resume={"decisions": [...]})`.
    """
    config = {"configurable": {"thread_id": req.thread_id}}
    pending = _get_pending_interrupt_info(req.thread_id)

    if not pending:
        raise HTTPException(
            status_code=400,
            detail=f"No pending HumanInTheLoopMiddleware interrupt found for thread_id='{req.thread_id}'.",
        )

    tool_name = pending["tool_name"]
    original_args = dict(pending["proposed_args"])

    if req.decision_type == "approve":
        decision: Dict[str, Any] = {"type": "approve"}
    elif req.decision_type == "edit":
        merged_args = {**original_args, **(req.edited_args or {})}
        decision = {
            "type": "edit",
            "edited_action": {
                "name": tool_name,
                "args": merged_args,
            },
        }
    elif req.decision_type == "reject":
        decision = {
            "type": "reject",
            "message": req.reject_message or "User rejected this action. Please ask how to proceed.",
        }
    else:
        raise HTTPException(status_code=400, detail=f"Invalid decision_type: {req.decision_type}")

    result = main_agent.invoke(
        Command(resume={"decisions": [decision]}),
        config=config,
    )
    formatted = format_agent_result(result)
    formatted["thread_id"] = req.thread_id
    formatted["resumed_from_tool"] = tool_name
    formatted["applied_decision"] = decision
    formatted["confirmed_bookings"] = BOOKINGS_DB
    return formatted


@app.get("/state/{thread_id}", summary="Inspect SQLite-persisted conversation state & interrupts")
def get_thread_state(thread_id: str) -> Dict[str, Any]:
    """Retrieve the persisted checkpoint state from SqliteSaver for a given thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = main_agent.get_state(config)

    if not snapshot or not snapshot.values:
        return {
            "thread_id": thread_id,
            "exists": False,
            "messages": [],
            "pending_interrupt": None,
            "confirmed_bookings": BOOKINGS_DB,
        }

    messages = snapshot.values.get("messages", [])
    serialized_msgs = []
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant" if isinstance(m, AIMessage) else "tool"
        content = m.content
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        entry: Dict[str, Any] = {"role": role, "content": str(content)}
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            entry["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")}
                for tc in m.tool_calls
            ]
        serialized_msgs.append(entry)

    return {
        "thread_id": thread_id,
        "exists": True,
        "next_nodes": list(snapshot.next),
        "pending_interrupt": _get_pending_interrupt_info(thread_id),
        "message_count": len(serialized_msgs),
        "messages": serialized_msgs,
        "confirmed_bookings": BOOKINGS_DB,
    }


@app.delete("/state/{thread_id}", summary="Delete SQLite checkpoints for a thread_id")
def reset_thread_state(thread_id: str) -> Dict[str, Any]:
    """Clears persisted SQLite checkpoints for the given thread_id."""
    cursor = sqlite_conn.cursor()
    for table in ["checkpoints", "writes"]:
        try:
            cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except Exception:
            pass
    sqlite_conn.commit()
    return {"status": "cleared", "thread_id": thread_id}


@app.get("/", response_class=HTMLResponse, summary="Interactive Web Console for Trip Planner API")
def index_ui() -> str:
    """Serves a lightweight interactive console to test /chat and /approve endpoints live."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Trip Planner Multi-Agent FastAPI Console</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen p-6">
  <div class="max-w-4xl mx-auto space-y-6">
    <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex items-center justify-between">
      <div>
        <span class="text-xs font-semibold uppercase tracking-wider bg-slate-100 text-slate-700 px-2.5 py-1 rounded">FastAPI + SqliteSaver + HITL Middleware</span>
        <h1 class="text-xl font-bold mt-2">Multi-Agent Trip Planner Live Console</h1>
        <p class="text-sm text-slate-500">Main Orchestrator + Hotel Sub-Agent + Flight Sub-Agent</p>
      </div>
      <div class="flex items-center gap-2">
        <input id="threadId" value="trip-session-1" class="border border-slate-300 rounded px-3 py-1.5 text-sm" />
        <button onclick="resetThread()" class="text-xs bg-slate-200 hover:bg-slate-300 px-3 py-2 rounded font-medium">Reset Thread</button>
      </div>
    </div>

    <div id="chatBox" class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4 min-h-[320px] max-h-[500px] overflow-y-auto">
      <div class="text-sm text-slate-400">Send a message below to start planning your trip (e.g., "Hi, I am Sachin. I want to plan a trip from Delhi to Goa with a budget of $1500").</div>
    </div>

    <div id="hitlBox" class="hidden bg-amber-50 border-2 border-amber-300 rounded-xl p-5 shadow-sm space-y-3">
      <div class="flex items-center justify-between">
        <span id="hitlStage" class="text-xs font-bold uppercase tracking-wider bg-amber-200 text-amber-900 px-2.5 py-1 rounded">HITL Interrupt</span>
        <span id="hitlTool" class="text-xs font-mono text-amber-800"></span>
      </div>
      <div id="hitlSubagentOutput" class="text-xs font-mono bg-white p-3 rounded border border-amber-200 whitespace-pre-wrap"></div>
      <label class="block text-xs font-semibold text-slate-700">Proposed Tool Arguments (Editable for 'Edit' decision):</label>
      <textarea id="hitlArgs" rows="4" class="w-full font-mono text-xs p-2.5 rounded border border-slate-300 bg-white"></textarea>
      <div class="flex gap-3">
        <button onclick="sendDecision('approve')" class="bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium px-4 py-2 rounded">Approve</button>
        <button onclick="sendDecision('edit')" class="bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium px-4 py-2 rounded">Edit & Proceed</button>
        <button onclick="sendDecision('reject')" class="bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium px-4 py-2 rounded">Reject</button>
      </div>
    </div>

    <div class="flex gap-3">
      <input id="userInput" type="text" placeholder="Type your message (e.g. Name, Budget, Source, Destination, or Flight Date)..." class="flex-1 border border-slate-300 rounded-lg px-4 py-2.5 text-sm" onkeydown="if(event.key==='Enter') sendChat()" />
      <button onclick="sendChat()" class="bg-slate-900 hover:bg-slate-800 text-white font-medium px-6 py-2.5 rounded-lg text-sm">Send</button>
    </div>
  </div>

  <script>
    function appendMsg(role, text) {
      if (!text) return;
      const box = document.getElementById('chatBox');
      const div = document.createElement('div');
      div.className = role === 'user' ? 'bg-slate-900 text-white p-3.5 rounded-lg text-sm ml-12 whitespace-pre-wrap' : 'bg-slate-100 text-slate-800 p-3.5 rounded-lg text-sm mr-12 whitespace-pre-wrap border border-slate-200';
      div.innerText = (role === 'user' ? 'You: ' : 'Main Agent: ') + text;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    function handleResponse(data) {
      if (data.assistant_message) appendMsg('assistant', data.assistant_message);
      const hitlBox = document.getElementById('hitlBox');
      if (data.interrupt) {
        hitlBox.classList.remove('hidden');
        document.getElementById('hitlStage').innerText = data.interrupt.stage;
        document.getElementById('hitlTool').innerText = 'Tool: ' + data.interrupt.tool_name;
        document.getElementById('hitlSubagentOutput').innerText = data.interrupt.latest_subagent_findings || 'Awaiting approval before sub-agent action...';
        document.getElementById('hitlArgs').value = JSON.stringify(data.interrupt.proposed_args, null, 2);
      } else {
        hitlBox.classList.add('hidden');
      }
    }

    async function sendChat() {
      const input = document.getElementById('userInput');
      const msg = input.value.trim();
      if (!msg) return;
      const thread_id = document.getElementById('threadId').value;
      appendMsg('user', msg);
      input.value = '';
      const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({thread_id, message: msg})
      });
      const data = await res.json();
      handleResponse(data);
    }

    async function sendDecision(decision_type) {
      const thread_id = document.getElementById('threadId').value;
      let edited_args = null;
      if (decision_type === 'edit') {
        edited_args = JSON.parse(document.getElementById('hitlArgs').value);
      }
      const res = await fetch('/approve', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({thread_id, decision_type, edited_args})
      });
      const data = await res.json();
      handleResponse(data);
    }

    async function resetThread() {
      const thread_id = document.getElementById('threadId').value;
      await fetch('/state/' + thread_id, {method: 'DELETE'});
      document.getElementById('chatBox').innerHTML = '<div class="text-sm text-slate-400">Thread reset. Start a new trip plan!</div>';
      document.getElementById('hitlBox').classList.add('hidden');
    }
  </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
