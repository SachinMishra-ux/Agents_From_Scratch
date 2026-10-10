"""
===============================================================================
⚡ SERVER MODULE — FastAPI WebSocket Gateway & Web Application
===============================================================================
This module hosts our application server and manages communication:

  1. Serves the interactive Web UI (index.html) and Visualizer (architecture.html).
  2. Provides the Bi-directional WebSocket endpoint (/ws):
     - Receives real-time 16kHz PCM audio bytes from the user's browser.
     - Feeds audio into our Sandwich Pipeline (STT ➔ Agent ➔ TTS).
     - Streams transcription text and synthesized audio bytes back to the browser.
  3. Provides REST test endpoints (/api/chat and /api/health).
===============================================================================
"""

import os
import sys
import json
import base64
import asyncio
from pathlib import Path
from typing import AsyncIterator

# Ensure current directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Voice Agent Modules
from events import (
    VoiceAgentEvent,
    STTChunkEvent,
    STTOutputEvent,
    AgentChunkEvent,
    AgentCompleteEvent,
    TTSChunkEvent,
    StatusEvent,
)
from agent import voice_agent, agent_stream, SYSTEM_PROMPT, CURRENT_ORDER
from tts import CartesiaTTSClient, tts_stream
from stt import AssemblyAISTTClient, stt_stream
from langchain_core.messages import HumanMessage, AIMessage

# Load environment configuration
load_dotenv(override=True)

app = FastAPI(title="LangChain Voice Agent", version="1.0.0")

# Enable Cross-Origin Resource Sharing (CORS) for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web assets (HTML, CSS, JS, Visualizer)
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------------------------------------------------------
# 1. HTTP ROUTES
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    """Serve the main Voice Agent Web UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Voice Agent Server is Active</h1>")


@app.get("/architecture")
async def architecture_page():
    """Serve the Interactive System Architecture Visualizer for students."""
    visualizer_file = Path(__file__).resolve().parent / "architecture_visualizer.html"
    if visualizer_file.exists():
        return FileResponse(visualizer_file)
    return FileResponse(STATIC_DIR / "architecture.html")


@app.get("/code-visualizer")
async def code_visualizer_page():
    """Serve the Interactive Code Architecture & Function Visualizer."""
    code_vis_file = Path(__file__).resolve().parent / "code_architecture_visualizer.html"
    if code_vis_file.exists():
        return FileResponse(code_vis_file)
    return FileResponse(STATIC_DIR / "code_visualizer.html")


@app.get("/api/health")
async def health_check():
    """Diagnostic health check for configured API keys."""
    return {
        "status": "healthy",
        "llm_configured": bool(os.getenv("SENSENOVA_API_KEY")),
        "stt_configured": bool(os.getenv("ASSEMBLYAI_API_KEY")),
        "tts_configured": bool(os.getenv("CARTESIA_API_KEY")),
        "langsmith_configured": bool(os.getenv("LANGSMITH_API_KEY")),
        "project": os.getenv("LANGSMITH_PROJECT", "voice-agent"),
    }


@app.post("/api/chat")
async def direct_chat(request: Request):
    """
    Direct Text-to-Speech agent testing endpoint.
    Useful for testing the LLM & TTS without needing a microphone.
    """
    body = await request.json()
    prompt = body.get("text", "Hello!")
    thread_id = body.get("thread_id", "test-web-session")

    tts = CartesiaTTSClient()
    
    # Run agent
    config = {"configurable": {"thread_id": thread_id}}
    input_msg = {"messages": [HumanMessage(content=prompt)]}
    
    response = await voice_agent.ainvoke(input_msg, config=config)
    messages = response.get("messages", [])
    last_msg = messages[-1] if messages else None
    agent_text = last_msg.content if last_msg else "I couldn't process that."
    if not isinstance(agent_text, str):
        agent_text = str(agent_text)

    # Synthesize audio with Cartesia
    audio_bytes = await tts.async_synthesize_text(agent_text)
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""

    return {
        "user_input": prompt,
        "agent_response": agent_text,
        "audio_base64": b64_audio,
        "audio_format": "wav",
    }


# -----------------------------------------------------------------------------
# 2. REAL-TIME WEBSOCKET ENDPOINT (/ws)
# -----------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_voice_endpoint(websocket: WebSocket):
    """
    Bi-directional Real-Time WebSocket endpoint:
    - Receives PCM audio chunks (16kHz 16-bit Mono) from client browser.
    - Feeds through STT ➔ LangChain Agent ➔ Cartesia TTS.
    - Streams live transcripts, agent text tokens, and synthesized audio chunks back.
    """
    await websocket.accept()
    thread_id = f"session_{id(websocket)}"
    
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    is_active = True

    # Helper: Send JSON event message to browser safely
    async def send_json(data: dict):
        if is_active:
            try:
                await websocket.send_text(json.dumps(data))
            except Exception:
                pass

    # Helper: Send binary audio chunks (WAV bytes) to browser safely
    async def send_bytes(data: bytes):
        if is_active:
            try:
                await websocket.send_bytes(data)
            except Exception:
                pass

    # Greet client with connected status
    await send_json({
        "type": "status",
        "status": "idle",
        "message": "Connected to Voice Agent. Ready for speech."
    })

    # Async generator reading audio from client queue
    async def client_audio_stream() -> AsyncIterator[bytes]:
        while is_active:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    # Background runner task executing the Sandwich Pipeline
    async def run_pipeline():
        nonlocal is_active
        try:
            # 1. STT Stream (Ears)
            stt = stt_stream(client_audio_stream(), sample_rate=16000)
            
            # 2. Agent Stream (Brain)
            agent_events = agent_stream(stt, thread_id=thread_id)

            # 3. TTS Stream (Mouth)
            pipeline_output = tts_stream(agent_events)

            # Forward output events to the browser
            async for event in pipeline_output:
                if not is_active:
                    break
                event_type = getattr(event, "type", None)

                if event_type == "tts_chunk":
                    # Send binary audio chunk to browser speaker queue
                    audio = getattr(event, "audio", b"")
                    if audio:
                        await send_bytes(audio)
                else:
                    # Send JSON event metadata (live transcript preview, agent tokens, status)
                    await send_json(event.to_dict())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Pipeline Exception]: {e}")
            await send_json({"type": "status", "status": "error", "message": str(e)})

    pipeline_task = asyncio.create_task(run_pipeline())

    # Listen for incoming client messages
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                # Incoming audio chunk (PCM) from microphone
                await audio_queue.put(message["bytes"])

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")

                    if msg_type == "text_input":
                        # Text test mode (Simulates speech input)
                        text = payload.get("text", "").strip()
                        if text:
                            await send_json({"type": "stt_output", "transcript": text})
                            await send_json({"type": "status", "status": "thinking", "message": "Agent thinking..."})
                            
                            tts_client = CartesiaTTSClient()
                            config = {"configurable": {"thread_id": thread_id}}
                            input_msg = {"messages": [HumanMessage(content=text)]}

                            full_response_parts = []
                            async for chunk, meta in voice_agent.astream(
                                input_msg, config=config, stream_mode="messages"
                            ):
                                if isinstance(chunk, AIMessage) and chunk.content:
                                    token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                                    if token:
                                        full_response_parts.append(token)
                                        await send_json({"type": "agent_chunk", "text": token})

                            full_res = "".join(full_response_parts).strip()
                            await send_json({"type": "agent_complete", "response": full_res})
                            
                            # Synthesize TTS and send audio
                            await send_json({"type": "status", "status": "speaking", "message": "Speaking response..."})
                            audio = await tts_client.async_synthesize_text(full_res)
                            if audio:
                                await send_bytes(audio)
                            await send_json({"type": "status", "status": "idle", "message": "Ready for next request."})

                    elif msg_type == "stop":
                        await send_json({"type": "status", "status": "idle", "message": "Stopped."})

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        is_active = False
        await audio_queue.put(None)
        pipeline_task.cancel()
        with asyncio.CancelledError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
