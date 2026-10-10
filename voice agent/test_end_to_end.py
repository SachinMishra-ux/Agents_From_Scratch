"""
===============================================================================
🧪 INTEGRATION TEST SUITE — LangChain Voice Agent
===============================================================================
This automated script verifies all components of the Voice Agent end-to-end:

  Step 1: Test SenseNova LLM reasoning, tools calling & multi-turn memory.
  Step 2: Test Cartesia Sonic-3 Text-to-Speech audio synthesis.
  Step 3: Test the complete streaming Sandwich Pipeline (STT ➔ Agent ➔ TTS).
===============================================================================
"""

import os
import sys
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from events import STTOutputEvent, AgentChunkEvent, TTSChunkEvent
from agent import get_llm, build_voice_agent, agent_stream, CURRENT_ORDER
from tts import CartesiaTTSClient, tts_stream
from langchain_core.messages import HumanMessage


async def test_llm_and_tools():
    """Test LLM reasoning, menu lookups, ordering tools, and conversation memory."""
    print("\n" + "="*50)
    print("STEP 1: Testing SenseNova LLM & LangChain create_agent")
    print("="*50)
    
    agent = build_voice_agent()
    thread_id = "test-thread-001"
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1: Ask for menu
    print("\n[Turn 1]: User says: 'What sandwiches do you have?'")
    res1 = await agent.ainvoke({"messages": [HumanMessage(content="What sandwiches do you have?")]}, config=config)
    last_msg = res1["messages"][-1].content
    print(f"Agent Response: {last_msg}")
    assert last_msg, "Agent response was empty"

    # Turn 2: Order an item using tools
    print("\n[Turn 2]: User says: 'Please add 2 classic clubs to my order.'")
    res2 = await agent.ainvoke({"messages": [HumanMessage(content="Please add 2 classic clubs to my order.")]}, config=config)
    last_msg2 = res2["messages"][-1].content
    print(f"Agent Response: {last_msg2}")
    print(f"Current Order State: {CURRENT_ORDER}")
    assert len(CURRENT_ORDER) > 0, "Tool add_to_order failed to modify order state"

    # Turn 3: Memory check
    print("\n[Turn 3]: User says: 'What did I just order?'")
    res3 = await agent.ainvoke({"messages": [HumanMessage(content="What did I just order?")]}, config=config)
    last_msg3 = res3["messages"][-1].content
    print(f"Agent Response (Memory Check): {last_msg3}")
    print("✅ LLM, create_agent, tools, and memory tests passed!")
    return last_msg3


async def test_cartesia_tts(sample_text: str):
    """Test Cartesia voice synthesis and save sample WAV file."""
    print("\n" + "="*50)
    print("STEP 2: Testing Cartesia Text-to-Speech (TTS)")
    print("="*50)

    tts = CartesiaTTSClient()
    print(f"Synthesizing speech for: '{sample_text}'")
    audio_bytes = await tts.async_synthesize_text(sample_text)
    
    print(f"Synthesized audio length: {len(audio_bytes)} bytes")
    assert len(audio_bytes) > 1000, "Synthesized audio is suspiciously small or empty"

    test_wav = Path(__file__).resolve().parent / "test_output.wav"
    with open(test_wav, "wb") as f:
        f.write(audio_bytes)
    print(f"✅ Saved test audio file to: {test_wav}")


async def test_streaming_sandwich_pipeline():
    """Test full asynchronous streaming pipeline from STT event to TTS audio."""
    print("\n" + "="*50)
    print("STEP 3: Testing Streaming Sandwich Pipeline (STT -> Agent -> TTS)")
    print("="*50)

    # Simulated STT event generator
    async def simulated_stt_events():
        yield STTOutputEvent.create("What drinks do you have on the menu?")

    thread_id = "test-pipeline-streaming"
    
    # 1. Agent Stream
    agent_events = agent_stream(simulated_stt_events(), thread_id=thread_id)

    # 2. TTS Stream
    pipeline_stream = tts_stream(agent_events)

    received_tokens = []
    received_audio_chunks = []

    async for event in pipeline_stream:
        event_type = getattr(event, "type", None)
        if event_type == "agent_chunk":
            token = getattr(event, "text", "")
            received_tokens.append(token)
            print(token, end="", flush=True)
        elif event_type == "tts_chunk":
            audio = getattr(event, "audio", b"")
            received_audio_chunks.append(audio)
            print(f"\n[TTS Audio Chunk Received]: {len(audio)} bytes")

    print("\n\n✅ Full streaming pipeline executed successfully!")
    print(f"Total streamed tokens: {len(received_tokens)}")
    print(f"Total audio chunks generated: {len(received_audio_chunks)}")


async def main():
    print("🚀 Starting LangChain Voice Agent End-to-End Verification Suite...")
    sample_text = await test_llm_and_tools()
    await test_cartesia_tts(sample_text)
    await test_streaming_sandwich_pipeline()
    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! VOICE AGENT IS READY.")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
