"""
===============================================================================
🥪 PIPELINE MODULE — Composing the Streaming Sandwich Architecture
===============================================================================
This module chains our three components together using LangChain's
RunnableGenerator abstraction:

  RunnableGenerator(stt_stream)       # Audio Bytes ➔ STT Events
  | RunnableGenerator(agent_stream)   # STT Events ➔ Agent Thought Tokens
  | RunnableGenerator(tts_stream)     # Agent Tokens ➔ Synthesized Audio

Students: The pipe operator (|) connects the output stream of each stage
directly into the input stream of the next stage, creating a single,
continuous, zero-buffering data pipeline!
===============================================================================
"""

from __future__ import annotations
import asyncio
from typing import AsyncIterator, Any
from langchain_core.runnables import RunnableGenerator

from events import VoiceAgentEvent
from stt import stt_stream
from agent import agent_stream
from tts import tts_stream


def build_voice_pipeline():
    """
    Construct the end-to-end Voice Agent Pipeline using LangChain RunnableGenerators:
    Audio (Bytes) ➔ STT (Events) ➔ Agent (Events) ➔ TTS (Audio + Events)
    """
    return (
        RunnableGenerator(stt_stream)
        | RunnableGenerator(agent_stream)
        | RunnableGenerator(tts_stream)
    )


async def run_voice_pipeline(
    audio_stream: AsyncIterator[bytes],
    thread_id: str = "voice-session-default",
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Convenience function to execute an incoming client audio byte stream
    through the STT ➔ Agent ➔ TTS Sandwich pipeline.
    """
    # 1. Feed audio into STT stream
    stt_events = stt_stream(audio_stream)

    # 2. Feed STT events into LangChain Agent stream
    agent_events = agent_stream(stt_events, thread_id=thread_id)

    # 3. Feed Agent events into Cartesia TTS stream
    output_events = tts_stream(agent_events)

    async for event in output_events:
        yield event
