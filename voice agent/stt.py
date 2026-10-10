"""
===============================================================================
👂 SPEECH-TO-TEXT (STT) MODULE — AssemblyAI v3 Real-Time Streaming & VAD
===============================================================================
This module handles the "ears" of our Voice Agent:

  1. Receives raw audio PCM bytes (16kHz 16-bit Mono) from the client browser.
  2. Streams the audio to AssemblyAI's v3 streaming WebSocket endpoint.
  3. Uses Voice Activity Detection (VAD) to track user pauses and speech boundaries.
  4. Emits two types of events:
     - STTChunkEvent: Partial words (for live preview on screen).
     - STTOutputEvent: Finalized sentence (triggers the LangChain Agent ONLY when end_of_turn=True).
===============================================================================
"""

from __future__ import annotations
import os
import asyncio
from typing import AsyncIterator, Optional, Any
from dotenv import load_dotenv
import assemblyai.streaming.v3 as aai_v3

# Load API keys from .env
load_dotenv(override=True)


class AssemblyAISTTClient:
    """
    Manages the persistent Real-Time WebSocket connection to AssemblyAI v3.
    Uses a thread-safe bridge to deliver speech events from AssemblyAI's background
    thread directly into Python's asyncio event loop.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        # Retrieve and validate API key
        self.api_key = str(api_key or os.getenv("ASSEMBLYAI_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY is not set.")
        
        self.sample_rate = sample_rate
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.loop = asyncio.get_running_loop()
        self._connected = False

        # Initialize the official AssemblyAI v3 streaming client
        self.client = aai_v3.RealTimeTranscriber(api_key=self.api_key)
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Register callbacks for AssemblyAI real-time events."""
        from events import STTChunkEvent, STTOutputEvent, StatusEvent

        # 1. Connection established
        def on_begin(client, event: aai_v3.BeginEvent):
            print(f"[AssemblyAI] Session started: {event.id} (VAD & End-of-Turn active)")
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait,
                StatusEvent.create(status="listening", message="Listening for complete sentences...")
            )

        # 2. Voice Activity Detected (User started speaking)
        def on_speech_started(client, event: aai_v3.SpeechStartedEvent):
            print("[AssemblyAI VAD] Speech started...")
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait,
                StatusEvent.create(status="listening", message="User speaking...")
            )

        # 3. Transcript Turn Received
        def on_turn(client, event: aai_v3.TurnEvent):
            transcript = (event.transcript or "").strip()
            if not transcript:
                return

            print(f"[AssemblyAI Turn] (end_of_turn={event.end_of_turn}, formatted={event.turn_is_formatted}): {transcript}")

            # CRITICAL CONCEPT FOR STUDENTS:
            # AssemblyAI emits partial updates while you talk (end_of_turn=False).
            # We ONLY trigger the AI Agent (STTOutputEvent) when end_of_turn is True!
            # Otherwise, the agent would interrupt you on every single word.
            if event.end_of_turn:
                print(f"[AssemblyAI FINAL TURN]: Triggering agent with -> '{transcript}'")
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait,
                    STTOutputEvent.create(transcript)
                )
            else:
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait,
                    STTChunkEvent.create(transcript)
                )

        # 4. Error handling
        def on_error(client, error: aai_v3.RealTimeError):
            print(f"[AssemblyAI Error]: {error}")
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait,
                StatusEvent.create(status="error", message=str(error))
            )

        # 5. Session termination
        def on_terminated(client, event: aai_v3.TerminationEvent):
            print(f"[AssemblyAI] Session terminated: {event.audio_duration_seconds}s processed")

        # Bind event listeners
        self.client.on(aai_v3.RealTimeEvents.Begin, on_begin)
        self.client.on(aai_v3.RealTimeEvents.SpeechStarted, on_speech_started)
        self.client.on(aai_v3.RealTimeEvents.Turn, on_turn)
        self.client.on(aai_v3.RealTimeEvents.Error, on_error)
        self.client.on(aai_v3.RealTimeEvents.Termination, on_terminated)

    def connect(self):
        """
        Connect to AssemblyAI Real-Time WebSocket.
        Configures VAD silence thresholds so natural pauses don't cut off speech early.
        """
        if not self._connected:
            self.client.connect(
                aai_v3.RealTimeParameters(
                    sample_rate=self.sample_rate,
                    encoding=aai_v3.Encoding.pcm_s16le,
                    format_turns=True,
                    min_turn_silence=700,   # 700ms silence needed to detect end of sentence
                    max_turn_silence=2000,  # 2.0s maximum turn silence window
                    end_of_turn_confidence_threshold=0.4,
                )
            )
            self._connected = True

    def send_audio(self, audio_chunk: bytes) -> None:
        """Stream raw 16-bit PCM audio bytes to AssemblyAI."""
        if not self._connected:
            self.connect()
        if audio_chunk:
            self.client.stream(audio_chunk)

    async def receive_events(self) -> AsyncIterator[Any]:
        """Asynchronously yield transcription events as they arrive."""
        while True:
            item = await self.queue.get()
            if item is None:
                break
            yield item

    def close(self) -> None:
        """Disconnect session gracefully and release resources."""
        if self._connected:
            try:
                self.client.disconnect(terminate=True)
            except Exception as e:
                print(f"[AssemblyAI Disconnect Error]: {e}")
            finally:
                self._connected = False
                self.loop.call_soon_threadsafe(self.queue.put_nowait, None)


async def stt_stream(
    audio_stream: AsyncIterator[bytes],
    sample_rate: int = 16000,
) -> AsyncIterator[Any]:
    """
    Transform stream: Audio Bytes ➔ STT Events (Producer-Consumer pattern).
    
    1. Producer Task: pumps audio bytes from the client WebSocket into AssemblyAI.
    2. Consumer Task: yields STTChunkEvent and STTOutputEvent as speech is recognized.
    """
    stt = AssemblyAISTTClient(sample_rate=sample_rate)
    stt.connect()

    async def _send_pump():
        try:
            async for chunk in audio_stream:
                if chunk:
                    stt.send_audio(chunk)
        except Exception as e:
            print(f"[STT Audio Pump Error]: {e}")
        finally:
            stt.close()

    send_task = asyncio.create_task(_send_pump())

    try:
        async for event in stt.receive_events():
            yield event
    finally:
        send_task.cancel()
        with asyncio.CancelledError:
            pass
        stt.close()
