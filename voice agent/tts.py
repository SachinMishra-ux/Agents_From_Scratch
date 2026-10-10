"""
===============================================================================
🗣️ TEXT-TO-SPEECH (TTS) MODULE — Cartesia Sonic-3 Streaming Audio Generation
===============================================================================
This module handles the "mouth" of our Voice Agent:

  1. Receives streamed text tokens (agent_chunk) from the LangChain Agent.
  2. Buffers text until natural sentence boundaries (. / ! / ?) are formed.
  3. Uses Cartesia's Sonic-3 ultra-fast voice model to synthesize realistic audio.
  4. Yields TTSChunkEvent (WAV audio bytes) for real-time browser speaker playback.
===============================================================================
"""

from __future__ import annotations
import os
import asyncio
import io
import re
from typing import AsyncIterator, Optional, Any
from dotenv import load_dotenv
from cartesia import Cartesia

# Load environment variables
load_dotenv(override=True)


class CartesiaTTSClient:
    """
    Manages text synthesis using Cartesia's high-speed, realistic TTS engine.
    
    Students: Cartesia's Sonic-3 model achieves sub-150ms time-to-first-audio,
    making voice conversation feel completely immediate and natural!
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        model_id: str = "sonic-3",
        sample_rate: int = 44100,
    ):
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY is not set in environment.")
        
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.client = Cartesia(api_key=self.api_key)

    def synthesize_text_to_wav(self, text: str) -> bytes:
        """
        Synthesize text into WAV audio bytes.
        
        Cleans accidental markdown characters so the voice model speaks cleanly.
        """
        if not text or not text.strip():
            return b""
        
        # Remove any stray asterisks, hashtags, or bracket symbols
        cleaned_text = re.sub(r"[*#_`\[\]]", "", text).strip()
        if not cleaned_text:
            return b""

        try:
            response = self.client.tts.generate(
                model_id=self.model_id,
                transcript=cleaned_text,
                voice=self.voice_id,
                output_format={
                    "container": "wav",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
                generation_config={
                    "speed": 1.0,
                    "volume": 1.0,
                    "emotion": "calm",
                },
            )
            # Read complete audio byte stream
            audio_bytes = response.read()
            return audio_bytes
        except Exception as err:
            print(f"[Cartesia TTS Error]: {err}")
            return b""

    async def async_synthesize_text(self, text: str) -> bytes:
        """
        Asynchronously synthesize text on a background thread so the
        FastAPI async event loop remains non-blocking and responsive.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.synthesize_text_to_wav, text)


async def tts_stream(
    event_stream: AsyncIterator[Any],
    tts_client: Optional[CartesiaTTSClient] = None,
) -> AsyncIterator[Any]:
    """
    Transform stream: Voice Events ➔ Voice Events (with synthesized Audio).
    
    Streaming Optimization for Students:
    Instead of waiting for the AI agent to finish its entire paragraph,
    we buffer words until a punctuation mark (., !, ?) is reached.
    The moment the first sentence is ready, Cartesia starts speaking immediately!
    """
    from events import TTSChunkEvent, StatusEvent

    if tts_client is None:
        tts_client = CartesiaTTSClient()

    text_buffer = ""

    async for event in event_stream:
        # Pass through all upstream events unchanged
        yield event

        event_type = getattr(event, "type", None)

        if event_type == "agent_chunk":
            token = getattr(event, "text", "")
            text_buffer += token

            # Check if we have formed a complete sentence boundary
            match = re.search(r"([.!?])\s+", text_buffer)
            if match and len(text_buffer) > 20:
                split_idx = match.end()
                sentence = text_buffer[:split_idx].strip()
                text_buffer = text_buffer[split_idx:]

                if sentence:
                    yield StatusEvent.create(status="speaking", message="Synthesizing speech...")
                    audio = await tts_client.async_synthesize_text(sentence)
                    if audio:
                        yield TTSChunkEvent.create(
                            audio=audio,
                            format="wav",
                            sample_rate=tts_client.sample_rate,
                        )

        elif event_type == "agent_complete":
            # Synthesize any remaining text in the buffer when the turn finishes
            remaining_text = text_buffer.strip() or getattr(event, "response", "").strip()
            text_buffer = ""
            if remaining_text:
                yield StatusEvent.create(status="speaking", message="Playing response...")
                audio = await tts_client.async_synthesize_text(remaining_text)
                if audio:
                    yield TTSChunkEvent.create(
                        audio=audio,
                        format="wav",
                        sample_rate=tts_client.sample_rate,
                    )
            yield StatusEvent.create(status="idle", message="Ready for next request.")
