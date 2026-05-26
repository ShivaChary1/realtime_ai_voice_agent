"""
backend/controllers/audio_buffer.py

Per-session audio buffer manager.
Responsibilities:
  - Accept PCM chunks and append to session buffer
  - Handle out-of-order chunks via reorder buffer
  - Flush buffer when speechEnd is signalled
  - Guard against buffer overflow (>30s)
  - Add silence padding before/after audio
  - Report T3 timestamp to latency logger
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from config import get_settings
from services.latency.latency_logger import Stage, get_record

logger = logging.getLogger(__name__)
settings = get_settings()

# Callback type: called when buffer is ready to be sent to STT
# Signature: async def on_flush(session_id: str, turn_id: str, pcm_bytes: bytes)
FlushCallback = Callable[[str, str, bytes], None]


@dataclass
class ChunkMeta:
    """Metadata for a single audio chunk."""
    chunk_index: int
    data: bytes
    received_at: float = field(default_factory=time.perf_counter)


class SessionAudioBuffer:
    """
    Audio buffer for one active session.

    Handles:
    - In-order chunk appending
    - Out-of-order reordering (up to 500ms wait)
    - Overflow protection
    - Silence padding injection
    """

    REORDER_WAIT_MS = 50  # wait this long for missing chunk before skipping

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._buffer = bytearray()
        self._next_expected_index = 0
        self._reorder_buffer: Dict[int, ChunkMeta] = {}
        self._has_speech = False   # True once first speech chunk received
        self._lock = asyncio.Lock()

    async def append(self, chunk: ChunkMeta):
        """
        Append a chunk to the buffer, handling reordering.
        """
        async with self._lock:
            if chunk.chunk_index == self._next_expected_index:
                # In-order: append immediately
                self._buffer.extend(chunk.data)
                self._has_speech = True
                self._next_expected_index += 1

                # Drain reorder buffer for any subsequent waiting chunks
                while self._next_expected_index in self._reorder_buffer:
                    waiting = self._reorder_buffer.pop(self._next_expected_index)
                    self._buffer.extend(waiting.data)
                    self._next_expected_index += 1

            elif chunk.chunk_index > self._next_expected_index:
                # Out-of-order: hold in reorder buffer
                self._reorder_buffer[chunk.chunk_index] = chunk
                logger.debug(
                    f"[BUFFER] Out-of-order chunk: session={self.session_id} "
                    f"expected={self._next_expected_index} got={chunk.chunk_index}"
                )
                # Schedule a drain after wait window
                asyncio.create_task(
                    self._delayed_drain(chunk.chunk_index)
                )
            else:
                # Duplicate or old chunk — discard
                logger.debug(
                    f"[BUFFER] Discarding duplicate chunk: "
                    f"session={self.session_id} index={chunk.chunk_index}"
                )

    async def _delayed_drain(self, awaited_index: int):
        """
        If a chunk is still missing after REORDER_WAIT_MS, skip past it.
        """
        await asyncio.sleep(self.REORDER_WAIT_MS / 1000)
        async with self._lock:
            if self._next_expected_index == awaited_index:
                # Still waiting — skip and drain whatever we have
                logger.warning(
                    f"[BUFFER] Skipping missing chunk: "
                    f"session={self.session_id} index={awaited_index}"
                )
                self._next_expected_index = awaited_index + 1
                while self._next_expected_index in self._reorder_buffer:
                    waiting = self._reorder_buffer.pop(self._next_expected_index)
                    self._buffer.extend(waiting.data)
                    self._next_expected_index += 1

    def flush(self, turn_id: str) -> Optional[bytes]:
        """
        Flush buffer contents, add silence padding, reset state.
        Returns None if audio is too short to process.
        """
        raw = bytes(self._buffer)
        self._buffer.clear()
        self._reorder_buffer.clear()
        self._next_expected_index = 0
        self._has_speech = False

        if len(raw) < settings.min_audio_bytes:
            logger.debug(
                f"[BUFFER] Flush too short: session={self.session_id} "
                f"bytes={len(raw)} min={settings.min_audio_bytes}"
            )
            return None

        # Add silence padding (300ms before + after)
        padded = self._add_silence_padding(raw)

        # Mark T3 in latency record
        record = get_record(self.session_id, turn_id)
        if record:
            record.mark("buffer_flush")
            record.measure(Stage.AUDIO_CAPTURE, "speech_end_client", "buffer_flush")

        logger.info(
            f"[BUFFER] Flushed: session={self.session_id} "
            f"raw_bytes={len(raw)} padded_bytes={len(padded)}"
        )
        return padded

    def is_overflow(self) -> bool:
        """True if buffer has exceeded max allowed size."""
        return len(self._buffer) >= settings.max_buffer_bytes

    def size_bytes(self) -> int:
        return len(self._buffer)

    def _add_silence_padding(self, pcm_bytes: bytes) -> bytes:
        """
        Prepend and append silence (zeros) of settings.silence_padding_ms duration.
        This significantly improves Whisper accuracy.
        """
        silence = bytes(settings.silence_padding_bytes)
        return silence + pcm_bytes + silence


# ── Global buffer registry ────────────────────────────────────────────────────

class AudioBufferManager:
    """
    Registry of per-session audio buffers.
    Singleton — injected via app.state.
    """

    def __init__(self):
        self._buffers: Dict[str, SessionAudioBuffer] = {}
        self._flush_callback: Optional[FlushCallback] = None

    def set_flush_callback(self, callback: FlushCallback):
        """Register the function to call when audio is ready for STT."""
        self._flush_callback = callback

    def create(self, session_id: str) -> SessionAudioBuffer:
        """Create and register a new buffer for a session."""
        buf = SessionAudioBuffer(session_id)
        self._buffers[session_id] = buf
        logger.debug(f"[BUFFER] Buffer created: session={session_id}")
        return buf

    def get(self, session_id: str) -> Optional[SessionAudioBuffer]:
        return self._buffers.get(session_id)

    def destroy(self, session_id: str):
        """Remove buffer when session ends."""
        self._buffers.pop(session_id, None)
        logger.debug(f"[BUFFER] Buffer destroyed: session={session_id}")

    async def handle_chunk(self, session_id: str, chunk: ChunkMeta):
        """
        Main entry point for incoming audio chunks.
        Creates buffer if missing, appends chunk, checks overflow.
        """
        buf = self._buffers.get(session_id)
        if buf is None:
            buf = self.create(session_id)

        await buf.append(chunk)

        # Overflow protection
        if buf.is_overflow():
            logger.warning(
                f"[BUFFER] Overflow detected: session={session_id} "
                f"bytes={buf.size_bytes()} — forcing flush"
            )
            await self.flush(session_id, turn_id="overflow")

    async def flush(self, session_id: str, turn_id: str):
        """
        Flush a session's buffer and invoke the STT callback.
        """
        buf = self._buffers.get(session_id)
        if buf is None:
            return

        pcm_bytes = buf.flush(turn_id)
        if pcm_bytes is None:
            return  # too short, discard

        if self._flush_callback:
            await self._flush_callback(session_id, turn_id, pcm_bytes)
        else:
            logger.warning(
                f"[BUFFER] No flush callback set — audio discarded: "
                f"session={session_id}"
            )

    def discard(self, session_id: str):
        """
        Discard buffer without processing (used for barge-in).
        """
        buf = self._buffers.get(session_id)
        if buf:
            buf._buffer.clear()
            buf._reorder_buffer.clear()
            buf._next_expected_index = 0
            buf._has_speech = False
            logger.info(f"[BUFFER] Buffer discarded (barge-in): session={session_id}")