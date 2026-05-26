"""
services/vad_service.py

Server-side Voice Activity Detection (VAD) using webrtcvad.
This is a FALLBACK — primary VAD runs client-side.
Used when:
  - Client VAD fails to send speechEnd signal
  - 8 seconds pass without a speechEnd from client
  - Debug/test mode without a frontend
"""

import logging
from collections import deque
from typing import Tuple

import webrtcvad

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VADService:
    """
    Wraps webrtcvad for per-session voice activity detection.

    Usage:
        vad = VADService()
        is_speech, should_flush = vad.process_chunk(session_id, pcm_bytes)
        if should_flush:
            trigger_stt(buffer)
    """

    # webrtcvad requires frame sizes of exactly 10ms, 20ms, or 30ms
    VALID_FRAME_MS = (10, 20, 30)

    def __init__(self):
        frame_ms = settings.vad_frame_ms
        if frame_ms not in self.VALID_FRAME_MS:
            raise ValueError(
                f"VAD frame size must be one of {self.VALID_FRAME_MS}, got {frame_ms}"
            )

        self._vad = webrtcvad.Vad(settings.vad_mode)
        self._frame_ms = frame_ms
        self._frame_bytes = self._calculate_frame_bytes(frame_ms)

        # Per-session state: tracks consecutive silent frames
        # key: session_id → deque of bool (True=speech, False=silence)
        self._session_frames: dict[str, deque] = {}

        logger.info(
            f"VADService initialized: mode={settings.vad_mode} "
            f"frame_ms={frame_ms} frame_bytes={self._frame_bytes}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def register_session(self, session_id: str):
        """Call this when a new WebSocket session starts."""
        self._session_frames[session_id] = deque(
            maxlen=settings.vad_silence_frames * 2
        )
        logger.debug(f"[VAD] Session registered: {session_id}")

    def unregister_session(self, session_id: str):
        """Call this when session ends to free memory."""
        self._session_frames.pop(session_id, None)
        logger.debug(f"[VAD] Session unregistered: {session_id}")

    def process_chunk(
        self, session_id: str, pcm_bytes: bytes
    ) -> Tuple[bool, bool]:
        """
        Process one PCM audio chunk.

        Returns:
            (is_speech: bool, should_flush: bool)
            should_flush=True means silence threshold reached → trigger STT
        """
        if session_id not in self._session_frames:
            self.register_session(session_id)

        # Split incoming bytes into valid VAD frames
        frames = self._split_into_frames(pcm_bytes)
        if not frames:
            return False, False

        # Evaluate each frame
        is_speech = False
        for frame in frames:
            try:
                frame_is_speech = self._vad.is_speech(frame, settings.audio_sample_rate)
                self._session_frames[session_id].append(frame_is_speech)
                if frame_is_speech:
                    is_speech = True
            except Exception as e:
                logger.warning(f"[VAD] Frame processing error: {e}")
                continue

        # Check if we've hit the silence threshold
        should_flush = self._check_silence_threshold(session_id)

        return is_speech, should_flush

    def reset_session(self, session_id: str):
        """Reset frame history for a session (called after flush)."""
        if session_id in self._session_frames:
            self._session_frames[session_id].clear()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_silence_threshold(self, session_id: str) -> bool:
        """
        Returns True if the last N consecutive frames are all silent.
        N = settings.vad_silence_frames
        """
        frames = self._session_frames.get(session_id)
        if not frames or len(frames) < settings.vad_silence_frames:
            return False

        # Check the last N frames
        recent = list(frames)[-settings.vad_silence_frames:]
        all_silent = not any(recent)  # True if all are False (silent)

        # Only flush if there was speech before the silence
        has_prior_speech = any(list(frames)[:-settings.vad_silence_frames])

        return all_silent and has_prior_speech

    def _split_into_frames(self, pcm_bytes: bytes) -> list[bytes]:
        """Split raw PCM bytes into fixed-size VAD frames."""
        frames = []
        offset = 0
        while offset + self._frame_bytes <= len(pcm_bytes):
            frames.append(pcm_bytes[offset: offset + self._frame_bytes])
            offset += self._frame_bytes
        return frames

    def _calculate_frame_bytes(self, frame_ms: int) -> int:
        """
        Calculate bytes per frame.
        PCM 16-bit mono: bytes = (sample_rate / 1000) * frame_ms * 2
        """
        samples_per_ms = settings.audio_sample_rate // 1000
        return samples_per_ms * frame_ms * 2  # *2 for 16-bit