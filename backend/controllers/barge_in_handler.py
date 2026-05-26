"""
backend/controllers/barge_in_handler.py

Handles barge-in: the user starts speaking while the agent is responding.
When this happens we must:
  1. Immediately notify the frontend to stop audio playback
  2. Discard the current audio buffer
  3. Cancel any in-flight LLM/TTS tasks for this session
  4. Reset session state to "listening"
"""

import asyncio
import logging
from typing import Dict, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class BargeInHandler:
    """
    Manages barge-in events per session.

    Usage flow:
        1. Agent starts TTS → call set_agent_speaking(session_id, True)
        2. Client sends barge_in message → call handle(session_id)
        3. Agent finishes speaking → call set_agent_speaking(session_id, False)
    """

    def __init__(self):
        # Track whether agent is currently speaking per session
        self._agent_speaking: Dict[str, bool] = {}

        # Store active asyncio tasks per session so we can cancel them
        self._active_tasks: Dict[str, list[asyncio.Task]] = {}

        # WebSocket references to send stop-playback signal
        self._websockets: Dict[str, WebSocket] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register_websocket(self, session_id: str, ws: WebSocket):
        """Register WebSocket for a session (needed to send stop signal)."""
        self._websockets[session_id] = ws
        self._agent_speaking[session_id] = False
        self._active_tasks[session_id] = []

    def unregister(self, session_id: str):
        """Clean up when session ends."""
        self._agent_speaking.pop(session_id, None)
        self._active_tasks.pop(session_id, None)
        self._websockets.pop(session_id, None)

    # ── Agent speaking state ──────────────────────────────────────────────────

    def set_agent_speaking(self, session_id: str, is_speaking: bool):
        """Call this when agent starts/stops TTS playback."""
        self._agent_speaking[session_id] = is_speaking
        logger.debug(
            f"[BARGE-IN] session={session_id} agent_speaking={is_speaking}"
        )

    def is_agent_speaking(self, session_id: str) -> bool:
        return self._agent_speaking.get(session_id, False)

    # ── Task tracking ─────────────────────────────────────────────────────────

    def track_task(self, session_id: str, task: asyncio.Task):
        """Register a cancellable task (LLM call, TTS stream) for this session."""
        tasks = self._active_tasks.get(session_id, [])
        # Clean up completed tasks
        tasks = [t for t in tasks if not t.done()]
        tasks.append(task)
        self._active_tasks[session_id] = tasks

    # ── Handle barge-in ───────────────────────────────────────────────────────

    async def handle(
        self,
        session_id: str,
        audio_buffer_manager,   # AudioBufferManager (avoid circular import)
    ) -> bool:
        """
        Process a barge-in event.

        Returns:
            True if barge-in was active (agent was speaking)
            False if agent was not speaking (spurious barge-in signal)
        """
        if not self.is_agent_speaking(session_id):
            logger.debug(
                f"[BARGE-IN] Spurious barge-in (agent not speaking): "
                f"session={session_id}"
            )
            return False

        logger.info(f"[BARGE-IN] Handling barge-in: session={session_id}")

        # 1. Cancel in-flight tasks
        await self._cancel_active_tasks(session_id)

        # 2. Discard audio buffer
        audio_buffer_manager.discard(session_id)

        # 3. Mark agent as not speaking
        self.set_agent_speaking(session_id, False)

        # 4. Send stop-playback signal to frontend
        await self._send_stop_playback(session_id)

        logger.info(
            f"[BARGE-IN] Handled successfully: session={session_id}"
        )
        return True

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _cancel_active_tasks(self, session_id: str):
        """Cancel all in-flight async tasks for this session."""
        tasks = self._active_tasks.get(session_id, [])
        cancelled = 0
        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled += 1
        self._active_tasks[session_id] = []

        if cancelled:
            logger.info(
                f"[BARGE-IN] Cancelled {cancelled} active task(s): "
                f"session={session_id}"
            )

    async def _send_stop_playback(self, session_id: str):
        """Send a stop_playback control message to the frontend."""
        ws = self._websockets.get(session_id)
        if ws is None:
            return
        try:
            import json
            await ws.send_text(
                json.dumps({
                    "type": "stop_playback",
                    "session_id": session_id,
                })
            )
            logger.debug(
                f"[BARGE-IN] stop_playback sent: session={session_id}"
            )
        except Exception as e:
            logger.warning(
                f"[BARGE-IN] Failed to send stop_playback: "
                f"session={session_id} error={e}"
            )