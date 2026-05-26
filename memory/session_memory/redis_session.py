"""
memory/session_memory/redis_session.py

Manages all session state in Redis.
Each WebSocket connection has one session.
Sessions survive disconnects for a 2-minute reconnect window.
"""

import json
import logging
from enum import Enum
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SessionStatus(str, Enum):
    ACTIVE       = "active"
    DISCONNECTED = "disconnected"
    ENDED        = "ended"


class RedisSessionManager:
    """
    Async Redis-backed session store.

    Key schema:
        session:{session_id}  →  JSON blob with full session state
    """

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    # ── Connection lifecycle ───────────────────────────────────────────────────

    async def connect(self):
        """Initialize the Redis connection pool."""
        self._client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        # Verify connection
        await self._client.ping()
        logger.info(
            f"✅ Redis connected at {settings.redis_host}:{settings.redis_port}"
        )

    async def disconnect(self):
        """Close the Redis connection pool."""
        if self._client:
            await self._client.aclose()
            logger.info("Redis connection closed.")

    # ── Session CRUD ──────────────────────────────────────────────────────────

    async def create_session(self, session_id: str, patient_id: Optional[str] = None) -> Dict:
        """
        Create a new session record.
        Returns the created session dict.
        """
        session = {
            "session_id":           session_id,
            "status":               SessionStatus.ACTIVE,
            "patient_id":           patient_id,
            "language":             "en",           # updated after first detection
            "conversation_history": [],
            "current_turn_id":      None,
            "created_at":           self._now(),
            "last_active":          self._now(),
        }
        await self._set(session_id, session, ttl=settings.session_ttl_seconds)
        logger.info(f"[SESSION] Created session_id={session_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Fetch session by ID. Returns None if not found."""
        raw = await self._client.get(self._key(session_id))
        if raw is None:
            logger.debug(f"[SESSION] Not found: session_id={session_id}")
            return None
        return json.loads(raw)

    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> Optional[Dict]:
        """
        Partial update — merges updates into existing session.
        Also refreshes the TTL to keep the session alive.
        """
        session = await self.get_session(session_id)
        if session is None:
            logger.warning(f"[SESSION] Update failed — not found: session_id={session_id}")
            return None

        session.update(updates)
        session["last_active"] = self._now()
        await self._set(session_id, session, ttl=settings.session_ttl_seconds)
        return session

    async def mark_disconnected(self, session_id: str):
        """
        Called when WebSocket closes.
        Reduces TTL to reconnect window — session survives for 2 minutes.
        """
        session = await self.get_session(session_id)
        if session:
            session["status"] = SessionStatus.DISCONNECTED
            session["last_active"] = self._now()
            await self._set(
                session_id, session, ttl=settings.disconnect_ttl_seconds
            )
            logger.info(
                f"[SESSION] Marked disconnected: session_id={session_id} "
                f"ttl={settings.disconnect_ttl_seconds}s"
            )

    async def mark_active(self, session_id: str):
        """Called on reconnect — restore full TTL."""
        await self.update_session(session_id, {"status": SessionStatus.ACTIVE})
        logger.info(f"[SESSION] Restored to active: session_id={session_id}")

    async def end_session(self, session_id: str):
        """Permanently delete session."""
        await self._client.delete(self._key(session_id))
        logger.info(f"[SESSION] Ended and deleted: session_id={session_id}")

    async def session_exists(self, session_id: str) -> bool:
        return await self._client.exists(self._key(session_id)) == 1

    # ── Conversation history helpers ──────────────────────────────────────────

    async def append_turn(self, session_id: str, role: str, content: str):
        """
        Append one message to the conversation history.
        role: 'user' | 'assistant'
        """
        session = await self.get_session(session_id)
        if session is None:
            return
        session["conversation_history"].append({"role": role, "content": content})
        # Keep last 20 turns to prevent context bloat
        session["conversation_history"] = session["conversation_history"][-20:]
        await self._set(session_id, session, ttl=settings.session_ttl_seconds)

    async def get_history(self, session_id: str):
        """Return conversation history for this session."""
        session = await self.get_session(session_id)
        return session.get("conversation_history", []) if session else []

    # ── Language update ───────────────────────────────────────────────────────

    async def set_language(self, session_id: str, language: str):
        """Update detected language for session."""
        await self.update_session(session_id, {"language": language})

    # ── Private helpers ───────────────────────────────────────────────────────

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    async def _set(self, session_id: str, data: Dict, ttl: int):
        await self._client.setex(
            self._key(session_id),
            ttl,
            json.dumps(data),
        )

    @staticmethod
    def _now() -> float:
        import time
        return time.time()