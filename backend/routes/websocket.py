"""
backend/routes/websocket.py

WebSocket endpoint — the main real-time communication channel.

Handles:
  - New connections and session creation/resumption
  - Binary audio frames → AudioBufferManager
  - JSON control messages → appropriate handlers
  - Disconnection and reconnect window management
  - Per-message latency timestamping
"""

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

from backend.api.ws_models import (
    InboundMessageType,
    OutboundMessageType,
    parse_inbound_message,
    SessionAckMessage,
    ErrorMessage,
    PongMessage,
    LatencyReportMessage,
)
from backend.controllers.audio_buffer import AudioBufferManager, ChunkMeta
from backend.controllers.barge_in_handler import BargeInHandler
from memory.session_memory.redis_session import RedisSessionManager, SessionStatus
from services.latency.latency_logger import Stage, start_turn, get_record, end_turn
from services.vad_service import VADService

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """
    Tracks all active WebSocket connections.
    Used for broadcasting and checking connection state.
    """

    def __init__(self):
        # session_id → WebSocket
        self._connections: dict[str, WebSocket] = {}

    def add(self, session_id: str, ws: WebSocket):
        self._connections[session_id] = ws

    def remove(self, session_id: str):
        self._connections.pop(session_id, None)

    def get(self, session_id: str) -> Optional[WebSocket]:
        return self._connections.get(session_id)

    @property
    def count(self) -> int:
        return len(self._connections)


# Shared connection manager (module-level singleton)
connection_manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    request: Request,
):
    """
    Main WebSocket handler.

    URL: ws://host:8000/ws/{session_id}

    Message flow:
        Binary frame  → audio chunk → AudioBufferManager
        JSON frame    → parse type → route to handler
    """
    # Pull shared services from app state
    session_manager: RedisSessionManager  = request.app.state.session_manager
    buffer_manager: AudioBufferManager    = request.app.state.buffer_manager
    barge_in_handler: BargeInHandler      = request.app.state.barge_in_handler
    vad_service: VADService               = request.app.state.vad_service

    await websocket.accept()
    logger.info(
        f"[WS] Connection accepted: session_id={session_id} "
        f"active_connections={connection_manager.count + 1}"
    )

    # ── Session setup ─────────────────────────────────────────────────────────
    session = await session_manager.get_session(session_id)
    is_resuming = False

    if session is None:
        # Brand new session
        session = await session_manager.create_session(session_id)
        status_msg = "created"
        logger.info(f"[WS] New session: session_id={session_id}")
    elif session["status"] == SessionStatus.DISCONNECTED:
        # Reconnect within window — restore
        await session_manager.mark_active(session_id)
        status_msg = "resumed"
        is_resuming = True
        logger.info(f"[WS] Session resumed: session_id={session_id}")
    else:
        # Duplicate connection — close old, accept new
        old_ws = connection_manager.get(session_id)
        if old_ws:
            await old_ws.close(code=1001, reason="Replaced by new connection")
        status_msg = "replaced"

    # Register with managers
    connection_manager.add(session_id, websocket)
    buffer_manager.create(session_id)
    barge_in_handler.register_websocket(session_id, websocket)
    vad_service.register_session(session_id)

    # Send session acknowledgement
    ack = SessionAckMessage(
        session_id=session_id,
        status=status_msg,
        timestamp=time.time(),
    )
    await websocket.send_text(ack.model_dump_json())

    # ── Message loop ──────────────────────────────────────────────────────────
    current_turn_id: Optional[str] = None

    try:
        while True:
            # Receive next frame (binary or text)
            message = await websocket.receive()

            # ── Binary frame = audio chunk ─────────────────────────────────
            if "bytes" in message and message["bytes"]:
                pcm_bytes = message["bytes"]

                # If no turn is active, start one
                if current_turn_id is None:
                    current_turn_id = str(uuid.uuid4())
                    latency_record = start_turn(session_id, current_turn_id)
                    latency_record.mark("first_audio_chunk")

                # Wrap in ChunkMeta — we derive chunk_index from sequential count
                # (real chunk_index comes from JSON metadata frame, see below)
                chunk = ChunkMeta(
                    chunk_index=_next_chunk_index(session_id),
                    data=pcm_bytes,
                )
                await buffer_manager.handle_chunk(session_id, chunk)

                # Run server-side VAD as fallback
                is_speech, should_flush = vad_service.process_chunk(
                    session_id, pcm_bytes
                )
                if should_flush:
                    logger.info(
                        f"[WS] Server VAD triggered flush: session={session_id}"
                    )
                    await _trigger_flush(
                        session_id, current_turn_id,
                        buffer_manager, session_manager, websocket
                    )
                    current_turn_id = None
                    vad_service.reset_session(session_id)

                # Refresh session TTL
                await session_manager.update_session(session_id, {})

            # ── Text frame = JSON control message ──────────────────────────
            elif "text" in message and message["text"]:
                await _handle_text_message(
                    raw_text=message["text"],
                    session_id=session_id,
                    current_turn_id=current_turn_id,
                    websocket=websocket,
                    session_manager=session_manager,
                    buffer_manager=buffer_manager,
                    barge_in_handler=barge_in_handler,
                    vad_service=vad_service,
                )

                # If speechEnd was handled, reset turn
                # (actual reset happens inside handler, returned via side-effect)
                # We track it separately below via a flag mechanism if needed

    except WebSocketDisconnect as e:
        logger.info(
            f"[WS] Disconnected: session_id={session_id} code={e.code}"
        )
    except Exception as e:
        logger.error(
            f"[WS] Unexpected error: session_id={session_id} error={e}",
            exc_info=True,
        )
        try:
            err = ErrorMessage(
                session_id=session_id,
                code="internal_error",
                message="An internal error occurred.",
            )
            await websocket.send_text(err.model_dump_json())
        except Exception:
            pass
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        connection_manager.remove(session_id)
        buffer_manager.destroy(session_id)
        barge_in_handler.unregister(session_id)
        vad_service.unregister_session(session_id)
        await session_manager.mark_disconnected(session_id)
        logger.info(f"[WS] Cleaned up: session_id={session_id}")


# ── Text message router ───────────────────────────────────────────────────────

async def _handle_text_message(
    raw_text: str,
    session_id: str,
    current_turn_id: Optional[str],
    websocket: WebSocket,
    session_manager: RedisSessionManager,
    buffer_manager: AudioBufferManager,
    barge_in_handler: BargeInHandler,
    vad_service: VADService,
):
    """Parse and dispatch a JSON control message."""
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(f"[WS] Invalid JSON: session={session_id}")
        await websocket.send_text(
            ErrorMessage(
                session_id=session_id,
                code="invalid_json",
                message="Could not parse message.",
            ).model_dump_json()
        )
        return

    msg_type = raw.get("type")
    logger.debug(f"[WS] Control message: session={session_id} type={msg_type}")

    if msg_type == InboundMessageType.SESSION_START:
        # Already handled during connection setup
        pass

    elif msg_type == InboundMessageType.SESSION_END:
        await session_manager.end_session(session_id)
        await websocket.close(code=1000, reason="Session ended by client")

    elif msg_type == "speech_end":
        # Client-side VAD detected end of speech
        turn_id = raw.get("turn_id") or current_turn_id
        if turn_id is None:
            return

        # Mark T2 — speech_end received on server
        record = get_record(session_id, turn_id)
        if record:
            record.mark("speech_end_server")
            # T1 (client speech_end) → from message timestamp
            client_ts = raw.get("timestamp", 0) / 1000  # ms → s
            record.timestamps["speech_end_client"] = client_ts

        await _trigger_flush(
            session_id, turn_id, buffer_manager, session_manager, websocket
        )
        vad_service.reset_session(session_id)

    elif msg_type == InboundMessageType.BARGE_IN:
        await barge_in_handler.handle(session_id, buffer_manager)

    elif msg_type == InboundMessageType.PING:
        pong = PongMessage(
            session_id=session_id,
            server_timestamp=time.time(),
        )
        await websocket.send_text(pong.model_dump_json())

    else:
        logger.warning(
            f"[WS] Unknown message type: session={session_id} type={msg_type}"
        )


# ── Flush trigger ─────────────────────────────────────────────────────────────

async def _trigger_flush(
    session_id: str,
    turn_id: str,
    buffer_manager: AudioBufferManager,
    session_manager: RedisSessionManager,
    websocket: WebSocket,
):
    """
    Flush the audio buffer for this turn.
    The flush callback (set in main.py) will hand PCM off to Stage 2 (STT).
    """
    logger.info(
        f"[WS] Triggering flush: session={session_id} turn={turn_id}"
    )
    await buffer_manager.flush(session_id, turn_id)


# ── Chunk index tracker ───────────────────────────────────────────────────────
# Simple per-session counter for binary frames (no metadata JSON sent)
_chunk_counters: dict[str, int] = {}


def _next_chunk_index(session_id: str) -> int:
    _chunk_counters[session_id] = _chunk_counters.get(session_id, -1) + 1
    return _chunk_counters[session_id]