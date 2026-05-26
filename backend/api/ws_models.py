"""
backend/api/ws_models.py

Pydantic models defining the WebSocket message protocol.
Both incoming (client → server) and outgoing (server → client) messages.
"""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


# ── Message Types ─────────────────────────────────────────────────────────────

class InboundMessageType(str, Enum):
    AUDIO_CHUNK   = "audio_chunk"
    SESSION_START = "session_start"
    SESSION_END   = "session_end"
    BARGE_IN      = "barge_in"
    PING          = "ping"


class OutboundMessageType(str, Enum):
    AUDIO_RESPONSE = "audio_response"   # TTS audio chunk back to client
    TRANSCRIPT     = "transcript"       # STT result shown in UI
    AGENT_TEXT     = "agent_text"       # Agent text response
    STOP_PLAYBACK  = "stop_playback"    # Tell client to stop audio
    SESSION_ACK    = "session_ack"      # Session creation confirmed
    ERROR          = "error"            # Error message
    PONG           = "pong"             # Ping response
    LATENCY_REPORT = "latency_report"   # Per-turn latency stats


# ── Inbound Messages (Client → Server) ───────────────────────────────────────

class AudioChunkMessage(BaseModel):
    """Raw PCM audio chunk from the browser."""
    type: InboundMessageType = InboundMessageType.AUDIO_CHUNK
    session_id: str
    turn_id: str                        # Groups chunks in same utterance
    chunk_index: int                    # Sequential index for reordering
    timestamp: float                    # Client timestamp (ms epoch)
    # Note: actual binary audio is sent as binary WebSocket frame
    # This model is for the metadata JSON frame that precedes it
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm_s16le"        # PCM signed 16-bit little-endian


class SessionStartMessage(BaseModel):
    type: InboundMessageType = InboundMessageType.SESSION_START
    session_id: str
    patient_id: Optional[str] = None
    timestamp: float


class SessionEndMessage(BaseModel):
    type: InboundMessageType = InboundMessageType.SESSION_END
    session_id: str
    timestamp: float


class BargeInMessage(BaseModel):
    type: InboundMessageType = InboundMessageType.BARGE_IN
    session_id: str
    turn_id: str                        # Turn that was interrupted
    timestamp: float


class PingMessage(BaseModel):
    type: InboundMessageType = InboundMessageType.PING
    session_id: str
    timestamp: float


# ── Outbound Messages (Server → Client) ──────────────────────────────────────

class TranscriptMessage(BaseModel):
    type: OutboundMessageType = OutboundMessageType.TRANSCRIPT
    session_id: str
    turn_id: str
    text: str
    language: str
    is_final: bool = True


class AgentTextMessage(BaseModel):
    type: OutboundMessageType = OutboundMessageType.AGENT_TEXT
    session_id: str
    turn_id: str
    text: str
    language: str
    is_partial: bool = False            # True for streaming tokens


class SessionAckMessage(BaseModel):
    type: OutboundMessageType = OutboundMessageType.SESSION_ACK
    session_id: str
    status: str = "created"            # created | resumed
    timestamp: float


class ErrorMessage(BaseModel):
    type: OutboundMessageType = OutboundMessageType.ERROR
    session_id: str
    code: str
    message: str


class LatencyReportMessage(BaseModel):
    type: OutboundMessageType = OutboundMessageType.LATENCY_REPORT
    session_id: str
    turn_id: str
    durations_ms: Dict[str, float]
    total_ms: float


class PongMessage(BaseModel):
    type: OutboundMessageType = OutboundMessageType.PONG
    session_id: str
    server_timestamp: float


# ── Message parser ────────────────────────────────────────────────────────────

def parse_inbound_message(raw: Dict[str, Any]):
    """
    Parse a raw dict into the correct inbound message model.
    Raises ValueError for unknown message types.
    """
    msg_type = raw.get("type")

    parsers = {
        InboundMessageType.AUDIO_CHUNK:   AudioChunkMessage,
        InboundMessageType.SESSION_START: SessionStartMessage,
        InboundMessageType.SESSION_END:   SessionEndMessage,
        InboundMessageType.BARGE_IN:      BargeInMessage,
        InboundMessageType.PING:          PingMessage,
    }

    parser = parsers.get(msg_type)
    if parser is None:
        raise ValueError(f"Unknown message type: {msg_type}")

    return parser(**raw)