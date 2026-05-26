"""
backend/main.py

FastAPI application entry point.
Initializes all services and wires them together.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.websocket import router as ws_router
from backend.controllers.audio_buffer import AudioBufferManager
from backend.controllers.barge_in_handler import BargeInHandler
from memory.session_memory.redis_session import RedisSessionManager
from services.vad_service import VADService
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── STT flush callback placeholder ────────────────────────────────────────────
# Stage 2 will register the real callback here.
# For now it logs to confirm Stage 1 is working end-to-end.
async def _stt_stub(session_id: str, turn_id: str, pcm_bytes: bytes):
    logger.info(
        f"[STT STUB] Ready to process: session={session_id} "
        f"turn={turn_id} bytes={len(pcm_bytes)} "
        f"duration_s={len(pcm_bytes) / (16000 * 2):.2f}s"
    )
    # Stage 2 will replace this with actual Whisper/Deepgram call


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down all shared services."""
    logger.info("🚀 Starting Voice AI Agent (Stage 1)...")

    # Redis session manager
    session_manager = RedisSessionManager()
    await session_manager.connect()
    app.state.session_manager = session_manager

    # Audio buffer manager with STT callback
    buffer_manager = AudioBufferManager()
    buffer_manager.set_flush_callback(_stt_stub)
    app.state.buffer_manager = buffer_manager

    # Barge-in handler
    app.state.barge_in_handler = BargeInHandler()

    # VAD service
    app.state.vad_service = VADService()

    logger.info("✅ All Stage 1 services initialized.")
    yield

    # Shutdown
    logger.info("🛑 Shutting down...")
    await session_manager.disconnect()
    logger.info("Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice AI Appointment Agent",
    description="Real-time multilingual voice agent — Stage 1: Audio Pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


@app.get("/health")
def health():
    return {
        "status": "running",
        "stage": "1 - Audio Pipeline",
        "version": "0.1.0",
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )