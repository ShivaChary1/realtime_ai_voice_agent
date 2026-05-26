"""
tests/test_stage1.py

Unit and integration tests for Stage 1: Audio Pipeline.

Run with:
    pytest tests/test_stage1.py -v
"""

import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from backend.controllers.audio_buffer import (
    AudioBufferManager,
    ChunkMeta,
    SessionAudioBuffer,
)
from backend.controllers.barge_in_handler import BargeInHandler
from config import get_settings
from memory.session_memory.redis_session import RedisSessionManager, SessionStatus
from services.latency.latency_logger import start_turn, end_turn, Stage
from services.vad_service import VADService

settings = get_settings()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def session_id():
    return str(uuid.uuid4())


@pytest.fixture
def turn_id():
    return str(uuid.uuid4())


@pytest.fixture
def sample_pcm_500ms():
    """
    Generate 500ms of silent PCM audio (16kHz, 16-bit, mono).
    500ms = 8000 samples = 16000 bytes.
    """
    samples = int(settings.audio_sample_rate * 0.5)
    return bytes(samples * 2)  # 16-bit = 2 bytes per sample


@pytest.fixture
def sample_pcm_100ms():
    """100ms PCM — below min_audio_ms, should be discarded."""
    samples = int(settings.audio_sample_rate * 0.1)
    return bytes(samples * 2)


# ── PCM Conversion Tests ──────────────────────────────────────────────────────

class TestPCMConversion:
    def test_correct_byte_size_for_500ms(self, sample_pcm_500ms):
        expected_bytes = int(settings.audio_sample_rate * 0.5) * 2
        assert len(sample_pcm_500ms) == expected_bytes

    def test_correct_byte_size_for_100ms(self, sample_pcm_100ms):
        expected_bytes = int(settings.audio_sample_rate * 0.1) * 2
        assert len(sample_pcm_100ms) == expected_bytes


# ── Audio Buffer Tests ────────────────────────────────────────────────────────

class TestSessionAudioBuffer:

    @pytest.mark.asyncio
    async def test_in_order_chunks_appended(self, session_id, sample_pcm_500ms):
        buf = SessionAudioBuffer(session_id)
        chunk1 = ChunkMeta(chunk_index=0, data=sample_pcm_500ms[:100])
        chunk2 = ChunkMeta(chunk_index=1, data=sample_pcm_500ms[100:200])
        await buf.append(chunk1)
        await buf.append(chunk2)
        assert len(buf._buffer) == 200

    @pytest.mark.asyncio
    async def test_out_of_order_chunks_reordered(self, session_id, sample_pcm_500ms):
        buf = SessionAudioBuffer(session_id)
        chunk0 = ChunkMeta(chunk_index=0, data=b"\x01" * 100)
        chunk2 = ChunkMeta(chunk_index=2, data=b"\x03" * 100)
        chunk1 = ChunkMeta(chunk_index=1, data=b"\x02" * 100)

        await buf.append(chunk0)
        await buf.append(chunk2)  # Out of order — held
        await buf.append(chunk1)  # Fills gap — drains 1 and 2

        await asyncio.sleep(0.01)  # let drain task run

        # All three should now be in buffer in order
        assert len(buf._buffer) == 300
        assert buf._buffer[:100]  == b"\x01" * 100
        assert buf._buffer[100:200] == b"\x02" * 100
        assert buf._buffer[200:300] == b"\x03" * 100

    @pytest.mark.asyncio
    async def test_duplicate_chunk_discarded(self, session_id):
        buf = SessionAudioBuffer(session_id)
        chunk = ChunkMeta(chunk_index=0, data=b"\x01" * 100)
        duplicate = ChunkMeta(chunk_index=0, data=b"\x02" * 100)
        await buf.append(chunk)
        await buf.append(duplicate)
        # Only first chunk should be in buffer
        assert len(buf._buffer) == 100
        assert buf._buffer == bytearray(b"\x01" * 100)

    @pytest.mark.asyncio
    async def test_flush_too_short_returns_none(self, session_id, sample_pcm_100ms, turn_id):
        buf = SessionAudioBuffer(session_id)
        chunk = ChunkMeta(chunk_index=0, data=sample_pcm_100ms)
        await buf.append(chunk)

        with patch("backend.controllers.audio_buffer.get_record", return_value=None):
            result = buf.flush(turn_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_flush_adds_silence_padding(self, session_id, sample_pcm_500ms, turn_id):
        buf = SessionAudioBuffer(session_id)
        chunk = ChunkMeta(chunk_index=0, data=sample_pcm_500ms)
        await buf.append(chunk)

        with patch("backend.controllers.audio_buffer.get_record", return_value=None):
            result = buf.flush(turn_id)

        assert result is not None
        # Should be longer than input due to padding
        assert len(result) > len(sample_pcm_500ms)
        # Padding = silence_padding_bytes * 2 (before + after)
        expected_len = len(sample_pcm_500ms) + settings.silence_padding_bytes * 2
        assert len(result) == expected_len

    @pytest.mark.asyncio
    async def test_flush_resets_buffer(self, session_id, sample_pcm_500ms, turn_id):
        buf = SessionAudioBuffer(session_id)
        await buf.append(ChunkMeta(chunk_index=0, data=sample_pcm_500ms))
        with patch("backend.controllers.audio_buffer.get_record", return_value=None):
            buf.flush(turn_id)
        assert len(buf._buffer) == 0

    def test_overflow_detection(self, session_id):
        buf = SessionAudioBuffer(session_id)
        # Fill beyond max_buffer_bytes
        buf._buffer = bytearray(settings.max_buffer_bytes + 1)
        assert buf.is_overflow() is True

    def test_no_overflow_within_limit(self, session_id):
        buf = SessionAudioBuffer(session_id)
        buf._buffer = bytearray(settings.max_buffer_bytes - 1)
        assert buf.is_overflow() is False


# ── AudioBufferManager Tests ──────────────────────────────────────────────────

class TestAudioBufferManager:

    @pytest.mark.asyncio
    async def test_flush_callback_called(self, session_id, sample_pcm_500ms, turn_id):
        manager = AudioBufferManager()
        received = {}

        async def callback(sid, tid, pcm):
            received["session_id"] = sid
            received["turn_id"] = tid
            received["bytes"] = len(pcm)

        manager.set_flush_callback(callback)
        manager.create(session_id)

        chunk = ChunkMeta(chunk_index=0, data=sample_pcm_500ms)
        with patch("backend.controllers.audio_buffer.get_record", return_value=None):
            await manager.handle_chunk(session_id, chunk)
            await manager.flush(session_id, turn_id)

        assert received.get("session_id") == session_id
        assert received.get("turn_id") == turn_id
        assert received.get("bytes", 0) > 0

    @pytest.mark.asyncio
    async def test_discard_clears_buffer(self, session_id, sample_pcm_500ms):
        manager = AudioBufferManager()
        manager.create(session_id)
        chunk = ChunkMeta(chunk_index=0, data=sample_pcm_500ms)
        await manager.handle_chunk(session_id, chunk)
        manager.discard(session_id)
        buf = manager.get(session_id)
        assert len(buf._buffer) == 0

    @pytest.mark.asyncio
    async def test_overflow_triggers_flush(self, session_id, turn_id):
        manager = AudioBufferManager()
        flushed = {"called": False}

        async def callback(sid, tid, pcm):
            flushed["called"] = True

        manager.set_flush_callback(callback)
        manager.create(session_id)

        # Directly set buffer to overflow size
        buf = manager.get(session_id)
        buf._buffer = bytearray(settings.max_buffer_bytes + 1)

        # Send any chunk to trigger overflow check
        small_chunk = ChunkMeta(chunk_index=0, data=b"\x00" * 32)
        with patch("backend.controllers.audio_buffer.get_record", return_value=None):
            await manager.handle_chunk(session_id, small_chunk)

        assert flushed["called"] is True


# ── Barge-In Handler Tests ────────────────────────────────────────────────────

class TestBargeInHandler:

    @pytest.mark.asyncio
    async def test_spurious_barge_in_ignored(self, session_id):
        handler = BargeInHandler()
        mock_ws = AsyncMock()
        handler.register_websocket(session_id, mock_ws)
        handler.set_agent_speaking(session_id, False)

        mock_buffer_manager = MagicMock()
        result = await handler.handle(session_id, mock_buffer_manager)

        assert result is False
        mock_buffer_manager.discard.assert_not_called()

    @pytest.mark.asyncio
    async def test_real_barge_in_processed(self, session_id):
        handler = BargeInHandler()
        mock_ws = AsyncMock()
        handler.register_websocket(session_id, mock_ws)
        handler.set_agent_speaking(session_id, True)

        mock_buffer_manager = MagicMock()
        result = await handler.handle(session_id, mock_buffer_manager)

        assert result is True
        mock_buffer_manager.discard.assert_called_once_with(session_id)
        assert handler.is_agent_speaking(session_id) is False

    @pytest.mark.asyncio
    async def test_stop_playback_sent_on_barge_in(self, session_id):
        handler = BargeInHandler()
        mock_ws = AsyncMock()
        handler.register_websocket(session_id, mock_ws)
        handler.set_agent_speaking(session_id, True)

        mock_buffer_manager = MagicMock()
        await handler.handle(session_id, mock_buffer_manager)

        # Verify stop_playback was sent
        mock_ws.send_text.assert_called_once()
        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "stop_playback"
        assert sent["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_active_tasks_cancelled_on_barge_in(self, session_id):
        handler = BargeInHandler()
        mock_ws = AsyncMock()
        handler.register_websocket(session_id, mock_ws)
        handler.set_agent_speaking(session_id, True)

        # Create a mock cancellable task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        handler.track_task(session_id, mock_task)

        mock_buffer_manager = MagicMock()
        await handler.handle(session_id, mock_buffer_manager)

        mock_task.cancel.assert_called_once()


# ── Latency Logger Tests ──────────────────────────────────────────────────────

class TestLatencyLogger:

    def test_start_and_end_turn(self, session_id, turn_id):
        record = start_turn(session_id, turn_id)
        assert record is not None
        assert "pipeline_start" in record.timestamps

        summary = end_turn(session_id, turn_id)
        assert summary is not None
        assert "total_ms" in summary
        assert summary["total_ms"] >= 0

    def test_measure_duration(self, session_id, turn_id):
        record = start_turn(session_id, turn_id)
        time.sleep(0.01)  # 10ms sleep
        record.mark("checkpoint_b")
        duration = record.measure("test_stage", "pipeline_start", "checkpoint_b")
        assert duration is not None
        assert duration >= 10.0  # at least 10ms

    def test_missing_checkpoint_returns_none(self, session_id, turn_id):
        record = start_turn(session_id, turn_id)
        duration = record.measure("test_stage", "nonexistent_a", "nonexistent_b")
        assert duration is None
        end_turn(session_id, turn_id)

    def test_budget_exceeded_logs_warning(self, session_id, turn_id, caplog):
        import logging
        record = start_turn(session_id, turn_id)

        # Force a large delay
        time.sleep(0.2)
        record.mark("late_checkpoint")

        with caplog.at_level(logging.WARNING):
            duration = record.measure(Stage.AUDIO_CAPTURE, "pipeline_start", "late_checkpoint")

        # Should have triggered a warning
        assert any("OVER BUDGET" in r.message for r in caplog.records)
        end_turn(session_id, turn_id)


# ── Redis Session Tests (mocked) ──────────────────────────────────────────────

class TestRedisSessionManager:

    @pytest.mark.asyncio
    async def test_create_and_get_session(self, session_id):
        manager = RedisSessionManager()

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.setex = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        manager._client = mock_redis

        # First get returns None (session doesn't exist yet)
        result = await manager.get_session(session_id)
        assert result is None

        # Create session
        import json as _json
        session_data = None

        async def mock_setex(key, ttl, data):
            nonlocal session_data
            session_data = _json.loads(data)

        mock_redis.setex = mock_setex
        created = await manager.create_session(session_id)

        assert created["session_id"] == session_id
        assert created["status"] == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_mark_disconnected_reduces_ttl(self, session_id):
        manager = RedisSessionManager()

        stored = {}
        async def mock_setex(key, ttl, data):
            stored["ttl"] = ttl

        import json as _json
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=_json.dumps({
            "session_id": session_id,
            "status": SessionStatus.ACTIVE,
            "conversation_history": [],
            "language": "en",
        }))
        mock_redis.setex = mock_setex

        manager._client = mock_redis
        await manager.mark_disconnected(session_id)

        assert stored["ttl"] == settings.disconnect_ttl_seconds

    @pytest.mark.asyncio
    async def test_append_turn_adds_to_history(self, session_id):
        manager = RedisSessionManager()

        history = []
        session_store = {
            "session_id": session_id,
            "status": SessionStatus.ACTIVE,
            "conversation_history": history,
            "language": "en",
        }

        import json as _json
        async def mock_get(key):
            return _json.dumps(session_store)

        async def mock_setex(key, ttl, data):
            session_store.update(_json.loads(data))

        mock_redis = AsyncMock()
        mock_redis.get = mock_get
        mock_redis.setex = mock_setex
        manager._client = mock_redis

        await manager.append_turn(session_id, "user", "Book an appointment")
        assert len(session_store["conversation_history"]) == 1
        assert session_store["conversation_history"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_history_capped_at_20_turns(self, session_id):
        manager = RedisSessionManager()

        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        session_store = {
            "session_id": session_id,
            "status": SessionStatus.ACTIVE,
            "conversation_history": long_history,
            "language": "en",
        }

        import json as _json
        async def mock_get(key):
            return _json.dumps(session_store)

        async def mock_setex(key, ttl, data):
            session_store.update(_json.loads(data))

        mock_redis = AsyncMock()
        mock_redis.get = mock_get
        mock_redis.setex = mock_setex
        manager._client = mock_redis

        # Add one more — should trim to 20
        await manager.append_turn(session_id, "user", "one more message")
        assert len(session_store["conversation_history"]) == 20