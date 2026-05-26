"""
services/latency/latency_logger.py

Centralized latency measurement across all pipeline stages.
Every stage calls this to log its duration.
Warns when budget is exceeded.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Stage names ────────────────────────────────────────────────────────────────
class Stage:
    AUDIO_CAPTURE    = "stage1_audio_capture"
    STT              = "stage2_stt"
    LANG_DETECT      = "stage3_language_detection"
    LLM_AGENT        = "stage4_llm_agent"
    TOOL_CALL        = "stage5_tool_call"
    TTS              = "stage6_tts"
    TOTAL_PIPELINE   = "total_pipeline"


# ── Per-session latency record ─────────────────────────────────────────────────
@dataclass
class LatencyRecord:
    session_id: str
    turn_id: str
    timestamps: Dict[str, float] = field(default_factory=dict)
    durations_ms: Dict[str, float] = field(default_factory=dict)

    def mark(self, checkpoint: str) -> float:
        """Record a named timestamp. Returns current time."""
        ts = time.perf_counter()
        self.timestamps[checkpoint] = ts
        return ts

    def measure(self, stage: str, start_checkpoint: str, end_checkpoint: str) -> Optional[float]:
        """
        Calculate duration between two checkpoints in milliseconds.
        Logs a warning if the stage exceeds its budget.
        """
        start = self.timestamps.get(start_checkpoint)
        end = self.timestamps.get(end_checkpoint)

        if start is None or end is None:
            logger.warning(
                f"[LATENCY] Missing checkpoint for stage={stage} "
                f"start={start_checkpoint} end={end_checkpoint}"
            )
            return None

        duration_ms = (end - start) * 1000
        self.durations_ms[stage] = duration_ms

        # Budget check
        budget = self._get_budget(stage)
        level = logging.WARNING if duration_ms > budget else logging.INFO

        logger.log(
            level,
            f"[LATENCY] session={self.session_id} turn={self.turn_id} "
            f"stage={stage} duration={duration_ms:.1f}ms budget={budget}ms "
            f"{'⚠ OVER BUDGET' if duration_ms > budget else '✓'}",
        )
        return duration_ms

    def summary(self) -> Dict[str, float]:
        """Return all measured durations for this turn."""
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "durations_ms": self.durations_ms,
            "total_ms": sum(self.durations_ms.values()),
        }

    def _get_budget(self, stage: str) -> float:
        budgets = {
            Stage.AUDIO_CAPTURE:  settings.stage1_latency_budget_ms,
            Stage.STT:            130.0,
            Stage.LANG_DETECT:    20.0,
            Stage.LLM_AGENT:      200.0,
            Stage.TOOL_CALL:      50.0,
            Stage.TTS:            100.0,
            Stage.TOTAL_PIPELINE: settings.total_latency_budget_ms,
        }
        return budgets.get(stage, 9999.0)


# ── In-memory store of active records ─────────────────────────────────────────
_active_records: Dict[str, LatencyRecord] = {}


def start_turn(session_id: str, turn_id: str) -> LatencyRecord:
    """Create a new latency record for a conversation turn."""
    record = LatencyRecord(session_id=session_id, turn_id=turn_id)
    record.mark("pipeline_start")
    _active_records[f"{session_id}:{turn_id}"] = record
    return record


def get_record(session_id: str, turn_id: str) -> Optional[LatencyRecord]:
    """Retrieve an existing latency record."""
    return _active_records.get(f"{session_id}:{turn_id}")


def end_turn(session_id: str, turn_id: str) -> Optional[Dict]:
    """
    Finalize a turn's latency record, log summary, and clean up.
    Returns the full summary dict.
    """
    key = f"{session_id}:{turn_id}"
    record = _active_records.pop(key, None)
    if not record:
        return None

    record.mark("pipeline_end")
    record.measure(Stage.TOTAL_PIPELINE, "pipeline_start", "pipeline_end")

    summary = record.summary()
    logger.info(
        f"[LATENCY SUMMARY] session={session_id} turn={turn_id} "
        f"total={summary['total_ms']:.1f}ms details={summary['durations_ms']}"
    )
    return summary