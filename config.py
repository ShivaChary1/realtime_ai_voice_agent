"""
config.py
Centralized configuration loaded from environment variables / .env file.
All modules import from here — never read os.environ directly.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Server ──────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"

    # ── Redis ───────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    session_ttl_seconds: int = 1800
    disconnect_ttl_seconds: int = 120

    # ── Audio ───────────────────────────────────────────
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_bit_depth: int = 16
    max_buffer_seconds: int = 30
    min_audio_ms: int = 500
    silence_padding_ms: int = 300
    chunk_size_ms: int = 250

    # ── VAD ─────────────────────────────────────────────
    vad_mode: int = 2
    vad_silence_frames: int = 8
    vad_frame_ms: int = 30

    # ── Latency ─────────────────────────────────────────
    stage1_latency_budget_ms: int = 150
    total_latency_budget_ms: int = 450

    # ── Logging ─────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_buffer_bytes(self) -> int:
        """Max buffer size in bytes (PCM 16-bit = 2 bytes per sample)."""
        return self.audio_sample_rate * self.max_buffer_seconds * 2

    @property
    def min_audio_bytes(self) -> int:
        """Minimum audio bytes to send to STT."""
        samples = int(self.audio_sample_rate * (self.min_audio_ms / 1000))
        return samples * 2  # 16-bit = 2 bytes

    @property
    def silence_padding_bytes(self) -> int:
        """Silence padding in bytes."""
        samples = int(self.audio_sample_rate * (self.silence_padding_ms / 1000))
        return samples * 2


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()