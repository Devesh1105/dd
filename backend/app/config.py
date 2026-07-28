"""Runtime configuration.

Everything is env-overridable so the same code runs on a laptop with zero
models installed, or on a GPU box with Whisper/XTTS/Demucs available.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # storage --------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DUB_DATA_DIR", str(BASE_DIR / "data"))))
    db_path: Path = field(init=False)
    media_dir: Path = field(init=False)

    # server ---------------------------------------------------------------
    host: str = field(default_factory=lambda: _env("DUB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("DUB_PORT", 8000))

    # pipeline -------------------------------------------------------------
    workers: int = field(default_factory=lambda: _env_int("DUB_WORKERS", 2))
    sample_rate: int = field(default_factory=lambda: _env_int("DUB_SAMPLE_RATE", 24000))
    chunk_seconds: float = field(default_factory=lambda: float(_env("DUB_CHUNK_SECONDS", "12")))
    max_upload_mb: int = field(default_factory=lambda: _env_int("DUB_MAX_UPLOAD_MB", 512))

    # providers: "auto" picks the best locally-installed engine, else offline
    asr_provider: str = field(default_factory=lambda: _env("DUB_ASR", "auto"))
    mt_provider: str = field(default_factory=lambda: _env("DUB_MT", "auto"))
    tts_provider: str = field(default_factory=lambda: _env("DUB_TTS", "auto"))
    separation_provider: str = field(default_factory=lambda: _env("DUB_SEPARATION", "auto"))
    lipsync_provider: str = field(default_factory=lambda: _env("DUB_LIPSYNC", "auto"))

    # optional third-party creds (never required for local operation)
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY", ""))
    whisper_model: str = field(default_factory=lambda: _env("DUB_WHISPER_MODEL", "base"))
    piper_voice_dir: str = field(default_factory=lambda: _env("DUB_PIPER_VOICE_DIR", ""))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.db_path = Path(_env("DUB_DB_PATH", str(self.data_dir / "dubbing.db")))
        self.media_dir = Path(_env("DUB_MEDIA_DIR", str(self.data_dir / "media")))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        p = self.media_dir / project_id
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
