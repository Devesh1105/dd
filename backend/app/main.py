"""FastAPI application entry point.

Run locally:  python -m backend.app.main   (or ./run.sh)
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, voicebank
from .api import jobs as jobs_api
from .api import projects as projects_api
from .api import voices as voices_api
from .config import BASE_DIR, settings
from .core.queue import manager
from .media import ffmpeg
from .pipeline import orchestrator  # noqa: F401  (registers job handlers)
from .providers import asr, separation, translate, tts  # noqa: F401  (registers providers)
from .providers.base import registry

FRONTEND_DIR = BASE_DIR / "frontend"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    seeded = voicebank.ensure_presets()
    if seeded:
        print(f"[dubbing] seeded {seeded} preset voices")
    await manager.start()
    print(f"[dubbing] ready on http://{settings.host}:{settings.port}")
    print(f"[dubbing] providers: asr={registry.get('asr', settings.asr_provider).name} "
          f"mt={registry.get('mt', settings.mt_provider).name} "
          f"tts={registry.get('tts', settings.tts_provider).name} "
          f"separation={registry.get('separation', settings.separation_provider).name}")
    if not ffmpeg.have_ffmpeg():
        print("[dubbing] ffmpeg not found — WAV in/out only, video muxing disabled")
    try:
        yield
    finally:
        await manager.stop()


app = FastAPI(
    title="AI Dubbing & Voice Platform",
    version="1.0.0",
    description="Local-first dubbing pipeline: ASR → translation → voice cloning → TTS → mix.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   f"http://{settings.host}:{settings.port}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_api.router)
app.include_router(voices_api.router)
app.include_router(jobs_api.router)


@app.get("/api/system", tags=["system"])
def system_info() -> dict[str, Any]:
    """What this installation can actually do right now."""
    from .providers.translate import CHARS_PER_SECOND, EXPANSION, LANGUAGE_NAMES

    active = {}
    for capability, configured in (("asr", settings.asr_provider), ("mt", settings.mt_provider),
                                   ("tts", settings.tts_provider),
                                   ("separation", settings.separation_provider),
                                   ("lipsync", settings.lipsync_provider), ("vc", "auto")):
        try:
            active[capability] = registry.get(capability, configured).name
        except Exception as exc:  # pragma: no cover
            active[capability] = f"error: {exc}"

    return {
        "version": app.version,
        "sample_rate": settings.sample_rate,
        "workers": settings.workers,
        "ffmpeg": ffmpeg.have_ffmpeg(),
        "active_providers": active,
        "providers": registry.describe(),
        "languages": [{"code": c, "name": LANGUAGE_NAMES.get(c, c), "expansion": EXPANSION.get(c, 1.0)}
                      for c in sorted(LANGUAGE_NAMES)],
        "chars_per_second": CHARS_PER_SECOND,
        "capabilities": {
            "video": ffmpeg.have_ffmpeg(),
            "lipsync": active.get("lipsync") not in (None, "none"),
            "real_translation": active.get("mt") != "passthrough",
            "real_asr": active.get("asr") != "offline",
            "neural_tts": active.get("tts") != "local_formant",
        },
    }


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run("backend.app.main:app", host=settings.host, port=settings.port,
                reload=False, log_level="info")


if __name__ == "__main__":
    main()
