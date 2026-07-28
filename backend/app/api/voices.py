"""Voice bank endpoints: presets, design, cloning, auditioning, conversion."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response

from .. import voicebank
from ..audio.wavio import read_wav, resample, to_mono, write_wav
from ..config import settings
from ..providers.base import registry
from ..schemas import DesignVoiceRequest, PreviewRequest, VoiceUpdate
from ..voice_design import EMOTIONS, PRESETS, VoiceParams

router = APIRouter(prefix="/api/voices", tags=["voices"])

MAX_REFERENCE_MB = 64


def _wav_response(samples: np.ndarray, sample_rate: int, filename: str) -> Response:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = write_wav(Path(tmp) / "out.wav", samples, sample_rate)
        data = path.read_bytes()
    return Response(content=data, media_type="audio/wav",
                    headers={"Content-Disposition": f'inline; filename="{filename}"',
                             "Cache-Control": "no-store"})


async def _read_upload(file: UploadFile, sample_rate: int) -> np.ndarray:
    """Decode an uploaded reference clip to mono float32."""
    from ..media import ffmpeg

    suffix = Path(file.filename or "").suffix.lower() or ".wav"
    if suffix not in (ffmpeg.AUDIO_EXT | ffmpeg.VIDEO_EXT):
        raise HTTPException(400, f"unsupported audio type {suffix}")

    tmp_dir = settings.media_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw = tmp_dir / f"upload_{abs(hash(file.filename or 'x')) % 10**8}{suffix}"
    size = 0
    try:
        with raw.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_REFERENCE_MB * 1024 * 1024:
                    raise HTTPException(413, f"reference audio exceeds {MAX_REFERENCE_MB} MB")
                fh.write(chunk)
        if size == 0:
            raise HTTPException(400, "uploaded file is empty")
        wav = tmp_dir / f"{raw.stem}_dec.wav"
        try:
            ffmpeg.extract_audio(raw, wav, sample_rate)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        data, sr = read_wav(wav)
        wav.unlink(missing_ok=True)
        return resample(to_mono(data), sr, sample_rate)
    finally:
        raw.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------
@router.get("")
def list_voices(kind: str | None = None, category: str | None = None,
                q: str | None = Query(None, alias="q")) -> list[dict[str, Any]]:
    return voicebank.list_voices(kind=kind, category=category, query=q)


@router.get("/archetypes")
def list_archetypes() -> dict[str, Any]:
    """The prompt library — ready-made descriptions for Voice Design."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for preset in PRESETS:
        by_category.setdefault(preset.category, []).append({
            "id": preset.id, "name": preset.name, "prompt": preset.prompt,
            "tags": preset.tags, "language": preset.language,
        })
    return {
        "categories": by_category,
        # neutral first so UI dropdowns default to it rather than to whatever
        # happens to sort first alphabetically
        "emotions": ["neutral"] + sorted(e for e in EMOTIONS if e != "neutral"),
        "prompt_structure": [
            "Age & gender — e.g. 'a young adult male, early 20s'",
            "Pitch & texture — gravelly, raspy, airy, deep, smooth, velvety",
            "Pacing & rhythm — fast-paced and rapid-fire vs. slow and deliberate",
            "Emotion & attitude — arrogant, tsundere, hyperactive, brooding, stoic",
            "Accent & style — classic shounen dub, dramatic anime dubbing, neutral broadcast",
        ],
    }


@router.get("/{voice_id}")
def get_voice(voice_id: str) -> dict[str, Any]:
    voice = voicebank.get_voice(voice_id)
    if voice is None:
        raise HTTPException(404, "voice not found")
    return voice


# --------------------------------------------------------------------------
# creation
# --------------------------------------------------------------------------
@router.post("/design")
def design_voice(body: DesignVoiceRequest) -> dict[str, Any]:
    try:
        return voicebank.create_designed_voice(
            body.name, body.prompt, body.language, tags=body.tags, overrides=body.overrides)
    except voicebank.VoiceError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/clone")
async def clone_voice(
    file: UploadFile = File(...),
    name: str = Form("Cloned voice"),
    kind: str = Form("instant"),
    language: str = Form("en"),
    prompt: str = Form(""),
) -> dict[str, Any]:
    """Zero-shot (instant) or professional cloning from a reference recording."""
    if kind not in ("instant", "professional"):
        raise HTTPException(400, "kind must be 'instant' or 'professional'")
    audio = await _read_upload(file, settings.sample_rate)
    try:
        return voicebank.create_cloned_voice(
            name, audio, settings.sample_rate, kind=kind, language=language, prompt=prompt)
    except voicebank.VoiceError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/match")
async def match_voice(file: UploadFile = File(...), limit: int = Form(5)) -> list[dict[str, Any]]:
    """Find the closest voices in the bank to an uploaded speaker sample."""
    audio = await _read_upload(file, settings.sample_rate)
    return voicebank.suggest_for_audio(audio, settings.sample_rate, limit=limit)


# --------------------------------------------------------------------------
# editing + auditioning
# --------------------------------------------------------------------------
@router.patch("/{voice_id}")
def update_voice(voice_id: str, body: VoiceUpdate) -> dict[str, Any]:
    voice = voicebank.update_voice(voice_id, body.name, body.params, body.tags)
    if voice is None:
        raise HTTPException(404, "voice not found")
    return voice


@router.delete("/{voice_id}")
def delete_voice(voice_id: str) -> dict[str, bool]:
    try:
        if not voicebank.delete_voice(voice_id):
            raise HTTPException(404, "voice not found")
    except voicebank.VoiceError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"deleted": True}


@router.post("/{voice_id}/preview")
def preview_voice(voice_id: str, body: PreviewRequest) -> Response:
    """Micro-playback used by the real-time auditioning panel."""
    if voicebank.get_voice(voice_id) is None:
        raise HTTPException(404, "voice not found")
    if body.emotion not in EMOTIONS:
        raise HTTPException(400, f"unknown emotion; try one of {sorted(EMOTIONS)}")
    text = body.text.strip()[:600]
    if not text:
        raise HTTPException(400, "preview text is required")

    ref = voicebank.voice_ref(voice_id)
    tts = registry.get("tts", settings.tts_provider)
    wav = tts.synthesize(text, ref, settings.sample_rate, emotion=body.emotion,
                         intensity=body.intensity, speed=body.speed)
    return _wav_response(wav, settings.sample_rate, f"{voice_id}_preview.wav")


@router.post("/preview-prompt")
def preview_prompt(body: DesignVoiceRequest, text: str = Query("This is a preview of the designed voice.")
                   ) -> Response:
    """Audition a Voice Design prompt without saving it."""
    from ..voice_design import design_voice as build

    params = build(body.prompt, seed=body.name)
    if body.overrides:
        params = VoiceParams.from_dict({**params.to_dict(), **body.overrides}).clamp()
    from ..audio import synth

    wav = synth.synthesize(text[:600], params, settings.sample_rate)
    return _wav_response(wav, settings.sample_rate, "prompt_preview.wav")


@router.post("/{voice_id}/convert")
async def convert_voice(voice_id: str, file: UploadFile = File(...),
                        strength: float = Form(1.0)) -> Response:
    """Voice-to-voice: re-voice an uploaded clip, keeping its timing and energy."""
    if voicebank.get_voice(voice_id) is None:
        raise HTTPException(404, "voice not found")
    audio = await _read_upload(file, settings.sample_rate)
    vc = registry.get("vc", "auto")
    out = vc.convert(audio, settings.sample_rate, voicebank.voice_ref(voice_id),
                     strength=float(np.clip(strength, 0.0, 1.0)))
    return _wav_response(out, settings.sample_rate, f"{voice_id}_converted.wav")
