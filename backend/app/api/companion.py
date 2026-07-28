"""3D companion endpoints: characters, chat, and speech with viseme tracks."""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import companion, voicebank
from ..audio.wavio import write_wav
from ..config import settings
from ..voice_design import EMOTIONS

router = APIRouter(prefix="/api/companion", tags=["companion"])


class CharacterCreate(BaseModel):
    name: str = "Companion"
    voice_id: str
    persona: str = ""
    greeting: str = ""
    default_emotion: str = "neutral"
    idle_lines: list[str] = Field(default_factory=list)
    appearance: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)


class CharacterUpdate(BaseModel):
    name: str | None = None
    voice_id: str | None = None
    persona: str | None = None
    greeting: str | None = None
    default_emotion: str | None = None
    appearance: dict[str, Any] | None = None
    tags: list[str] | None = None


class SayRequest(BaseModel):
    text: str
    emotion: str | None = None
    intensity: float = 1.0
    speed: float = 1.0


class ChatRequest(BaseModel):
    message: str
    speak: bool = True


def _character_or_404(character_id: str) -> dict[str, Any]:
    character = companion.get_character(character_id)
    if character is None:
        raise HTTPException(404, "character not found")
    return character


def _audio_payload(wav: np.ndarray) -> str:
    """WAV as a data URI — one round trip for audio + visemes keeps them in sync."""
    with tempfile.TemporaryDirectory() as tmp:
        path = write_wav(Path(tmp) / "say.wav", wav, settings.sample_rate)
        return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()


# --------------------------------------------------------------------------
# characters
# --------------------------------------------------------------------------
@router.get("/characters")
def list_characters() -> list[dict[str, Any]]:
    return companion.list_characters()


@router.get("/characters/{character_id}")
def get_character(character_id: str) -> dict[str, Any]:
    return _character_or_404(character_id)


@router.post("/characters")
def create_character(body: CharacterCreate) -> dict[str, Any]:
    try:
        return companion.create_character(
            body.name, body.voice_id, body.persona, body.greeting, body.appearance,
            body.default_emotion, body.idle_lines, body.tags)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.patch("/characters/{character_id}")
def update_character(character_id: str, body: CharacterUpdate) -> dict[str, Any]:
    _character_or_404(character_id)
    if body.voice_id and voicebank.get_voice(body.voice_id) is None:
        raise HTTPException(400, f"unknown voice {body.voice_id!r}")
    updated = companion.update_character(character_id, **body.model_dump(exclude_none=True))
    if updated is None:
        raise HTTPException(404, "character not found")
    return updated


@router.delete("/characters/{character_id}")
def delete_character(character_id: str) -> dict[str, bool]:
    try:
        if not companion.delete_character(character_id):
            raise HTTPException(404, "character not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"deleted": True}


# --------------------------------------------------------------------------
# speech + conversation
# --------------------------------------------------------------------------
@router.post("/characters/{character_id}/say")
def say(character_id: str, body: SayRequest) -> dict[str, Any]:
    """Render a line: audio plus the viseme timeline that animates the mouth."""
    character = _character_or_404(character_id)
    text = body.text.strip()[:800]
    if not text:
        raise HTTPException(400, "text is required")
    if body.emotion and body.emotion not in EMOTIONS:
        raise HTTPException(400, f"unknown emotion; try one of {sorted(EMOTIONS)}")

    wav, visemes, emotion = companion.speak(
        character, text, body.emotion, body.intensity, body.speed)
    return {
        "text": text,
        "emotion": emotion,
        "duration": round(wav.size / settings.sample_rate, 3),
        "visemes": visemes,
        "audio": _audio_payload(wav),
    }


@router.post("/characters/{character_id}/chat")
def chat(character_id: str, body: ChatRequest) -> dict[str, Any]:
    character = _character_or_404(character_id)
    message = body.message.strip()[:2000]
    if not message:
        raise HTTPException(400, "message is required")

    reply, emotion, engine = companion.chat(character, message)
    payload: dict[str, Any] = {"reply": reply, "emotion": emotion, "engine": engine}
    if body.speak:
        wav, visemes, emotion = companion.speak(character, reply, emotion)
        payload.update({
            "duration": round(wav.size / settings.sample_rate, 3),
            "visemes": visemes,
            "audio": _audio_payload(wav),
            "emotion": emotion,
        })
    return payload


@router.get("/characters/{character_id}/history")
def get_history(character_id: str) -> list[dict[str, Any]]:
    _character_or_404(character_id)
    return companion.history(character_id)


@router.delete("/characters/{character_id}/history")
def clear_history(character_id: str) -> dict[str, bool]:
    _character_or_404(character_id)
    companion.clear_history(character_id)
    return {"cleared": True}


@router.post("/characters/{character_id}/idle")
def idle(character_id: str) -> dict[str, Any]:
    """A spontaneous in-character line, used when the user goes quiet."""
    character = _character_or_404(character_id)
    text, emotion = companion.idle_line(character)
    wav, visemes, emotion = companion.speak(character, text, emotion)
    return {
        "text": text, "emotion": emotion, "visemes": visemes,
        "duration": round(wav.size / settings.sample_rate, 3),
        "audio": _audio_payload(wav),
    }


@router.get("/visemes")
def viseme_reference(text: str = "Hello there", speed: float = 1.0) -> dict[str, Any]:
    """Inspect the lip-sync track for a line without rendering audio."""
    return {"visemes": companion.viseme_track(text, speed), "shapes": companion.VISEMES}
