"""Request/response models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(default="Untitled project", max_length=200)
    source_language: str = "auto"
    target_language: str = "es"
    auto_start: bool = True


class ProjectUpdate(BaseModel):
    name: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    settings: dict[str, Any] | None = None


class SegmentUpdate(BaseModel):
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    source_text: str | None = None
    target_text: str | None = None
    emotion: str | None = None
    voice_id: str | None = None
    locked: bool | None = None


class SegmentBulkUpdate(BaseModel):
    segments: list[dict[str, Any]]


class ScriptUpload(BaseModel):
    script: str
    language: str = "auto"


class DesignVoiceRequest(BaseModel):
    name: str = "Designed voice"
    prompt: str
    language: str = "en"
    tags: list[str] = Field(default_factory=list)
    overrides: dict[str, float] | None = None


class VoiceUpdate(BaseModel):
    name: str | None = None
    params: dict[str, float] | None = None
    tags: list[str] | None = None


class PreviewRequest(BaseModel):
    text: str = "This is how the voice sounds when it reads a line of dialogue."
    emotion: str = "neutral"
    intensity: float = 1.0
    speed: float = 1.0


class DubRequest(BaseModel):
    target_language: str | None = None
    settings: dict[str, Any] | None = None
