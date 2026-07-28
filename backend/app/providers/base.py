"""Provider contracts + registry.

Every AI capability in the pipeline sits behind one of these interfaces, so
the orchestrator never knows whether speech came from a local formant
synthesiser, a self-hosted XTTS/F5/Chatterbox checkpoint, or ElevenLabs.

Selection order for `auto`: the best *locally installed* engine, falling back
to the always-available offline implementation. Nothing here ever requires
network access unless the operator explicitly configures an API provider.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..voice_design import VoiceParams


# --------------------------------------------------------------------------
# shared value objects
# --------------------------------------------------------------------------
@dataclass
class Utterance:
    start: float
    end: float
    text: str
    speaker: str = "SPK_1"
    language: str = "en"
    confidence: float = 1.0


@dataclass
class VoiceRef:
    """Everything a TTS engine might need to render a specific voice."""

    id: str
    name: str = ""
    kind: str = "preset"                    # preset|instant|professional|designed|converted
    params: VoiceParams = field(default_factory=VoiceParams)
    embedding: np.ndarray | None = None
    reference_path: str | None = None       # for zero-shot conditioning
    provider: str = "local"
    provider_voice_id: str | None = None
    language: str = "en"
    prompt: str = ""


class ASRProvider(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "auto"
                   ) -> list[Utterance]: ...


class MTProvider(Protocol):
    name: str

    def translate(self, texts: list[str], source: str, target: str,
                  char_budgets: list[int] | None = None) -> list[str]: ...


class TTSProvider(Protocol):
    name: str

    def synthesize(self, text: str, voice: VoiceRef, sample_rate: int,
                   emotion: str = "neutral", intensity: float = 1.0,
                   speed: float = 1.0) -> np.ndarray: ...


class VCProvider(Protocol):
    name: str

    def convert(self, audio: np.ndarray, sample_rate: int, voice: VoiceRef,
                strength: float = 1.0) -> np.ndarray: ...


class SeparationProvider(Protocol):
    name: str

    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]: ...


class LipsyncProvider(Protocol):
    name: str

    def sync(self, video_path: str, audio_path: str, out_path: str) -> str | None: ...


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class Registry:
    """Maps capability -> {provider name: factory}, with an `auto` chain."""

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Any]] = {}
        self._auto_order: dict[str, list[str]] = {}
        self._cache: dict[tuple[str, str], Any] = {}

    def register(self, capability: str, name: str, factory, auto_rank: int | None = None) -> None:
        self._factories.setdefault(capability, {})[name] = factory
        if auto_rank is not None:
            order = self._auto_order.setdefault(capability, [])
            order.append(name)
            order.sort(key=lambda n: getattr(self._factories[capability][n], "auto_rank", 100))

    def available(self, capability: str) -> list[str]:
        out = []
        for name, factory in self._factories.get(capability, {}).items():
            probe = getattr(factory, "is_available", None)
            if probe is None or probe():
                out.append(name)
        return out

    def get(self, capability: str, name: str = "auto"):
        key = (capability, name)
        if key in self._cache:
            return self._cache[key]
        factories = self._factories.get(capability, {})
        if not factories:
            raise KeyError(f"no providers registered for {capability!r}")

        chosen = None
        if name and name != "auto":
            if name not in factories:
                raise KeyError(f"unknown {capability} provider {name!r}; have {sorted(factories)}")
            chosen = factories[name]
        else:
            ordered = sorted(self._auto_order.get(capability, list(factories)),
                             key=lambda n: getattr(factories[n], "auto_rank", 100))
            for candidate in ordered:
                factory = factories[candidate]
                probe = getattr(factory, "is_available", None)
                if probe is None or probe():
                    chosen, name = factory, candidate
                    break
        if chosen is None:  # pragma: no cover - offline providers are always available
            raise RuntimeError(f"no usable provider for {capability}")
        instance = chosen()
        self._cache[key] = instance
        self._cache[(capability, name)] = instance
        return instance

    def describe(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for capability, factories in self._factories.items():
            entries = {}
            for name, factory in factories.items():
                probe = getattr(factory, "is_available", None)
                entries[name] = {
                    "available": bool(probe()) if probe else True,
                    "requires": getattr(factory, "requires", ""),
                    "description": (getattr(factory, "__doc__", "") or "").strip().split("\n")[0],
                }
            out[capability] = entries
        return out


registry = Registry()


def provider(capability: str, name: str, rank: int = 100, requires: str = ""):
    """Class decorator: register a provider implementation."""

    def deco(cls):
        cls.auto_rank = rank
        cls.requires = requires
        cls.name = name
        registry.register(capability, name, cls, auto_rank=rank)
        return cls

    return deco
