"""Speech recognition providers."""
from __future__ import annotations

import re

import numpy as np

from ..audio import dsp
from ..config import settings
from .base import ASRProvider, Utterance, module_available, provider


@provider("asr", "faster_whisper", rank=10, requires="pip install faster-whisper")
class FasterWhisperASR:
    """Whisper via CTranslate2 — fastest CPU/GPU option, word timestamps."""

    @staticmethod
    def is_available() -> bool:
        return module_available("faster_whisper")

    def __init__(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore

        device = "cuda" if _cuda_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(settings.whisper_model, device=device, compute_type=compute)

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "auto"
                   ) -> list[Utterance]:
        mono = np.asarray(audio, dtype=np.float32)
        if sample_rate != 16000:
            from ..audio.wavio import resample
            mono = resample(mono, sample_rate, 16000)
        segments, info = self._model.transcribe(
            mono, language=None if language in ("auto", "") else language,
            vad_filter=True, word_timestamps=False,
        )
        detected = getattr(info, "language", language) or "en"
        return [
            Utterance(start=float(s.start), end=float(s.end), text=s.text.strip(),
                      language=detected, confidence=float(getattr(s, "avg_logprob", 0.0)))
            for s in segments if s.text.strip()
        ]


@provider("asr", "whisper", rank=20, requires="pip install openai-whisper")
class OpenAIWhisperASR:
    """Reference OpenAI Whisper implementation (PyTorch)."""

    @staticmethod
    def is_available() -> bool:
        return module_available("whisper")

    def __init__(self) -> None:
        import whisper  # type: ignore

        self._model = whisper.load_model(settings.whisper_model)

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "auto"
                   ) -> list[Utterance]:
        from ..audio.wavio import resample

        mono = resample(np.asarray(audio, dtype=np.float32), sample_rate, 16000)
        result = self._model.transcribe(mono, language=None if language == "auto" else language)
        lang = result.get("language", "en")
        return [
            Utterance(start=float(s["start"]), end=float(s["end"]), text=s["text"].strip(), language=lang)
            for s in result.get("segments", []) if s.get("text", "").strip()
        ]


@provider("asr", "openai_api", rank=30, requires="OPENAI_API_KEY")
class OpenAIAPIASR:
    """Hosted Whisper through any OpenAI-compatible /audio/transcriptions endpoint."""

    @staticmethod
    def is_available() -> bool:
        return bool(settings.openai_api_key)

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "auto"
                   ) -> list[Utterance]:
        import json
        import tempfile
        import urllib.request
        from pathlib import Path

        from ..audio.wavio import write_wav

        with tempfile.TemporaryDirectory() as tmp:
            path = write_wav(Path(tmp) / "a.wav", audio, sample_rate)
            body, content_type = _multipart({
                "model": "whisper-1",
                "response_format": "verbose_json",
                **({} if language == "auto" else {"language": language}),
            }, "file", "a.wav", path.read_bytes())
        req = urllib.request.Request(
            f"{settings.openai_base_url.rstrip('/')}/audio/transcriptions",
            data=body, method="POST",
            headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": content_type},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        lang = data.get("language", "en")
        segments = data.get("segments") or []
        if not segments and data.get("text"):
            return [Utterance(0.0, len(audio) / sample_rate, data["text"].strip(), language=lang)]
        return [Utterance(float(s["start"]), float(s["end"]), s["text"].strip(), language=lang)
                for s in segments if s.get("text", "").strip()]


@provider("asr", "offline", rank=90)
class OfflineASR:
    """VAD-segmented placeholder transcript — no model download required.

    It cannot invent the words that were spoken, so it emits correctly-timed
    empty segments and marks them `needs_transcript`. The transcript matrix in
    the UI (or `POST /projects/{id}/script`) is then used to fill them in,
    which is exactly the workflow used when a script already exists.
    """

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "auto"
                   ) -> list[Utterance]:
        spans = dsp.detect_speech(audio, sample_rate)
        lang = "en" if language == "auto" else language
        return [Utterance(s.start, s.end, "", language=lang, confidence=0.0) for s in spans]


def align_script(spans: list, script: str, language: str = "en") -> list[Utterance]:
    """Distribute a known script across detected speech spans by duration.

    Sentences are assigned to spans proportionally to span length, which is a
    solid forced-alignment approximation when the script is already correct.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…。！？])\s+", script.strip()) if s.strip()]
    if not spans:
        return []
    if not sentences:
        return [Utterance(s.start, s.end, "", language=language) for s in spans]

    total = sum(max(0.05, s.duration) for s in spans)
    weights = [max(0.05, s.duration) / total for s in spans]
    out: list[Utterance] = []
    cursor = 0
    remaining = len(sentences)
    for i, span in enumerate(spans):
        if i == len(spans) - 1:
            take = remaining
        else:
            take = int(round(weights[i] * len(sentences)))
            take = max(1 if remaining > (len(spans) - i - 1) else 0, min(take, remaining - (len(spans) - i - 1)))
        chunk = " ".join(sentences[cursor:cursor + take]).strip()
        cursor += take
        remaining -= take
        out.append(Utterance(span.start, span.end, chunk, language=language))
    return out


def _cuda_available() -> bool:
    if not module_available("torch"):
        return False
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _multipart(fields: dict[str, str], file_field: str, filename: str,
               content: bytes) -> tuple[bytes, str]:
    boundary = "----dubbing" + "".join(np.random.default_rng().choice(list("abcdef0123456789"), 16))
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
    )
    parts.append(content)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
