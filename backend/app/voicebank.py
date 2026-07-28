"""Voice bank: presets, cloning, design, and vector lookup.

Implements the "thousands of voices without thousands of models" model: one
universal synthesiser plus a small per-voice embedding + parameter set, stored
as rows in SQLite and searched by cosine similarity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import db
from .audio import dsp
from .audio.wavio import peak_normalize, read_wav, resample, to_mono, write_wav
from .config import settings
from .providers.base import VoiceRef
from .voice_design import PRESETS, VoiceParams, design_voice, params_from_embedding

MIN_CLONE_SECONDS = 3.0
RECOMMENDED_CLONE_SECONDS = 30.0
PROFESSIONAL_CLONE_SECONDS = 1800.0


class VoiceError(ValueError):
    """Raised when a clone request cannot be satisfied."""


@dataclass
class CloneReport:
    duration: float
    speech_ratio: float
    snr_db: float
    quality: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 2),
            "speech_ratio": round(self.speech_ratio, 3),
            "snr_db": round(self.snr_db, 1),
            "quality": self.quality,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------
def ensure_presets() -> int:
    """Seed the built-in studio + character voices (idempotent)."""
    inserted = 0
    with db.tx() as conn:
        for preset in PRESETS:
            exists = conn.execute("SELECT 1 FROM voices WHERE id=?", (preset.id,)).fetchone()
            if exists:
                continue
            params = preset.params
            conn.execute(
                "INSERT INTO voices (id, name, kind, category, language, prompt, params, "
                "embedding, provider, owner, tags, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (preset.id, preset.name, "preset", preset.category, preset.language,
                 preset.prompt, json.dumps(params.to_dict()),
                 db.pack_vector(_params_to_embedding(params)), "local", "system",
                 json.dumps(preset.tags), db.now()),
            )
            inserted += 1
    return inserted


PROBE_TEXT = ("The quick brown fox jumps over the lazy dog, "
              "while she sold seashells by the shore.")


def _params_to_embedding(params: VoiceParams) -> np.ndarray:
    """Embed a *designed* voice by rendering a probe utterance and running the
    same encoder used for clones.

    Deriving the vector analytically would put presets in a different space
    from cloned voices, and cosine similarity across the two would be
    meaningless. Synthesising a fixed probe line keeps one shared space, so
    "find the preset closest to this speaker" genuinely works.
    """
    from .audio import synth

    wav = synth.synthesize(PROBE_TEXT, params, settings.sample_rate, seed=7)
    return dsp.speaker_embedding(wav, settings.sample_rate)


# --------------------------------------------------------------------------
# reference-audio conditioning
# --------------------------------------------------------------------------
def prepare_reference(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, CloneReport]:
    """Clean a clone reference: mono, trimmed, de-hummed, normalised.

    Also grades the sample, because clone quality is dominated by input
    hygiene and users should be told before they hear a bad voice.
    """
    x = to_mono(np.asarray(audio, dtype=np.float32))
    if x.size == 0:
        raise VoiceError("reference audio is empty")

    duration = x.size / sample_rate
    warnings: list[str] = []

    # high-pass to kill rumble/DC, which otherwise poisons the pitch estimate
    x = x - float(np.mean(x))
    spans = dsp.detect_speech(x, sample_rate, min_speech=0.2, min_silence=0.2, sensitivity=1.2)
    speech = np.concatenate([x[int(s.start * sample_rate):int(s.end * sample_rate)]
                             for s in spans]) if spans else x
    speech_seconds = speech.size / sample_rate
    speech_ratio = speech_seconds / max(duration, 1e-6)

    noise_mask = np.ones(x.size, dtype=bool)
    for s in spans:
        noise_mask[int(s.start * sample_rate):int(s.end * sample_rate)] = False
    noise = x[noise_mask]
    sig_rms = float(np.sqrt(np.mean(speech ** 2) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(noise ** 2) + 1e-12)) if noise.size > sample_rate // 10 else 1e-5
    snr = 20.0 * float(np.log10(max(sig_rms, 1e-9) / max(noise_rms, 1e-9)))

    if speech_seconds < MIN_CLONE_SECONDS:
        raise VoiceError(
            f"need at least {MIN_CLONE_SECONDS:.0f}s of speech, found {speech_seconds:.1f}s"
        )
    if speech_seconds < 10:
        warnings.append("Under 10s of speech — clone accuracy will be limited.")
    if snr < 12:
        warnings.append(f"Noisy reference (~{snr:.0f} dB SNR). Record somewhere quieter.")
    if speech_ratio < 0.35:
        warnings.append("Mostly silence — trim the clip to continuous speech.")

    quality = "excellent" if speech_seconds >= RECOMMENDED_CLONE_SECONDS and snr >= 20 else \
              "good" if speech_seconds >= 10 and snr >= 14 else "fair"

    cleaned = peak_normalize(speech, 0.9)
    return cleaned, CloneReport(duration, speech_ratio, snr, quality, warnings)


# --------------------------------------------------------------------------
# creation
# --------------------------------------------------------------------------
def create_designed_voice(name: str, prompt: str, language: str = "en",
                          owner: str = "local", tags: list[str] | None = None,
                          overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prompt → voice, the Voice Design flow."""
    if not prompt.strip():
        raise VoiceError("a voice description is required")
    voice_id = db.new_id("voice")
    params = design_voice(prompt, seed=voice_id)
    if overrides:
        params = VoiceParams.from_dict({**params.to_dict(), **overrides}).clamp()
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO voices (id, name, kind, category, language, prompt, params, embedding, "
            "provider, owner, tags, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (voice_id, name.strip() or "Designed voice", "designed", "custom", language, prompt,
             json.dumps(params.to_dict()), db.pack_vector(_params_to_embedding(params)),
             "local", owner, json.dumps(tags or []), db.now()),
        )
    return get_voice(voice_id)  # type: ignore[return-value]


def create_cloned_voice(name: str, audio: np.ndarray, sample_rate: int,
                        kind: str = "instant", language: str = "en",
                        owner: str = "local", prompt: str = "",
                        tags: list[str] | None = None) -> dict[str, Any]:
    """Zero-shot / professional cloning: clean → embed → derive params → store."""
    cleaned, report = prepare_reference(audio, sample_rate)
    if kind == "professional" and cleaned.size / sample_rate < 60:
        report.warnings.append(
            "Professional clones normally use 30+ minutes of studio audio; "
            "this will behave like an instant clone."
        )

    voice_id = db.new_id("voice")
    embedding = dsp.speaker_embedding(cleaned, sample_rate)
    params = params_from_embedding(embedding)
    if prompt.strip():  # let a description refine the cloned timbre
        params = params.blend(design_voice(prompt, seed=voice_id), 0.35)
    params.clamp()

    ref_dir = settings.media_dir / "voices"
    ref_path = write_wav(ref_dir / f"{voice_id}.wav", cleaned, sample_rate)

    with db.tx() as conn:
        conn.execute(
            "INSERT INTO voices (id, name, kind, category, language, prompt, params, embedding, "
            "reference_path, provider, owner, tags, training_seconds, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (voice_id, name.strip() or "Cloned voice", kind, "cloned", language, prompt,
             json.dumps(params.to_dict()), db.pack_vector(embedding), str(ref_path),
             "local", owner, json.dumps(tags or []), cleaned.size / sample_rate, db.now()),
        )
    voice = get_voice(voice_id)
    assert voice is not None
    voice["report"] = report.to_dict()
    return voice


def create_cloned_voice_from_file(name: str, path: str | Path, **kwargs) -> dict[str, Any]:
    from .media import ffmpeg

    path = Path(path)
    tmp = settings.media_dir / "tmp" / f"{path.stem}_ref.wav"
    ffmpeg.extract_audio(path, tmp, settings.sample_rate)
    data, sr = read_wav(tmp)
    try:
        return create_cloned_voice(name, to_mono(data), sr, **kwargs)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# read / update
# --------------------------------------------------------------------------
def get_voice(voice_id: str) -> dict[str, Any] | None:
    row = db.get_conn().execute("SELECT * FROM voices WHERE id=?", (voice_id,)).fetchone()
    return db.row_to_dict(row, ("params", "tags"))


def list_voices(kind: str | None = None, category: str | None = None,
                query: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    sql = "SELECT * FROM voices WHERE 1=1"
    args: list[Any] = []
    if kind:
        sql += " AND kind = ?"
        args.append(kind)
    if category:
        sql += " AND category = ?"
        args.append(category)
    if query:
        sql += " AND (name LIKE ? OR prompt LIKE ? OR tags LIKE ?)"
        like = f"%{query}%"
        args += [like, like, like]
    sql += " ORDER BY (kind='preset') DESC, created_at DESC LIMIT ?"
    args.append(limit)
    rows = db.get_conn().execute(sql, args).fetchall()
    return [d for d in (db.row_to_dict(r, ("params", "tags")) for r in rows) if d]


def update_voice(voice_id: str, name: str | None = None, params: dict | None = None,
                 tags: list[str] | None = None) -> dict[str, Any] | None:
    current = get_voice(voice_id)
    if current is None:
        return None
    merged = VoiceParams.from_dict({**current["params"], **(params or {})}).clamp()
    with db.tx() as conn:
        conn.execute(
            "UPDATE voices SET name=?, params=?, tags=? WHERE id=?",
            (name or current["name"], json.dumps(merged.to_dict()),
             json.dumps(tags if tags is not None else current["tags"]), voice_id),
        )
    return get_voice(voice_id)


def delete_voice(voice_id: str) -> bool:
    row = db.get_conn().execute("SELECT reference_path, owner FROM voices WHERE id=?",
                                (voice_id,)).fetchone()
    if row is None:
        return False
    if row["owner"] == "system":
        raise VoiceError("built-in preset voices cannot be deleted")
    with db.tx() as conn:
        conn.execute("DELETE FROM voices WHERE id=?", (voice_id,))
    if row["reference_path"]:
        Path(row["reference_path"]).unlink(missing_ok=True)
    return True


def voice_ref(voice_id: str | None) -> VoiceRef:
    """Load a `VoiceRef` for the synthesis providers, with a safe default."""
    if voice_id:
        row = db.get_conn().execute("SELECT * FROM voices WHERE id=?", (voice_id,)).fetchone()
        if row is not None:
            return VoiceRef(
                id=row["id"], name=row["name"], kind=row["kind"],
                params=VoiceParams.from_dict(json.loads(row["params"] or "{}")),
                embedding=db.unpack_vector(row["embedding"]),
                reference_path=row["reference_path"], provider=row["provider"],
                provider_voice_id=row["provider_voice_id"], language=row["language"],
                prompt=row["prompt"] or "",
            )
    return VoiceRef(id="default", name="Default", params=VoiceParams())


def find_similar(embedding: np.ndarray, limit: int = 8) -> list[dict[str, Any]]:
    return db.search_voices_by_embedding(embedding, limit=limit)


def suggest_for_audio(audio: np.ndarray, sample_rate: int, limit: int = 5) -> list[dict[str, Any]]:
    """'Which of my voices sounds most like this speaker?' — used to
    auto-assign voices to diarized speakers."""
    return find_similar(dsp.speaker_embedding(to_mono(audio), sample_rate), limit=limit)
