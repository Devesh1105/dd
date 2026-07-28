"""The dubbing pipeline.

    ingest -> stem split -> VAD + diarization -> ASR -> translation
           -> voice assignment -> TTS -> time-fit -> mix -> render

Each stage is a resumable step that reports progress through `JobContext`,
which the UI renders as "Step 3/9: Isolating background music...". Segment
work is chunked and fanned out across a thread pool so a long file is
processed in parallel rather than serially.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .. import db, voicebank
from ..audio import dsp
from ..audio.wavio import peak_normalize, read_wav, resample, to_mono, write_wav
from ..config import settings
from ..core.queue import JobContext, Step, manager
from ..media import ffmpeg
from ..providers import asr as asr_mod  # noqa: F401  (registers providers)
from ..providers import separation as sep_mod  # noqa: F401
from ..providers import translate as mt_mod
from ..providers import tts as tts_mod  # noqa: F401
from ..providers.base import Utterance, registry

DUB_STEPS = [
    Step("ingest", "Decoding source media", 1.0),
    Step("separate", "Isolating background music", 2.0),
    Step("segment", "Detecting speech and speakers", 1.5),
    Step("transcribe", "Transcribing dialogue", 3.0),
    Step("translate", "Translating and fitting length", 1.5),
    Step("voices", "Assigning voices", 0.5),
    Step("synthesize", "Generating dubbed speech", 4.0),
    Step("mix", "Mixing with background", 1.0),
    Step("render", "Rendering output", 1.0),
]

RENDER_STEPS = [
    Step("synthesize", "Generating dubbed speech", 4.0),
    Step("mix", "Mixing with background", 1.0),
    Step("render", "Rendering output", 1.0),
]


# --------------------------------------------------------------------------
# asset helpers
# --------------------------------------------------------------------------
def _set_asset(project_id: str, role: str, path: Path, meta: dict[str, Any] | None = None) -> None:
    with db.tx() as conn:
        conn.execute("DELETE FROM assets WHERE project_id=? AND role=?", (project_id, role))
        conn.execute(
            "INSERT INTO assets (id, project_id, role, path, meta, created_at) VALUES (?,?,?,?,?,?)",
            (db.new_id("ast"), project_id, role, str(path), json.dumps(meta or {}), db.now()),
        )


def get_asset(project_id: str, role: str) -> dict[str, Any] | None:
    row = db.get_conn().execute(
        "SELECT * FROM assets WHERE project_id=? AND role=? ORDER BY created_at DESC LIMIT 1",
        (project_id, role)).fetchone()
    return db.row_to_dict(row, ("meta",))


def _load_asset(project_id: str, role: str, sample_rate: int) -> np.ndarray | None:
    asset = get_asset(project_id, role)
    if not asset or not Path(asset["path"]).exists():
        return None
    data, sr = read_wav(asset["path"])
    return resample(to_mono(data), sr, sample_rate)


def _project(project_id: str) -> dict[str, Any]:
    row = db.get_conn().execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if row is None:
        raise ValueError(f"project {project_id} not found")
    out = db.row_to_dict(row, ("settings",))
    assert out is not None
    return out


def _set_status(project_id: str, status: str) -> None:
    with db.tx() as conn:
        conn.execute("UPDATE projects SET status=?, updated_at=? WHERE id=?",
                     (status, db.now(), project_id))


def _segments(project_id: str) -> list[dict[str, Any]]:
    rows = db.get_conn().execute(
        "SELECT * FROM segments WHERE project_id=? ORDER BY idx", (project_id,)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------
def step_ingest(ctx: JobContext, project: dict[str, Any]) -> np.ndarray:
    ctx.begin("ingest", "Decoding source media")
    sr = int(project["sample_rate"])
    source = Path(project["source_path"])
    if not source.exists():
        raise FileNotFoundError(f"source media missing: {source}")

    out_dir = settings.project_dir(project["id"])
    wav_path = ffmpeg.extract_audio(source, out_dir / "original.wav", sr)
    ctx.progress(0.6, "Analysing waveform")

    data, file_sr = read_wav(wav_path)
    mono = resample(to_mono(data), file_sr, sr)
    duration = mono.size / sr
    _set_asset(project["id"], "original", wav_path,
               {"duration": duration, "sample_rate": sr,
                "peaks": ffmpeg.waveform_peaks(mono, 1600),
                "lufs": round(dsp.lufs_estimate(mono, sr), 1)})
    if ffmpeg.is_video(source):
        _set_asset(project["id"], "video", source, {})
    with db.tx() as conn:
        conn.execute("UPDATE projects SET duration=?, updated_at=? WHERE id=?",
                     (duration, db.now(), project["id"]))
    ctx.progress(1.0, f"Loaded {duration:.1f}s of audio")
    return mono


def step_separate(ctx: JobContext, project: dict[str, Any], audio: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    ctx.begin("separate", "Isolating background music")
    sr = int(project["sample_rate"])
    opts = project["settings"] or {}
    if not opts.get("separate_background", True):
        ctx.progress(1.0, "Stem split disabled — using original as background")
        silence = np.zeros_like(audio)
        _set_asset(project["id"], "vocals", write_wav(
            settings.project_dir(project["id"]) / "vocals.wav", audio, sr), {})
        _set_asset(project["id"], "background", write_wav(
            settings.project_dir(project["id"]) / "background.wav", silence, sr), {})
        return audio, silence

    provider = registry.get("separation", settings.separation_provider)
    ctx.log(f"separation provider: {provider.name}")
    # process in chunks so progress moves on long files
    chunk = int(30 * sr)
    vocals_parts, bg_parts = [], []
    total = max(1, int(np.ceil(audio.size / chunk)))
    for i in range(total):
        ctx.check_cancelled()
        seg = audio[i * chunk:(i + 1) * chunk]
        if seg.size == 0:
            continue
        v, b = provider.separate(seg, sr)
        vocals_parts.append(v[:seg.size])
        bg_parts.append(b[:seg.size])
        ctx.progress((i + 1) / total, f"Isolating background music ({i + 1}/{total})")
    vocals = np.concatenate(vocals_parts) if vocals_parts else audio
    background = np.concatenate(bg_parts) if bg_parts else np.zeros_like(audio)

    out_dir = settings.project_dir(project["id"])
    _set_asset(project["id"], "vocals", write_wav(out_dir / "vocals.wav", vocals, sr),
               {"peaks": ffmpeg.waveform_peaks(vocals, 1200)})
    _set_asset(project["id"], "background", write_wav(out_dir / "background.wav", background, sr),
               {"peaks": ffmpeg.waveform_peaks(background, 1200)})
    return vocals, background


def step_segment(ctx: JobContext, project: dict[str, Any], vocals: np.ndarray) -> list[dsp.Span]:
    ctx.begin("segment", "Detecting speech")
    sr = int(project["sample_rate"])
    opts = project["settings"] or {}
    spans = dsp.detect_speech(
        vocals, sr,
        min_speech=float(opts.get("min_speech", 0.35)),
        min_silence=float(opts.get("min_silence", 0.45)),
        sensitivity=float(opts.get("vad_sensitivity", 1.0)),
        max_segment=float(opts.get("max_segment", settings.chunk_seconds)),
    )
    if not spans:
        spans = [dsp.Span(0.0, vocals.size / sr)]
    ctx.progress(0.4, f"Found {len(spans)} speech segments")

    # diarization: embed each span, cluster, then persist speaker profiles
    max_speakers = int(opts.get("max_speakers", 6))
    embeddings = [dsp.speaker_embedding(vocals[int(s.start * sr):int(s.end * sr)], sr) for s in spans]
    labels = cluster_labels(embeddings, max_speakers, float(opts.get("diarization_threshold", 0.91)))
    ctx.progress(0.8, f"Identified {len(set(labels))} speaker(s)")

    by_speaker: dict[str, list[np.ndarray]] = {}
    seconds: dict[str, float] = {}
    for span, label in zip(spans, labels):
        name = f"SPK_{label + 1}"
        by_speaker.setdefault(name, []).append(dsp.speaker_embedding(
            vocals[int(span.start * sr):int(span.end * sr)], sr))
        seconds[name] = seconds.get(name, 0.0) + span.duration

    with db.tx() as conn:
        conn.execute("DELETE FROM speakers WHERE project_id=?", (project["id"],))
        for name, embs in by_speaker.items():
            centroid = np.mean(np.stack(embs), axis=0)
            conn.execute(
                "INSERT INTO speakers (project_id, speaker, embedding, total_seconds) VALUES (?,?,?,?)",
                (project["id"], name, db.pack_vector(centroid), seconds.get(name, 0.0)),
            )

    ctx.result["speakers"] = sorted(by_speaker)
    ctx.result["speaker_labels"] = [f"SPK_{l + 1}" for l in labels]
    ctx.progress(1.0, f"{len(spans)} segments, {len(by_speaker)} speaker(s)")
    return spans


def cluster_labels(embeddings: list[np.ndarray], max_speakers: int, threshold: float) -> list[int]:
    if max_speakers <= 1:
        return [0] * len(embeddings)
    return dsp.cluster_speakers(embeddings, max_speakers=max_speakers, threshold=threshold)


def step_transcribe(ctx: JobContext, project: dict[str, Any], vocals: np.ndarray,
                    spans: list[dsp.Span]) -> list[Utterance]:
    ctx.begin("transcribe", "Transcribing dialogue")
    sr = int(project["sample_rate"])
    opts = project["settings"] or {}
    script = (opts.get("script") or "").strip()

    if script:
        ctx.log("using supplied script for alignment")
        utterances = asr_mod.align_script(spans, script, project["source_language"])
        ctx.progress(1.0, f"Aligned script across {len(utterances)} segments")
        return utterances

    provider = registry.get("asr", settings.asr_provider)
    ctx.log(f"ASR provider: {provider.name}")
    utterances = provider.transcribe(vocals, sr, project["source_language"] or "auto")

    if provider.name == "offline":
        ctx.log("No ASR model installed — segments created without text. "
                "Add a transcript in the editor or install faster-whisper.")
    ctx.progress(1.0, f"{len(utterances)} lines transcribed")
    return utterances or [Utterance(s.start, s.end, "") for s in spans]


def step_translate(ctx: JobContext, project: dict[str, Any], utterances: list[Utterance]
                   ) -> list[str]:
    ctx.begin("translate", "Translating dialogue")
    source = project["source_language"] or "auto"
    target = project["target_language"]
    if source == "auto":
        joined = " ".join(u.text for u in utterances if u.text)[:400]
        source = mt_mod.detect_language(joined) if joined else "en"

    texts = [u.text for u in utterances]
    if not any(t.strip() for t in texts):
        ctx.progress(1.0, "No text to translate")
        return texts
    if source == target:
        ctx.progress(1.0, "Source and target language match — skipping translation")
        return texts

    budgets = [mt_mod.char_budget(u.end - u.start, target) for u in utterances]
    provider = registry.get("mt", settings.mt_provider)
    ctx.log(f"MT provider: {provider.name} ({source} → {target})")

    out: list[str] = []
    batch = 24
    for i in range(0, len(texts), batch):
        ctx.check_cancelled()
        out.extend(provider.translate(texts[i:i + batch], source, target, budgets[i:i + batch]))
        ctx.progress(min(1.0, (i + batch) / max(1, len(texts))),
                     f"Translating {min(i + batch, len(texts))}/{len(texts)}")
    if provider.name == "passthrough":
        ctx.log("No translation engine configured — text passed through untranslated.")
    return out


def step_assign_voices(ctx: JobContext, project: dict[str, Any], speakers: list[str]) -> dict[str, str]:
    ctx.begin("voices", "Assigning voices")
    opts = project["settings"] or {}
    mapping: dict[str, str] = dict(opts.get("voice_map") or {})

    rows = db.get_conn().execute("SELECT * FROM speakers WHERE project_id=?",
                                 (project["id"],)).fetchall()
    for row in rows:
        name = row["speaker"]
        if mapping.get(name):
            continue
        if opts.get("clone_speakers", True):
            mapping[name] = _clone_speaker_voice(project, name, row)
        else:
            emb = db.unpack_vector(row["embedding"])
            matches = voicebank.find_similar(emb, limit=1) if emb is not None else []
            mapping[name] = matches[0]["id"] if matches else "studio_anchor_m"

    with db.tx() as conn:
        for name, voice_id in mapping.items():
            conn.execute("UPDATE speakers SET voice_id=? WHERE project_id=? AND speaker=?",
                         (voice_id, project["id"], name))
        merged = {**(project["settings"] or {}), "voice_map": mapping}
        conn.execute("UPDATE projects SET settings=? WHERE id=?",
                     (json.dumps(merged), project["id"]))
    ctx.progress(1.0, f"Voices assigned for {len(mapping)} speaker(s)")
    return mapping


def _clone_speaker_voice(project: dict[str, Any], speaker: str, row) -> str:
    """Cross-lingual clone: keep the original speaker's identity in the dub."""
    sr = int(project["sample_rate"])
    vocals = _load_asset(project["id"], "vocals", sr)
    segs = [s for s in _segments(project["id"]) if s["speaker"] == speaker]
    if vocals is None or not segs:
        emb = db.unpack_vector(row["embedding"])
        matches = voicebank.find_similar(emb, limit=1) if emb is not None else []
        return matches[0]["id"] if matches else "studio_anchor_m"

    clips = [vocals[int(s["start"] * sr):int(s["end"] * sr)] for s in segs[:40]]
    reference = np.concatenate([c for c in clips if c.size]) if clips else vocals
    try:
        voice = voicebank.create_cloned_voice(
            f"{project['name']} · {speaker}", reference, sr, kind="instant",
            language=project["target_language"], owner=project["id"],
            tags=["auto-clone", project["id"]],
        )
        return voice["id"]
    except voicebank.VoiceError:
        emb = db.unpack_vector(row["embedding"])
        matches = voicebank.find_similar(emb, limit=1) if emb is not None else []
        return matches[0]["id"] if matches else "studio_anchor_m"


def step_synthesize(ctx: JobContext, project: dict[str, Any]) -> np.ndarray:
    """Render every segment and lay it on the dub timeline.

    Segments are independent, so they fan out across the executor — this is
    the chunk-and-parallelise stage that keeps long files tractable.
    """
    ctx.begin("synthesize", "Generating dubbed speech")
    sr = int(project["sample_rate"])
    opts = project["settings"] or {}
    segments = _segments(project["id"])
    if not segments:
        ctx.progress(1.0, "Nothing to synthesize")
        return np.zeros(int(project["duration"] * sr) or sr, dtype=np.float32)

    tts = registry.get("tts", settings.tts_provider)
    ctx.log(f"TTS provider: {tts.name}")
    voice_refs = {vid: voicebank.voice_ref(vid)
                  for vid in {s["voice_id"] for s in segments if s["voice_id"]}}
    default_ref = voicebank.voice_ref(None)
    out_dir = settings.project_dir(project["id"]) / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    global_speed = float(opts.get("speed", 1.0))
    allow_overflow = bool(opts.get("allow_overflow", False))
    done = {"n": 0}

    def render(seg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, float]:
        text = (seg["target_text"] or seg["source_text"] or "").strip()
        target_seconds = max(0.12, seg["end"] - seg["start"])
        if not text:
            return seg, np.zeros(int(target_seconds * sr), dtype=np.float32), 1.0
        ref = voice_refs.get(seg["voice_id"], default_ref)
        wav = tts.synthesize(text, ref, sr, emotion=seg["emotion"] or "neutral",
                             intensity=float(opts.get("emotion_intensity", 1.0)),
                             speed=global_speed)
        if allow_overflow:
            return seg, wav.astype(np.float32), 1.0
        fitted, rate = dsp.fit_to_duration(wav, target_seconds, sr)
        return seg, fitted, rate

    results: list[tuple[dict[str, Any], np.ndarray, float]] = []
    workers = max(1, min(settings.workers * 2, 8))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tts") as pool:
        for seg, wav, rate in pool.map(render, segments):
            ctx.check_cancelled()
            results.append((seg, wav, rate))
            done["n"] += 1
            ctx.progress(done["n"] / len(segments),
                         f"Generating dubbed speech ({done['n']}/{len(segments)})")

    total_n = max(int(project["duration"] * sr), 1)
    for seg, wav, _ in results:
        total_n = max(total_n, int(seg["start"] * sr) + wav.size)
    timeline = np.zeros(total_n, dtype=np.float32)

    with db.tx() as conn:
        for seg, wav, rate in results:
            start = int(seg["start"] * sr)
            end = min(total_n, start + wav.size)
            if end > start:
                timeline[start:end] += wav[:end - start]
            path = out_dir / f"{seg['idx']:05d}.wav"
            write_wav(path, wav, sr)
            conn.execute("UPDATE segments SET audio_path=?, fit_rate=? WHERE id=?",
                         (str(path), float(rate), seg["id"]))

    timeline = dsp.limiter(timeline)
    _set_asset(project["id"], "dubbed",
               write_wav(settings.project_dir(project["id"]) / "dubbed.wav", timeline, sr),
               {"peaks": ffmpeg.waveform_peaks(timeline, 1600)})
    return timeline


def step_mix(ctx: JobContext, project: dict[str, Any], dubbed: np.ndarray) -> np.ndarray:
    ctx.begin("mix", "Mixing with background")
    sr = int(project["sample_rate"])
    opts = project["settings"] or {}
    background = _load_asset(project["id"], "background", sr)
    if background is None:
        background = np.zeros_like(dubbed)

    n = max(dubbed.size, background.size)
    speech = np.pad(dubbed, (0, n - dubbed.size))
    bg = np.pad(background, (0, n - background.size))

    bg_gain = float(opts.get("background_gain", 1.0))
    if bool(opts.get("duck_background", True)):
        bg = dsp.duck(bg, speech, sr, depth_db=float(opts.get("duck_depth_db", -11.0)))
    ctx.progress(0.5, "Applying room tone")

    room = float(opts.get("reverb", 0.10))
    if room > 0.001:
        speech = dsp.reverb(speech, sr, amount=room, room=float(opts.get("room_size", 0.5)))

    mixed = dsp.limiter(speech * float(opts.get("speech_gain", 1.0)) + bg * bg_gain)
    target_lufs = float(opts.get("target_lufs", -16.0))
    current = dsp.lufs_estimate(mixed, sr)
    if current > -60:
        mixed = dsp.limiter(mixed * float(np.clip(10 ** ((target_lufs - current) / 20.0), 0.2, 4.0)))
    ctx.progress(1.0, f"Mixed at {dsp.lufs_estimate(mixed, sr):.1f} LUFS")
    return mixed


def step_render(ctx: JobContext, project: dict[str, Any], mixed: np.ndarray) -> dict[str, Any]:
    ctx.begin("render", "Rendering output")
    sr = int(project["sample_rate"])
    out_dir = settings.project_dir(project["id"])
    wav_path = write_wav(out_dir / "mixed.wav", mixed, sr)
    _set_asset(project["id"], "mixed", wav_path,
               {"peaks": ffmpeg.waveform_peaks(mixed, 1600),
                "lufs": round(dsp.lufs_estimate(mixed, sr), 1),
                "duration": mixed.size / sr})
    outputs: dict[str, Any] = {"audio": str(wav_path)}

    mp3 = ffmpeg.to_mp3(wav_path, out_dir / "mixed.mp3")
    if mp3:
        outputs["mp3"] = str(mp3)

    video_asset = get_asset(project["id"], "video")
    if video_asset and Path(video_asset["path"]).exists():
        ctx.progress(0.4, "Muxing dubbed audio into video")
        lip = registry.get("lipsync", settings.lipsync_provider)
        final_video = None
        if lip.name != "none":
            ctx.progress(0.5, "Re-aligning lip movements")
            final_video = lip.sync(video_asset["path"], str(wav_path), str(out_dir / "dubbed_lip.mp4"))
        if final_video is None:
            final_video = ffmpeg.mux_audio(video_asset["path"], wav_path, out_dir / "dubbed.mp4")
        if final_video:
            _set_asset(project["id"], "output_video", Path(final_video), {})
            outputs["video"] = str(final_video)
        else:
            ctx.log("ffmpeg not available — video output skipped, audio still rendered.")

    ctx.progress(1.0, "Render complete")
    return outputs


# --------------------------------------------------------------------------
# job entry points
# --------------------------------------------------------------------------
@manager.register("dub", DUB_STEPS)
def run_dub(ctx: JobContext) -> dict[str, Any]:
    project = _project(ctx.project_id)
    _set_status(project["id"], "processing")

    audio = step_ingest(ctx, project)
    vocals, _ = step_separate(ctx, project, audio)
    spans = step_segment(ctx, project, vocals)
    utterances = step_transcribe(ctx, project, vocals, spans)
    translations = step_translate(ctx, project, utterances)

    speaker_labels = ctx.result.get("speaker_labels") or []
    _write_segments(project, utterances, translations, spans, speaker_labels)

    project = _project(ctx.project_id)
    mapping = step_assign_voices(ctx, project, ctx.result.get("speakers", []))
    _apply_voice_map(project["id"], mapping)

    project = _project(ctx.project_id)
    dubbed = step_synthesize(ctx, project)
    mixed = step_mix(ctx, project, dubbed)
    outputs = step_render(ctx, project, mixed)

    _set_status(project["id"], "ready")
    return {"outputs": outputs, "segments": len(_segments(project["id"])),
            "speakers": mapping, "duration": project["duration"]}


@manager.register("render", RENDER_STEPS)
def run_render(ctx: JobContext) -> dict[str, Any]:
    """Re-render after transcript/voice/timing edits — skips ASR and MT."""
    project = _project(ctx.project_id)
    _set_status(project["id"], "processing")
    dubbed = step_synthesize(ctx, project)
    mixed = step_mix(ctx, project, dubbed)
    outputs = step_render(ctx, project, mixed)
    _set_status(project["id"], "ready")
    return {"outputs": outputs, "segments": len(_segments(project["id"]))}


def _write_segments(project: dict[str, Any], utterances: list[Utterance], translations: list[str],
                    spans: list[dsp.Span], speaker_labels: list[str]) -> None:
    """Persist the transcript matrix, mapping each line onto a speaker."""
    def speaker_for(u: Utterance) -> str:
        if not speaker_labels:
            return "SPK_1"
        mid = (u.start + u.end) / 2.0
        best, best_d = 0, float("inf")
        for i, span in enumerate(spans):
            d = abs((span.start + span.end) / 2.0 - mid)
            if d < best_d:
                best, best_d = i, d
        return speaker_labels[best] if best < len(speaker_labels) else "SPK_1"

    with db.tx() as conn:
        conn.execute("DELETE FROM segments WHERE project_id=?", (project["id"],))
        for i, u in enumerate(utterances):
            target = translations[i] if i < len(translations) else u.text
            conn.execute(
                "INSERT INTO segments (id, project_id, idx, start, end, speaker, source_text, "
                "target_text, emotion) VALUES (?,?,?,?,?,?,?,?,?)",
                (db.new_id("seg"), project["id"], i, float(u.start), float(u.end),
                 speaker_for(u), u.text, target, "neutral"),
            )


def _apply_voice_map(project_id: str, mapping: dict[str, str]) -> None:
    with db.tx() as conn:
        for speaker, voice_id in mapping.items():
            conn.execute(
                "UPDATE segments SET voice_id=? WHERE project_id=? AND speaker=? AND voice_id IS NULL",
                (voice_id, project_id, speaker),
            )
