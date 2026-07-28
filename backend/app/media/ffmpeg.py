"""Optional ffmpeg integration.

ffmpeg unlocks MP3/MP4/MOV input and video muxing. Without it the platform
still runs end-to-end on WAV audio — every call here degrades gracefully.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..audio.wavio import read_wav, resample, to_mono, write_wav

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def have_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def probe(path: str | Path) -> dict:
    if not have_ffprobe():
        return {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        return json.loads(out.stdout) if out.returncode == 0 else {}
    except Exception:
        return {}


def is_video(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix.lower() in VIDEO_EXT:
        return True
    info = probe(path)
    return any(s.get("codec_type") == "video" for s in info.get("streams", []))


def extract_audio(src: str | Path, dst: str | Path, sample_rate: int) -> Path:
    """Decode any media file to mono WAV at `sample_rate`.

    Falls back to the stdlib WAV reader when ffmpeg is unavailable, which
    covers the WAV-only local workflow.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if have_ffmpeg():
        cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate),
               "-acodec", "pcm_s16le", str(dst)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode == 0 and dst.exists():
            return dst
        raise RuntimeError(f"ffmpeg failed to decode {src.name}: {proc.stderr[-400:]}")

    if src.suffix.lower() != ".wav":
        raise RuntimeError(
            f"{src.suffix or 'this format'} needs ffmpeg. Install ffmpeg, or upload a .wav file."
        )
    data, sr = read_wav(src)
    write_wav(dst, resample(to_mono(data), sr, sample_rate), sample_rate)
    return dst


def mux_audio(video: str | Path, audio: str | Path, dst: str | Path) -> Path | None:
    """Replace a video's audio track. Returns None without ffmpeg."""
    if not have_ffmpeg():
        return None
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-c:v", "copy", "-map", "0:v:0",
           "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    return dst if proc.returncode == 0 and dst.exists() else None


def to_mp3(src: str | Path, dst: str | Path, bitrate: str = "192k") -> Path | None:
    if not have_ffmpeg():
        return None
    proc = subprocess.run(["ffmpeg", "-y", "-i", str(src), "-b:a", bitrate, str(dst)],
                          capture_output=True, text=True, timeout=1800)
    return Path(dst) if proc.returncode == 0 else None


def waveform_peaks(samples: np.ndarray, buckets: int = 1600) -> list[float]:
    """Min/max envelope for the canvas waveform, downsampled for the wire."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return []
    buckets = max(1, min(buckets, x.size))
    pad = (-x.size) % buckets
    if pad:
        x = np.pad(x, (0, pad))
    reshaped = x.reshape(buckets, -1)
    peaks = np.max(np.abs(reshaped), axis=1)
    return [round(float(v), 4) for v in peaks]
