"""Dependency-light WAV I/O and resampling.

We deliberately avoid libsndfile/torchaudio: the stdlib `wave` module plus
numpy covers every format this pipeline actually produces (PCM WAV), and
anything else is routed through ffmpeg when it is installed.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

__all__ = ["read_wav", "write_wav", "resample", "to_mono", "peak_normalize", "db_to_gain"]


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file as float32 in [-1, 1]. Returns (samples, sample_rate).

    Multi-channel input is returned as shape (channels, n); mono as (n,).
    """
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if width == 1:  # unsigned 8-bit
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:  # packed 24-bit
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        ints = np.where(ints & 0x800000, ints - (1 << 24), ints)
        data = ints.astype(np.float32) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:  # pragma: no cover - wave module rejects other widths first
        raise ValueError(f"unsupported sample width: {width}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).T.copy()
    return data, sr


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write float or int audio as 16-bit PCM WAV. Accepts (n,) or (channels, n)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        channels, interleaved = 1, x
    else:
        channels = x.shape[0]
        interleaved = x.T.reshape(-1)
    clipped = np.clip(interleaved, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return path


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32)
    return samples.mean(axis=0).astype(np.float32)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited-enough linear resampler; plenty for speech at 16-48 kHz."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)
    if samples.ndim == 2:
        return np.stack([resample(ch, src_rate, dst_rate) for ch in samples])
    duration = samples.shape[0] / float(src_rate)
    n_out = max(1, int(round(duration * dst_rate)))
    src_t = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_t = np.arange(n_out, dtype=np.float64) / dst_rate
    return np.interp(dst_t, src_t, samples).astype(np.float32)


def peak_normalize(samples: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < 1e-9:
        return samples.astype(np.float32)
    return (samples * (target / peak)).astype(np.float32)


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))
