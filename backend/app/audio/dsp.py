"""Signal processing used across the dubbing pipeline.

Contains the pieces that are genuinely algorithmic rather than model-based:
STFT/mel analysis, energy VAD, speaker embeddings, WSOLA time-stretching
(for fitting dubbed speech into the original timestamps), vocal/background
stem separation and the mix bus.

All of it is numpy-only so it runs on a CPU-only laptop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMBED_DIM = 64


# --------------------------------------------------------------------------
# spectral analysis
# --------------------------------------------------------------------------
def frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if x.shape[0] < frame:
        x = np.pad(x, (0, frame - x.shape[0]))
    n_frames = 1 + (x.shape[0] - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def stft(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Returns complex spectrogram of shape (frames, bins)."""
    window = np.hanning(n_fft).astype(np.float32)
    frames = frame_signal(np.asarray(x, dtype=np.float32), n_fft, hop) * window
    return np.fft.rfft(frames, axis=1)


def istft(spec: np.ndarray, n_fft: int = 1024, hop: int = 256, length: int | None = None) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * window
    n = (frames.shape[0] - 1) * hop + n_fft
    out = np.zeros(n, dtype=np.float32)
    norm = np.zeros(n, dtype=np.float32)
    for i in range(frames.shape[0]):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        norm[s:s + n_fft] += window ** 2
    valid = norm > 0.05 * float(norm.max() or 1.0)
    out = np.where(valid, out / np.maximum(norm, 1e-8), 0.0).astype(np.float32)
    if length is not None:
        out = out[:length] if out.shape[0] >= length else np.pad(out, (0, length - out.shape[0]))
    return out.astype(np.float32)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 40, fmin: float = 60.0,
                   fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sample_rate / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    freqs = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            fb[i, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        if hi > mid:
            fb[i, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return fb


def log_mel(x: np.ndarray, sample_rate: int, n_fft: int = 1024, hop: int = 256,
            n_mels: int = 40) -> np.ndarray:
    mag = np.abs(stft(x, n_fft, hop))
    fb = mel_filterbank(sample_rate, n_fft, n_mels)
    return np.log(np.maximum(mag @ fb.T, 1e-8))


# --------------------------------------------------------------------------
# voice activity detection
# --------------------------------------------------------------------------
@dataclass
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def detect_speech(x: np.ndarray, sample_rate: int, frame_ms: float = 30.0,
                  min_speech: float = 0.35, min_silence: float = 0.45,
                  sensitivity: float = 1.0, max_segment: float = 12.0) -> list[Span]:
    """Energy + spectral-flatness VAD with hysteresis.

    `sensitivity` > 1 makes the detector greedier (keeps quieter speech).
    `max_segment` caps utterance length so one continuous monologue still gets
    chunked into dubbable, parallelisable pieces.
    """
    if x.size == 0:
        return []
    frame = max(64, int(sample_rate * frame_ms / 1000.0))
    hop = frame // 2
    frames = frame_signal(x, frame, hop)
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    # spectral flatness separates tonal speech from broadband hiss
    mag = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1)) + 1e-9
    flatness = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)

    noise_floor = float(np.percentile(energy, 15))
    peak = float(np.percentile(energy, 95))
    threshold = (noise_floor + 0.18 * (peak - noise_floor)) / max(sensitivity, 1e-3)
    active = (energy > threshold) & (flatness < 0.55)

    spans: list[Span] = []
    in_speech = False
    start_i = 0
    silence_run = 0
    max_silence_frames = int(min_silence * sample_rate / hop)
    for i, flag in enumerate(active):
        if flag:
            if not in_speech:
                in_speech, start_i = True, i
            silence_run = 0
        elif in_speech:
            silence_run += 1
            if silence_run > max_silence_frames:
                spans.append(Span(start_i * hop / sample_rate, (i - silence_run) * hop / sample_rate))
                in_speech = False
    if in_speech:
        spans.append(Span(start_i * hop / sample_rate, len(active) * hop / sample_rate))

    merged: list[Span] = []
    for s in spans:
        if s.duration < min_speech:
            continue
        if merged and s.start - merged[-1].end < min_silence \
                and (s.end - merged[-1].start) <= max_segment:
            merged[-1] = Span(merged[-1].start, s.end)
        else:
            merged.append(s)

    # split anything still longer than max_segment at its quietest interior point
    out: list[Span] = []
    for s in merged:
        queue = [s]
        while queue:
            span = queue.pop(0)
            if span.duration <= max_segment:
                out.append(span)
                continue
            lo = int((span.start + min_speech) * sample_rate / hop)
            hi = int((span.end - min_speech) * sample_rate / hop)
            if hi <= lo:
                out.append(span)
                continue
            cut = (lo + int(np.argmin(energy[lo:hi]))) * hop / sample_rate
            queue = [Span(span.start, cut), Span(cut, span.end)] + queue
    return out


# --------------------------------------------------------------------------
# pitch + speaker embedding
# --------------------------------------------------------------------------
def estimate_f0(x: np.ndarray, sample_rate: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """Median autocorrelation pitch over voiced frames, in Hz (0.0 if unvoiced)."""
    if x.size < sample_rate // 20:
        return 0.0
    frame = int(sample_rate * 0.04)
    hop = frame // 2
    frames = frame_signal(x, frame, hop)
    lo, hi = int(sample_rate / fmax), int(sample_rate / fmin)
    picks: list[float] = []
    for f in frames:
        if np.sqrt(np.mean(f ** 2)) < 1e-3:
            continue
        f = f - f.mean()
        corr = np.correlate(f, f, mode="full")[frame - 1:]
        if corr[0] <= 0:
            continue
        seg = corr[lo:hi]
        if seg.size == 0:
            continue
        lag = int(np.argmax(seg)) + lo
        if corr[lag] / corr[0] > 0.3:
            picks.append(sample_rate / lag)
    return float(np.median(picks)) if picks else 0.0


def speaker_embedding(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Deterministic, model-free speaker embedding.

    Concatenates mean/std log-mel statistics (timbre), delta statistics
    (articulation rate) and normalised pitch/energy descriptors, then
    L2-normalises. It is not x-vector accurate, but it is stable, cheap and
    good enough to cluster speakers and to drive the offline synthesiser.
    Swap in a real encoder by registering a different provider.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size < sample_rate // 10:
        x = np.pad(x, (0, max(0, sample_rate // 10 - x.size)))
    lm = log_mel(x, sample_rate, n_mels=20)
    mean = lm.mean(axis=0)
    std = lm.std(axis=0)
    delta = np.abs(np.diff(lm, axis=0)).mean(axis=0) if lm.shape[0] > 1 else np.zeros_like(mean)

    # Raw log-mel values are large, negative and highly correlated, which makes
    # every cosine similarity ~1.0 and clustering useless. Z-normalising each
    # block turns them into level-invariant *shape* descriptors, which is what
    # actually distinguishes speakers.
    def _shape(v: np.ndarray) -> np.ndarray:
        centred = v - float(np.mean(v))
        scale = float(np.std(centred))
        return centred / scale if scale > 1e-6 else centred

    mean, std, delta = _shape(mean), _shape(std), _shape(delta)
    f0 = estimate_f0(x, sample_rate)
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    # pitch is the single strongest speaker cue, so it is weighted above the
    # 20-band shape blocks rather than being one dimension among sixty
    extras = np.array([
        3.5 * (f0 / 200.0 - 1.0),
        np.log1p(rms * 100) / 5.0,
        2.0 * float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),  # zero-crossing rate
        1.5 * float(np.mean(mean[-6:]) - np.mean(mean[:6])),    # spectral tilt
    ], dtype=np.float32)
    # 20 + 20 + 20 + 4 lands exactly on EMBED_DIM, so nothing is truncated
    vec = np.concatenate([mean, std, delta, extras]).astype(np.float32)[:EMBED_DIM]
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 1e-9 else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cluster_speakers(embeddings: list[np.ndarray], max_speakers: int = 6,
                     threshold: float = 0.91) -> list[int]:
    """Greedy agglomerative clustering on cosine similarity → speaker ids."""
    labels: list[int] = []
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    for emb in embeddings:
        best, best_sim = -1, -1.0
        for i, c in enumerate(centroids):
            sim = cosine(emb, c)
            if sim > best_sim:
                best, best_sim = i, sim
        if best >= 0 and (best_sim >= threshold or len(centroids) >= max_speakers):
            labels.append(best)
            centroids[best] = (centroids[best] * counts[best] + emb) / (counts[best] + 1)
            counts[best] += 1
        else:
            labels.append(len(centroids))
            centroids.append(emb.copy())
            counts.append(1)
    return labels


# --------------------------------------------------------------------------
# time-scale modification (pacing / time-sync)
# --------------------------------------------------------------------------
def time_stretch(x: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
    """WSOLA time-scaling: `rate` > 1 speeds up, keeping pitch intact."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or abs(rate - 1.0) < 1e-3:
        return x
    rate = float(np.clip(rate, 0.25, 4.0))
    frame = max(256, int(sample_rate * 0.040))
    syn_hop = frame // 2
    ana_hop = int(round(syn_hop * rate))
    search = max(1, int(sample_rate * 0.006))
    window = np.hanning(frame).astype(np.float32)

    out_len = int(np.ceil(x.size / rate)) + frame
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    ana_pos, syn_pos, offset = 0, 0, 0
    while ana_pos + frame + search < x.size and syn_pos + frame < out_len:
        start = int(np.clip(ana_pos + offset, 0, x.size - frame))
        seg = x[start:start + frame] * window
        out[syn_pos:syn_pos + frame] += seg
        norm[syn_pos:syn_pos + frame] += window ** 2

        # natural continuation of what we just wrote
        tail_start = start + syn_hop
        target = x[tail_start:tail_start + frame]
        next_center = ana_pos + ana_hop
        lo = int(np.clip(next_center - search, 0, max(0, x.size - frame)))
        hi = int(np.clip(next_center + search, 0, max(0, x.size - frame)))
        best_off, best_score = 0, -np.inf
        if target.size == frame:
            for cand in range(lo, hi + 1, max(1, search // 8)):
                c = x[cand:cand + frame]
                if c.size != frame:
                    continue
                score = float(np.dot(c, target))
                if score > best_score:
                    best_score, best_off = score, cand - next_center
        offset = best_off
        ana_pos = next_center
        syn_pos += syn_hop

    # WSOLA leaves a partially-covered tail; dividing there amplifies clicks
    covered = norm > 0.25 * float(norm.max() or 1.0)
    out = np.where(covered, out / np.maximum(norm, 1e-6), 0.0).astype(np.float32)
    target_len = max(1, int(round(x.size / rate)))
    if out.size >= target_len:
        return out[:target_len]
    return np.pad(out, (0, target_len - out.size)).astype(np.float32)


def fit_to_duration(x: np.ndarray, target_seconds: float, sample_rate: int,
                    max_speedup: float = 1.45, max_slowdown: float = 0.72) -> tuple[np.ndarray, float]:
    """Fit a clip into a timestamp slot; returns (audio, applied_rate).

    Beyond the stretch limits we pad with silence rather than making speech
    unintelligible — matching what human dubbing studios do.
    """
    target_n = max(1, int(round(target_seconds * sample_rate)))
    if x.size == 0:
        return np.zeros(target_n, dtype=np.float32), 1.0
    rate = x.size / target_n
    applied = float(np.clip(rate, max_slowdown, max_speedup))
    y = time_stretch(x, applied, sample_rate) if abs(applied - 1.0) > 1e-3 else x
    if y.size > target_n:
        fade = min(int(0.01 * sample_rate), target_n)
        y = y[:target_n].copy()
        if fade > 1:
            y[-fade:] *= np.linspace(1.0, 0.0, fade)
    else:
        y = np.pad(y, (0, target_n - y.size))
    return limiter(y.astype(np.float32)), applied


# --------------------------------------------------------------------------
# stem separation (vocals vs background)
# --------------------------------------------------------------------------
def _median_filter(mat: np.ndarray, size: int, axis: int) -> np.ndarray:
    if size <= 1:
        return mat
    pad = size // 2
    padded = np.pad(mat, [(pad, pad) if a == axis else (0, 0) for a in range(mat.ndim)], mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=axis)
    return np.median(windows, axis=-1)


def separate_stems(x: np.ndarray, sample_rate: int, n_fft: int = 2048, hop: int = 512,
                   background_seconds: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    """Stationary-background subtraction → (vocals, background).

    A music bed is roughly stationary over a second or two while speech is
    transient, so the per-bin median over a long window is a good estimate of
    the background. Whatever rises above it is voice. Classic HPSS was tried
    first and performed badly here: speech is harmonic *and* transient, so it
    landed mostly in the background stem.

    Demucs replaces this when installed (see providers/separation.py); this
    keeps stem splitting functional with zero downloads.
    """
    if x.size < n_fft * 2:
        return x.astype(np.float32), np.zeros_like(x, dtype=np.float32)

    spec = stft(x, n_fft, hop)
    mag = np.abs(spec)

    span = max(3, int(background_seconds * sample_rate / hop) | 1)  # odd window
    span = min(span, max(3, (mag.shape[0] // 2) * 2 - 1))
    background_mag = _median_filter(mag, span, axis=0)

    # what stands above the stationary bed is voice
    excess = np.maximum(mag - background_mag, 0.0)
    vocal_mask = excess / (excess + background_mag + 1e-9)

    # speech lives roughly between 120 Hz and 8 kHz; outside it, keep the bed
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    band = np.clip((freqs - 70.0) / 90.0, 0.0, 1.0) * np.clip((9000.0 - freqs) / 2000.0, 0.0, 1.0)
    vocal_mask = np.clip(vocal_mask * (0.35 + 0.65 * band[None, :]), 0.0, 1.0)

    # soften the mask in time so it does not chatter between frames
    kernel = np.ones(3, dtype=np.float32) / 3.0
    vocal_mask = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, vocal_mask)

    vocals = istft(spec * vocal_mask, n_fft, hop, length=x.size)
    background = istft(spec * (1.0 - vocal_mask), n_fft, hop, length=x.size)
    return vocals.astype(np.float32), background.astype(np.float32)


# --------------------------------------------------------------------------
# mix bus
# --------------------------------------------------------------------------
def envelope(x: np.ndarray, sample_rate: int, attack_ms: float = 5.0,
             release_ms: float = 120.0) -> np.ndarray:
    a = np.exp(-1.0 / max(1.0, sample_rate * attack_ms / 1000.0))
    r = np.exp(-1.0 / max(1.0, sample_rate * release_ms / 1000.0))
    rect = np.abs(x)
    out = np.empty_like(rect)
    prev = 0.0
    for i, v in enumerate(rect):
        coeff = a if v > prev else r
        prev = coeff * prev + (1.0 - coeff) * v
        out[i] = prev
    return out


def duck(background: np.ndarray, speech: np.ndarray, sample_rate: int,
         depth_db: float = -11.0) -> np.ndarray:
    """Side-chain the background under the dubbed speech."""
    n = max(background.size, speech.size)
    bg = np.pad(background, (0, n - background.size))
    sp = np.pad(speech, (0, n - speech.size))
    # cheap envelope via frame-wise RMS + smoothing, avoids a per-sample loop
    frame = max(64, sample_rate // 100)
    pad = (-sp.size) % frame
    rms = np.sqrt(np.mean(np.pad(sp, (0, pad)).reshape(-1, frame) ** 2, axis=1) + 1e-12)
    ctrl = np.repeat(rms, frame)[:n]
    ctrl = ctrl / max(float(ctrl.max()), 1e-6)
    kernel = np.hanning(max(3, sample_rate // 20))
    kernel /= kernel.sum()
    ctrl = np.convolve(ctrl, kernel, mode="same")
    floor = 10.0 ** (depth_db / 20.0)
    gain = 1.0 - (1.0 - floor) * np.clip(ctrl * 2.5, 0.0, 1.0)
    return (bg * gain).astype(np.float32)


def reverb(x: np.ndarray, sample_rate: int, amount: float = 0.12, room: float = 0.5) -> np.ndarray:
    """Schroeder reverb — restores a little of the room tone lost in TTS."""
    if amount <= 0.001:
        return x
    out = np.zeros(x.size + int(sample_rate * 0.5), dtype=np.float32)
    dry = np.pad(x, (0, out.size - x.size))
    wet = np.zeros_like(out)
    for delay_ms, gain in ((29.7, 0.78), (37.1, 0.74), (41.1, 0.70), (43.7, 0.68)):
        d = max(1, int(sample_rate * delay_ms / 1000.0 * (0.6 + room)))
        buf = np.zeros_like(out)
        buf[d:] = dry[:-d]
        fb = gain * (0.5 + 0.5 * room)
        acc = buf.copy()
        for _ in range(3):
            shifted = np.zeros_like(acc)
            shifted[d:] = acc[:-d]
            acc = buf + fb * shifted
        wet += acc / 4.0
    out = (1.0 - amount) * dry + amount * np.tanh(wet * 0.6)
    return out[:x.size].astype(np.float32)


def limiter(x: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= ceiling:
        return x.astype(np.float32)
    return (np.tanh(x / peak * 1.2) * ceiling).astype(np.float32)


def lufs_estimate(x: np.ndarray, sample_rate: int) -> float:
    """Rough integrated loudness (K-weighting approximated by a high-shelf)."""
    if x.size == 0:
        return -70.0
    pre = np.convolve(x, np.array([1.0, -0.97], dtype=np.float32), mode="same")
    ms = float(np.mean(pre ** 2) + 1e-12)
    return float(-0.691 + 10.0 * np.log10(ms))
