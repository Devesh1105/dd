"""Offline formant synthesiser.

This is the fallback TTS engine that lets the whole platform run on a laptop
with no models downloaded and no API keys. It is a vectorised Klatt-style
source/filter synthesiser:

    text -> phones -> per-frame (F1,F2,F3, voicing, noise, amplitude)
    excitation (glottal pulses + jitter + noise) -> STFT -> vocal-tract
    magnitude response -> ISTFT -> waveform

It is intelligible and, crucially, *voice-conditioned*: pitch, vocal-tract
length, rasp, breathiness and pacing all come from `VoiceParams`, which are
derived either from a designed prompt or from a cloned speaker embedding.
Register a neural provider (XTTS/Piper/F5/ElevenLabs) for production audio.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from ..voice_design import VoiceParams

# --------------------------------------------------------------------------
# phone inventory
# --------------------------------------------------------------------------
VOWEL, NASAL, LIQUID, FRIC, STOP, SILENCE = "V", "N", "L", "F", "S", "_"


@dataclass
class Phone:
    kind: str
    f1: float = 500.0
    f2: float = 1500.0
    f3: float = 2500.0
    duration: float = 0.08          # seconds at speed 1.0
    voiced: bool = True
    amp: float = 1.0
    noise: float = 0.0              # aperiodic gain
    noise_center: float = 4000.0
    noise_width: float = 2500.0
    glide_to: tuple[float, float, float] | None = None  # diphthongs


def _v(f1, f2, f3, dur=0.095, glide=None) -> Phone:
    return Phone(VOWEL, f1, f2, f3, dur, True, 1.0, 0.0, glide_to=glide)


VOWELS: dict[str, Phone] = {
    "aa": _v(730, 1090, 2440),
    "ae": _v(660, 1720, 2410),
    "ah": _v(640, 1190, 2390, 0.085),
    "ax": _v(500, 1500, 2500, 0.060),          # schwa
    "eh": _v(530, 1840, 2480),
    "ih": _v(390, 1990, 2550, 0.075),
    "iy": _v(300, 2290, 3010, 0.100),
    "ao": _v(570, 840, 2410),
    "ow": _v(500, 900, 2400, 0.110, glide=(400, 800, 2300)),
    "uh": _v(440, 1020, 2240, 0.075),
    "uw": _v(320, 870, 2250, 0.100),
    "ey": _v(530, 1840, 2480, 0.115, glide=(340, 2200, 2900)),
    "ay": _v(730, 1090, 2440, 0.125, glide=(340, 2200, 2900)),
    "oy": _v(570, 840, 2410, 0.125, glide=(340, 2200, 2900)),
    "aw": _v(730, 1090, 2440, 0.125, glide=(360, 900, 2300)),
    "er": _v(490, 1350, 1690, 0.100),
}

CONSONANTS: dict[str, Phone] = {
    "m": Phone(NASAL, 280, 900, 2200, 0.065, True, 0.62),
    "n": Phone(NASAL, 280, 1700, 2600, 0.060, True, 0.62),
    "ng": Phone(NASAL, 280, 2300, 2750, 0.065, True, 0.58),
    "l": Phone(LIQUID, 360, 1300, 2500, 0.060, True, 0.80),
    "r": Phone(LIQUID, 320, 1200, 1600, 0.060, True, 0.80),
    "w": Phone(LIQUID, 300, 610, 2200, 0.055, True, 0.75),
    "y": Phone(LIQUID, 300, 2200, 3000, 0.050, True, 0.75),
    "f": Phone(FRIC, 400, 1200, 2400, 0.075, False, 0.30, 0.85, 5200, 3000),
    "v": Phone(FRIC, 400, 1200, 2400, 0.060, True, 0.42, 0.45, 4800, 3000),
    "th": Phone(FRIC, 400, 1400, 2600, 0.070, False, 0.26, 0.80, 5800, 3200),
    "dh": Phone(FRIC, 400, 1400, 2600, 0.055, True, 0.40, 0.40, 5000, 3000),
    "s": Phone(FRIC, 400, 1400, 2600, 0.085, False, 0.34, 1.00, 6400, 2600),
    "z": Phone(FRIC, 400, 1400, 2600, 0.070, True, 0.44, 0.55, 6000, 2600),
    "sh": Phone(FRIC, 400, 1800, 2600, 0.090, False, 0.38, 1.00, 3200, 1900),
    "zh": Phone(FRIC, 400, 1800, 2600, 0.070, True, 0.44, 0.55, 3200, 1900),
    "h": Phone(FRIC, 500, 1500, 2500, 0.055, False, 0.24, 0.70, 1800, 3500),
    "p": Phone(STOP, 400, 1100, 2200, 0.070, False, 0.34, 0.75, 1200, 1400),
    "b": Phone(STOP, 350, 1100, 2200, 0.058, True, 0.44, 0.30, 1200, 1200),
    "t": Phone(STOP, 400, 1700, 2600, 0.068, False, 0.36, 0.85, 3900, 2200),
    "d": Phone(STOP, 350, 1700, 2600, 0.056, True, 0.46, 0.35, 3500, 2000),
    "k": Phone(STOP, 400, 1900, 2400, 0.072, False, 0.34, 0.80, 2400, 2000),
    "g": Phone(STOP, 350, 1900, 2400, 0.058, True, 0.44, 0.32, 2200, 1800),
    "ch": Phone(FRIC, 400, 1800, 2600, 0.095, False, 0.38, 0.95, 3400, 2000),
    "jh": Phone(FRIC, 400, 1800, 2600, 0.080, True, 0.46, 0.55, 3200, 2000),
}

PAUSE_SHORT = Phone(SILENCE, 500, 1500, 2500, 0.055, False, 0.0, 0.0)
PAUSE_LONG = Phone(SILENCE, 500, 1500, 2500, 0.180, False, 0.0, 0.0)

# --------------------------------------------------------------------------
# grapheme -> phone
# --------------------------------------------------------------------------
_DIGRAPHS = [
    ("tch", ["ch"]), ("sch", ["sh"]), ("igh", ["ay"]),
    ("ch", ["ch"]), ("sh", ["sh"]), ("th", ["th"]), ("ph", ["f"]), ("wh", ["w"]),
    ("ck", ["k"]), ("ng", ["ng"]), ("qu", ["k", "w"]), ("gh", ["g"]),
    ("ee", ["iy"]), ("ea", ["iy"]), ("oo", ["uw"]), ("ou", ["aw"]), ("ow", ["ow"]),
    ("ai", ["ey"]), ("ay", ["ey"]), ("oi", ["oy"]), ("oy", ["oy"]), ("au", ["ao"]),
    ("aw", ["ao"]), ("ie", ["iy"]), ("ei", ["ey"]), ("ue", ["uw"]), ("ui", ["uw"]),
    ("er", ["er"]), ("ir", ["er"]), ("ur", ["er"]), ("ar", ["aa", "r"]), ("or", ["ao", "r"]),
]

_SINGLE = {
    "a": ["ae"], "b": ["b"], "c": ["k"], "d": ["d"], "e": ["eh"], "f": ["f"],
    "g": ["g"], "h": ["h"], "i": ["ih"], "j": ["jh"], "k": ["k"], "l": ["l"],
    "m": ["m"], "n": ["n"], "o": ["ao"], "p": ["p"], "q": ["k"], "r": ["r"],
    "s": ["s"], "t": ["t"], "u": ["ah"], "v": ["v"], "w": ["w"], "x": ["k", "s"],
    "y": ["ih"], "z": ["z"],
}

# fallback syllable nuclei for non-latin scripts (CJK, Devanagari, Cyrillic, …)
_NON_LATIN_NUCLEI = ["aa", "iy", "uw", "eh", "ow", "ah"]

_NUMBERS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return "".join(_NUMBERS.get(c, c) + (" " if c in _NUMBERS else "") for c in text)


def word_to_phones(word: str) -> list[str]:
    """Rule-based grapheme-to-phone. Latin script uses digraph rules; other
    scripts fall back to a deterministic CV syllable sketch so that any
    language still produces speech-shaped, correctly-timed audio."""
    w = word.lower()
    if not w:
        return []
    if not re.search(r"[a-z]", w):
        out: list[str] = []
        for i, ch in enumerate(w):
            if not ch.strip():
                continue
            code = ord(ch)
            out.append(["k", "t", "s", "m", "n", "l", "r", "d"][code % 8])
            out.append(_NON_LATIN_NUCLEI[(code >> 3) % len(_NON_LATIN_NUCLEI)])
        return out or ["ax"]

    phones: list[str] = []
    i = 0
    while i < len(w):
        matched = False
        for gr, ph in _DIGRAPHS:
            if w.startswith(gr, i):
                phones.extend(ph)
                i += len(gr)
                matched = True
                break
        if matched:
            continue
        ch = w[i]
        if ch in _SINGLE:
            # silent trailing 'e' after a consonant lengthens the previous vowel
            if ch == "e" and i == len(w) - 1 and len(w) > 2 and w[i - 1] not in "aeiou":
                for j in range(len(phones) - 1, -1, -1):
                    if phones[j] in VOWELS:
                        phones[j] = {"ae": "ey", "eh": "iy", "ih": "ay", "ao": "ow", "ah": "uw"}.get(phones[j], phones[j])
                        break
            else:
                phones.extend(_SINGLE[ch])
        i += 1
    return phones or ["ax"]


def text_to_phones(text: str, speed: float = 1.0) -> list[Phone]:
    """Full utterance → timed phone sequence, with prosodic stress + pauses."""
    text = _normalise(text)
    if not text:
        return []
    seq: list[Phone] = []
    tokens = re.findall(r"[^\s]+", text)
    for t_i, token in enumerate(tokens):
        core = re.sub(r"^[^\w]+|[^\w]+$", "", token)
        trailing = token[len(token.rstrip(".,;:!?…—-")):] if token != core else ""
        names = word_to_phones(core)
        stressed = len(core) > 3 and t_i % 2 == 0
        vowel_seen = False
        for name in names:
            base = VOWELS.get(name) or CONSONANTS.get(name)
            if base is None:
                continue
            p = Phone(**vars(base))
            if p.kind == VOWEL:
                if not vowel_seen and stressed:
                    p.duration *= 1.22
                    p.amp *= 1.15
                    vowel_seen = True
                elif vowel_seen:
                    p.duration *= 0.88
                    p.amp *= 0.88
            seq.append(p)
        if any(c in trailing for c in ".!?"):
            seq.append(Phone(**vars(PAUSE_LONG)))
        elif any(c in trailing for c in ",;:—…"):
            seq.append(Phone(**vars(PAUSE_SHORT)))
        elif t_i < len(tokens) - 1:
            seq.append(Phone(SILENCE, 500, 1500, 2500, 0.030, False, 0.0, 0.0))

    inv = 1.0 / max(0.3, speed)
    for p in seq:
        p.duration *= inv
    return seq


def estimate_duration(text: str, params: VoiceParams | None = None) -> float:
    """Predicted spoken length in seconds — used by the translation
    length-expansion check before any audio is rendered."""
    params = params or VoiceParams()
    return float(sum(p.duration for p in text_to_phones(text, params.speed)) + 0.08)


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------
def _frame_targets(phones: list[Phone], sample_rate: int, hop: int, params: VoiceParams
                   ) -> tuple[np.ndarray, int]:
    """Expand the phone list into per-frame parameter rows."""
    total = sum(p.duration for p in phones)
    n_samples = max(hop * 4, int(total * sample_rate))
    n_frames = max(2, n_samples // hop + 1)
    # columns: f1 f2 f3 voiced noise amp noise_center noise_width
    rows = np.zeros((n_frames, 8), dtype=np.float32)

    t = 0.0
    idx = 0
    for p in phones:
        end = t + p.duration
        start_f = int(t * sample_rate / hop)
        end_f = max(start_f + 1, int(end * sample_rate / hop))
        span = end_f - start_f
        if start_f >= n_frames:
            break
        end_f = min(end_f, n_frames)
        ramp = np.linspace(0.0, 1.0, max(1, end_f - start_f), dtype=np.float32)
        tgt = p.glide_to or (p.f1, p.f2, p.f3)
        rows[start_f:end_f, 0] = p.f1 + (tgt[0] - p.f1) * ramp
        rows[start_f:end_f, 1] = p.f2 + (tgt[1] - p.f2) * ramp
        rows[start_f:end_f, 2] = p.f3 + (tgt[2] - p.f3) * ramp
        rows[start_f:end_f, 3] = 1.0 if p.voiced else 0.0
        rows[start_f:end_f, 4] = p.noise
        # short attack/decay so stops and plosives keep their transient
        env = np.ones(end_f - start_f, dtype=np.float32)
        edge = max(1, int(0.2 * span))
        if env.size > 2 * edge:
            env[:edge] = np.linspace(0.35, 1.0, edge) ** (1.0 / max(0.4, params.attack))
            env[-edge:] = np.linspace(1.0, 0.55, edge)
        rows[start_f:end_f, 5] = p.amp * env
        rows[start_f:end_f, 6] = p.noise_center
        rows[start_f:end_f, 7] = p.noise_width
        t = end
        idx += 1

    rows[:, 6] = np.where(rows[:, 6] <= 0, 4000.0, rows[:, 6])
    rows[:, 7] = np.where(rows[:, 7] <= 0, 2500.0, rows[:, 7])
    # smooth formant trajectories — coarticulation
    k = np.hanning(5).astype(np.float32)
    k /= k.sum()
    for c in (0, 1, 2, 5):
        rows[:, c] = np.convolve(rows[:, c], k, mode="same")
    return rows, n_samples


def _pitch_contour(n_frames: int, params: VoiceParams, hop: int, sample_rate: int,
                   question: bool, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    # declination + phrase accent
    contour = 1.0 - 0.16 * t
    contour += params.f0_range * 0.55 * np.sin(2 * np.pi * (1.4 * t + 0.12))
    contour += params.f0_range * 0.25 * np.sin(2 * np.pi * (3.7 * t + 0.5))
    if question:
        contour += 0.28 * np.clip((t - 0.72) / 0.28, 0.0, 1.0)
    if params.vibrato > 0.001:
        rate = 5.2 * hop / sample_rate
        contour += params.vibrato * 0.2 * np.sin(2 * np.pi * rate * np.arange(n_frames))
    if params.tremor > 0.001:
        walk = np.cumsum(rng.normal(0.0, params.tremor * 0.05, n_frames))
        contour += np.clip(walk, -0.35, 0.35)
    return (params.f0 * np.clip(contour, 0.45, 2.2)).astype(np.float32)


def _excitation(rows: np.ndarray, n_samples: int, sample_rate: int, hop: int,
                params: VoiceParams, question: bool, rng: np.random.Generator) -> np.ndarray:
    n_frames = rows.shape[0]
    f0_frames = _pitch_contour(n_frames, params, hop, sample_rate, question, rng)
    idx = np.clip(np.arange(n_samples) // hop, 0, n_frames - 1)
    f0 = f0_frames[idx]
    voiced = rows[idx, 3]

    # jitter: cycle-to-cycle pitch perturbation is what reads as "rasp"
    jitter = 1.0 + params.rasp * 0.35 * rng.normal(0.0, 1.0, n_samples)
    jitter = np.convolve(jitter, np.ones(24, dtype=np.float32) / 24.0, mode="same")
    phase = np.cumsum(f0 * jitter / sample_rate)
    frac = np.mod(phase, 1.0)

    # Rosenberg-like glottal pulse: smooth open phase, sharp closure
    open_q = 0.62
    pulse = np.where(
        frac < open_q,
        0.5 * (1.0 - np.cos(np.pi * frac / open_q)),
        np.cos(np.pi * 0.5 * (frac - open_q) / max(1e-6, 1.0 - open_q)),
    ).astype(np.float32)
    pulse = pulse - pulse.mean()

    if params.growl > 0.01:
        sub = np.sin(2 * np.pi * (phase * 0.5)).astype(np.float32)
        pulse = (1.0 - 0.45 * params.growl) * pulse + 0.45 * params.growl * sub
        pulse += params.growl * 0.25 * np.sign(pulse) * np.abs(pulse) ** 2

    noise = rng.normal(0.0, 1.0, n_samples).astype(np.float32)
    aperiodic = rows[idx, 4]
    src = voiced * (pulse + params.breathiness * 0.55 * noise) + aperiodic * noise * 0.9
    amp = rows[idx, 5]
    if params.tremor > 0.001:
        wobble = 1.0 + params.tremor * 0.25 * np.sin(2 * np.pi * 6.5 * np.arange(n_samples) / sample_rate)
        amp = amp * wobble
    return (src * amp).astype(np.float32)


def _vocal_tract(rows: np.ndarray, freqs: np.ndarray, params: VoiceParams) -> np.ndarray:
    """Per-frame magnitude response: three formant resonances + tilt + noise band."""
    shift = params.formant_shift
    f = freqs[None, :]
    resp = np.zeros((rows.shape[0], freqs.size), dtype=np.float32)
    for col, (amp, bw) in enumerate(((1.0, 78.0), (0.72, 105.0), (0.42, 150.0))):
        center = rows[:, col][:, None] * shift
        half = bw * (1.0 + 0.35 * params.rasp)
        resp += amp * (half ** 2) / ((f - center) ** 2 + half ** 2)

    # fricative/burst energy sits in its own band, not in the formants
    nc = rows[:, 6][:, None] * (0.85 + 0.15 * shift)
    nw = rows[:, 7][:, None]
    band = np.exp(-0.5 * ((f - nc) / np.maximum(nw, 1.0)) ** 2)
    resp += rows[:, 4][:, None] * band * 1.35

    tilt = (1000.0 / np.maximum(freqs, 90.0)) ** (0.55 / max(0.3, params.brightness))
    resp *= tilt[None, :]
    resp *= np.clip((freqs - 55.0) / 90.0, 0.0, 1.0)[None, :]      # high-pass rumble
    resp *= np.clip((freqs.max() - freqs) / 2500.0, 0.0, 1.0)[None, :]  # anti-alias roll-off
    return resp.astype(np.float32)


def synthesize(text: str, params: VoiceParams, sample_rate: int = 24000,
               seed: int | None = None) -> np.ndarray:
    """Render `text` in the voice described by `params`."""
    params = VoiceParams(**params.to_dict()).clamp()
    phones = text_to_phones(text, params.speed)
    if not phones:
        return np.zeros(int(0.05 * sample_rate), dtype=np.float32)

    n_fft, hop = 512, 128
    if seed is None:  # stable across processes, unlike hash()
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    rows, n_samples = _frame_targets(phones, sample_rate, hop, params)
    src = _excitation(rows, n_samples, sample_rate, hop, params, text.strip().endswith("?"), rng)

    window = np.hanning(n_fft).astype(np.float32)
    if src.size < n_fft:
        src = np.pad(src, (0, n_fft - src.size))
    n_frames = 1 + (src.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = src[idx] * window
    spec = np.fft.rfft(frames, axis=1)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    frame_rows = rows[np.clip(np.arange(n_frames), 0, rows.shape[0] - 1)]
    spec *= _vocal_tract(frame_rows, freqs, params)

    out_frames = np.fft.irfft(spec, n=n_fft, axis=1) * window
    out = np.zeros(src.size, dtype=np.float32)
    norm = np.zeros(src.size, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += out_frames[i]
        norm[s:s + n_fft] += window ** 2
    # only trust samples with full window coverage: the ramp-in/ramp-out
    # regions have near-zero norm and would explode into edge clicks
    valid = norm > 0.05 * float(norm.max() or 1.0)
    out = np.where(valid, out / np.maximum(norm, 1e-6), 0.0).astype(np.float32)

    out = np.tanh(out * (1.4 * params.energy))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-6:
        out *= min(0.92, 0.62 * params.energy) / peak
    fade = min(int(0.008 * sample_rate), out.size // 2)
    if fade > 1:
        out[:fade] *= np.linspace(0.0, 1.0, fade)
        out[-fade:] *= np.linspace(1.0, 0.0, fade)
    return out.astype(np.float32)


def convert_voice(source: np.ndarray, sample_rate: int, target: VoiceParams,
                  source_f0: float | None = None, strength: float = 1.0) -> np.ndarray:
    """Voice-to-voice (RVC-style) conversion, offline.

    Keeps the source timing, phrasing and energy contour, and re-voices it:
    pitch is shifted toward the target's f0 and the spectral envelope is
    warped by the target's vocal-tract length. Swap in RVC / Chatterbox via
    the provider registry for production quality.
    """
    from . import dsp  # local import avoids a cycle at module load

    x = np.asarray(source, dtype=np.float32)
    if x.size == 0:
        return x
    src_f0 = source_f0 if source_f0 else dsp.estimate_f0(x, sample_rate)
    ratio = 1.0
    if src_f0 > 40.0:
        ratio = float(np.clip(target.f0 / src_f0, 0.5, 2.0))
    ratio = 1.0 + (ratio - 1.0) * float(np.clip(strength, 0.0, 1.0))

    # pitch shift = resample (changes pitch + formants) then restore duration
    if abs(ratio - 1.0) > 1e-3:
        shifted = np.interp(
            np.arange(0, x.size, ratio, dtype=np.float64),
            np.arange(x.size, dtype=np.float64), x,
        ).astype(np.float32)
        y = dsp.time_stretch(shifted, shifted.size / max(1, x.size), sample_rate)
    else:
        y = x.copy()
    if y.size < x.size:
        y = np.pad(y, (0, x.size - y.size))
    y = y[:x.size]

    # formant warp: stretch the magnitude spectrum along the frequency axis
    warp = float(np.clip(target.formant_shift / ratio, 0.7, 1.45))
    if abs(warp - 1.0) > 0.01:
        n_fft, hop = 1024, 256
        spec = dsp.stft(y, n_fft, hop)
        mag, phase = np.abs(spec), np.angle(spec)
        bins = np.arange(mag.shape[1])
        warped = np.stack([np.interp(bins, bins * warp, row, left=0.0, right=0.0) for row in mag])
        y = dsp.istft(warped * np.exp(1j * phase), n_fft, hop, length=x.size)

    if target.rasp > 0.05:
        rng = np.random.default_rng(1234)
        env = np.abs(y) + 1e-4
        y = y + target.rasp * 0.25 * env * rng.normal(0.0, 1.0, y.size).astype(np.float32)
    return dsp.limiter(y.astype(np.float32))
