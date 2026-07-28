#!/usr/bin/env python3
"""Generate a demo clip: two speakers over background music.

Gives you something to dub immediately without hunting for media:

    python scripts/make_sample.py data/sample.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.audio import synth  # noqa: E402
from backend.app.audio.wavio import write_wav  # noqa: E402
from backend.app.voice_design import PRESETS_BY_ID  # noqa: E402

SR = 24000

DIALOGUE = [
    ("studio_anchor_m", "Good evening. Tonight we look at how machine dubbing changes film."),
    ("anime_genki_girl", "Oh, I love this topic! Can a computer really copy someone's voice?"),
    ("studio_anchor_m", "It can. A short recording is enough to capture a speaker's identity."),
    ("anime_genki_girl", "That is amazing. Does it work in other languages too?"),
    ("studio_anchor_m", "Yes. The same voice can speak a language the person never learned."),
]


def backing_track(duration: float) -> np.ndarray:
    """Simple pad + pulse so stem separation has something to pull apart."""
    t = np.arange(int(duration * SR), dtype=np.float32) / SR
    chords = [(146.83, 185.00, 220.00), (130.81, 164.81, 196.00),
              (174.61, 220.00, 261.63), (164.81, 207.65, 246.94)]
    music = np.zeros_like(t)
    bar = 3.2
    for i, chord in enumerate(chords * (int(duration / (bar * len(chords))) + 1)):
        start, end = i * bar, min(duration, (i + 1) * bar)
        if start >= duration:
            break
        seg = (t >= start) & (t < end)
        local = t[seg] - start
        env = np.minimum(1.0, local / 0.4) * np.exp(-0.18 * local)
        for f in chord:
            music[seg] += 0.10 * env * np.sin(2 * np.pi * f * local)
            music[seg] += 0.03 * env * np.sin(2 * np.pi * f * 2 * local)
    beat = np.zeros_like(t)
    for k in range(int(duration * 2)):
        i = int(k * 0.5 * SR)
        n = min(int(0.09 * SR), beat.size - i)
        if n > 0:
            beat[i:i + n] += 0.16 * np.exp(-28 * np.arange(n) / SR) * \
                np.random.default_rng(k).normal(0, 1, n)
    return (music + beat).astype(np.float32)


def main(out_path: str = "data/sample.wav") -> None:
    clips = []
    for voice_id, line in DIALOGUE:
        wav = synth.synthesize(line, PRESETS_BY_ID[voice_id].params, SR)
        clips.append(np.concatenate([wav, np.zeros(int(0.45 * SR), dtype=np.float32)]))
    speech = np.concatenate(clips)
    duration = speech.size / SR
    mixed = np.clip(speech * 0.95 + backing_track(duration) * 0.42, -1.0, 1.0)

    path = write_wav(out_path, mixed, SR)
    print(f"wrote {path}  ({duration:.1f}s, {len(DIALOGUE)} lines, 2 speakers)")
    print("Transcript (paste into the editor or POST /api/projects/{id}/script):")
    for _, line in DIALOGUE:
        print(f"  {line}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample.wav")
