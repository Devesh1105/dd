"""Background/vocal stem separation and lip-sync providers."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..audio import dsp
from ..audio.wavio import read_wav, resample, to_mono, write_wav
from .base import LipsyncProvider, SeparationProvider, module_available, provider


@provider("separation", "demucs", rank=10, requires="pip install demucs")
class DemucsSeparation:
    """Demucs v4 — state-of-the-art music source separation."""

    @staticmethod
    def is_available() -> bool:
        return module_available("demucs")

    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        import torch  # type: ignore
        from demucs.apply import apply_model  # type: ignore
        from demucs.pretrained import get_model  # type: ignore

        model = get_model("htdemucs")
        model.eval()
        target_sr = model.samplerate
        mono = resample(to_mono(audio), sample_rate, target_sr)
        wav = torch.from_numpy(np.stack([mono, mono]))[None]
        with torch.no_grad():
            sources = apply_model(model, wav, device="cuda" if torch.cuda.is_available() else "cpu")[0]
        names = model.sources
        vocals = sources[names.index("vocals")].mean(0).cpu().numpy()
        background = sum(sources[i].mean(0).cpu().numpy() for i, n in enumerate(names) if n != "vocals")
        return (resample(vocals.astype(np.float32), target_sr, sample_rate),
                resample(np.asarray(background, dtype=np.float32), target_sr, sample_rate))


@provider("separation", "spectral", rank=90)
class SpectralSeparation:
    """Offline HPSS separation — no models, runs anywhere."""

    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        return dsp.separate_stems(to_mono(audio), sample_rate)


@provider("lipsync", "wav2lip", rank=10, requires="Wav2Lip checkpoint + DUB_WAV2LIP_DIR")
class Wav2LipSync:
    """Wav2Lip — re-renders mouth movements to match the dubbed audio."""

    @staticmethod
    def is_available() -> bool:
        import os

        root = os.environ.get("DUB_WAV2LIP_DIR", "")
        return bool(root) and (Path(root) / "inference.py").exists() and shutil.which("ffmpeg") is not None

    def sync(self, video_path: str, audio_path: str, out_path: str) -> str | None:
        import os

        root = Path(os.environ["DUB_WAV2LIP_DIR"])
        ckpt = os.environ.get("DUB_WAV2LIP_CKPT", str(root / "checkpoints" / "wav2lip_gan.pth"))
        cmd = ["python", str(root / "inference.py"), "--checkpoint_path", ckpt,
               "--face", video_path, "--audio", audio_path, "--outfile", out_path]
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=3600)
        return out_path if proc.returncode == 0 and Path(out_path).exists() else None


@provider("lipsync", "none", rank=90)
class NoLipsync:
    """No lip-sync — audio is muxed onto the original video unchanged."""

    def sync(self, video_path: str, audio_path: str, out_path: str) -> str | None:
        return None
