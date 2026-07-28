"""Text-to-speech + voice-conversion providers.

Local-first ordering:
  1. XTTS v2 / F5-TTS / Chatterbox / Fish / Piper if installed locally
  2. ElevenLabs / PlayHT / Cartesia if API keys are configured
  3. the built-in formant synthesiser, which always works

Every adapter receives the same `VoiceRef`, so a cloned voice created once is
renderable by whichever engine happens to be present.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from ..audio import synth
from ..audio.wavio import read_wav, resample, to_mono
from ..config import settings
from ..voice_design import VoiceParams
from .base import TTSProvider, VCProvider, VoiceRef, module_available, provider


# ==========================================================================
# local, always-available
# ==========================================================================
@provider("tts", "local_formant", rank=80)
class LocalFormantTTS:
    """Built-in formant synthesiser — offline, CPU-only, no downloads."""

    def synthesize(self, text: str, voice: VoiceRef, sample_rate: int,
                   emotion: str = "neutral", intensity: float = 1.0,
                   speed: float = 1.0) -> np.ndarray:
        params = VoiceParams(**voice.params.to_dict())
        if emotion and emotion != "neutral":
            params = params.with_emotion(emotion, intensity)
        params.speed *= max(0.3, speed)
        params.clamp()
        return synth.synthesize(text, params, sample_rate)


@provider("vc", "local_morph", rank=80)
class LocalVoiceConversion:
    """Offline speech-to-speech morphing (pitch + formant transfer)."""

    def convert(self, audio: np.ndarray, sample_rate: int, voice: VoiceRef,
                strength: float = 1.0) -> np.ndarray:
        return synth.convert_voice(audio, sample_rate, voice.params, strength=strength)


# ==========================================================================
# self-hosted open-source engines
# ==========================================================================
@provider("tts", "xtts", rank=10, requires="pip install TTS  (Coqui XTTS v2)")
class XTTSProvider:
    """Coqui XTTS v2 — zero-shot cross-lingual cloning from a short sample."""

    @staticmethod
    def is_available() -> bool:
        return module_available("TTS")

    def __init__(self) -> None:
        from TTS.api import TTS  # type: ignore

        device = "cuda" if _cuda() else "cpu"
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self._rate = 24000

    def synthesize(self, text: str, voice: VoiceRef, sample_rate: int,
                   emotion: str = "neutral", intensity: float = 1.0,
                   speed: float = 1.0) -> np.ndarray:
        ref = voice.reference_path
        kwargs = {"text": text, "language": (voice.language or "en")[:2], "speed": speed}
        if ref and Path(ref).exists():
            kwargs["speaker_wav"] = ref
        else:
            kwargs["speaker"] = self._tts.speakers[0] if getattr(self._tts, "speakers", None) else None
        wav = np.asarray(self._tts.tts(**{k: v for k, v in kwargs.items() if v is not None}),
                         dtype=np.float32)
        return resample(wav, self._rate, sample_rate)


@provider("tts", "f5", rank=12, requires="pip install f5-tts")
class F5TTSProvider:
    """F5-TTS — non-autoregressive cloning from 6-10s of reference audio."""

    @staticmethod
    def is_available() -> bool:
        return module_available("f5_tts")

    def __init__(self) -> None:
        from f5_tts.api import F5TTS  # type: ignore

        self._model = F5TTS()

    def synthesize(self, text: str, voice: VoiceRef, sample_rate: int,
                   emotion: str = "neutral", intensity: float = 1.0,
                   speed: float = 1.0) -> np.ndarray:
        wav, sr, _ = self._model.infer(
            ref_file=voice.reference_path, ref_text="", gen_text=text, speed=speed, remove_silence=True,
        )
        return resample(np.asarray(wav, dtype=np.float32), int(sr), sample_rate)


@provider("tts", "chatterbox", rank=14, requires="pip install chatterbox-tts")
class ChatterboxProvider:
    """Resemble Chatterbox — MIT-licensed, low-VRAM, expressive control."""

    @staticmethod
    def is_available() -> bool:
        return module_available("chatterbox")

    def __init__(self) -> None:
        from chatterbox.tts import ChatterboxTTS  # type: ignore

        self._model = ChatterboxTTS.from_pretrained(device="cuda" if _cuda() else "cpu")

    def synthesize(self, text: str, voice: VoiceRef, sample_rate: int,
                   emotion: str = "neutral", intensity: float = 1.0,
                   speed: float = 1.0) -> np.ndarray:
        exaggeration = float(np.clip(0.5 + 0.4 * intensity, 0.25, 1.5)) if emotion != "neutral" else 0.5
        wav = self._model.generate(text, audio_prompt_path=voice.reference_path,
                                   exaggeration=exaggeration)
        arr = np.asarray(getattr(wav, "cpu", lambda: wav)(), dtype=np.float32).squeeze()
        return resample(arr, int(getattr(self._model, "sr", 24000)), sample_rate)


@provider("tts", "piper", rank=16, requires="pip install piper-tts + a .onnx voice")
class PiperProvider:
    """Piper — fast CPU neural TTS, preset voices only (no cloning)."""

    @staticmethod
    def is_available() -> bool:
        return module_available("piper") and bool(settings.piper_voice_dir) \
            and Path(settings.piper_voice_dir).is_dir()

    def __init__(self) -> None:
        from piper.voice import PiperVoice  # type: ignore

        models = sorted(Path(settings.piper_voice_dir).glob("*.onnx"))
        if not models:
            raise RuntimeError("no .onnx voices in DUB_PIPER_VOICE_DIR")
        self._voice = PiperVoice.load(str(models[0]))
        self._rate = self._voice.config.sample_rate

    def synthesize(self, text: str, voice: VoiceRef, sample_rate: int,
                   emotion: str = "neutral", intensity: float = 1.0,
                   speed: float = 1.0) -> np.ndarray:
        chunks = [np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
                  for b in self._voice.synthesize_stream_raw(text, length_scale=1.0 / max(0.3, speed))]
        wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        return resample(wav, self._rate, sample_rate)


@provider("vc", "rvc", rank=10, requires="pip install rvc-python + a trained .pth")
class RVCProvider:
    """RVC — retrieval-based voice-to-voice conversion (speech and singing)."""

    @staticmethod
    def is_available() -> bool:
        return module_available("rvc_python") and bool(os.environ.get("DUB_RVC_MODEL"))

    def __init__(self) -> None:
        from rvc_python.infer import RVCInference  # type: ignore

        self._rvc = RVCInference(model_path=os.environ["DUB_RVC_MODEL"],
                                 device="cuda:0" if _cuda() else "cpu")

    def convert(self, audio: np.ndarray, sample_rate: int, voice: VoiceRef,
                strength: float = 1.0) -> np.ndarray:
        import tempfile

        from ..audio.wavio import write_wav

        with tempfile.TemporaryDirectory() as tmp:
            src = write_wav(Path(tmp) / "in.wav", audio, sample_rate)
            dst = Path(tmp) / "out.wav"
            self._rvc.infer_file(str(src), str(dst))
            wav, sr = read_wav(dst)
        return resample(to_mono(wav), sr, sample_rate)


# ==========================================================================
# commercial API aggregation
# ==========================================================================
@provider("tts", "elevenlabs", rank=20, requires="ELEVENLABS_API_KEY")
class ElevenLabsProvider:
    """ElevenLabs — highest-fidelity commercial cloning and Voice Design."""

    BASE = "https://api.elevenlabs.io/v1"

    @staticmethod
    def is_available() -> bool:
        return bool(settings.elevenlabs_api_key)

    def synthesize(self, text: str, voice: VoiceRef, sample_rate: int,
                   emotion: str = "neutral", intensity: float = 1.0,
                   speed: float = 1.0) -> np.ndarray:
        voice_id = voice.provider_voice_id or "21m00Tcm4TlvDq8ikWAM"
        body = json.dumps({
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": float(np.clip(0.6 - 0.25 * intensity, 0.05, 1.0)),
                "similarity_boost": 0.85,
                "style": float(np.clip(0.35 * intensity, 0.0, 1.0)),
                "speed": float(np.clip(speed, 0.7, 1.2)),
            },
        }).encode()
        req = urllib.request.Request(
            f"{self.BASE}/text-to-speech/{voice_id}?output_format=pcm_24000",
            data=body, method="POST",
            headers={"xi-api-key": settings.elevenlabs_api_key, "Content-Type": "application/json",
                     "Accept": "audio/pcm"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
        wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return resample(wav, 24000, sample_rate)

    def create_voice_from_sample(self, name: str, sample_paths: list[str]) -> str | None:
        """Instant (zero-shot) clone → returns the provider voice id."""
        from .asr import _multipart  # reuse the multipart encoder

        if not sample_paths:
            return None
        body, ctype = _multipart({"name": name}, "files", Path(sample_paths[0]).name,
                                 Path(sample_paths[0]).read_bytes())
        req = urllib.request.Request(f"{self.BASE}/voices/add", data=body, method="POST",
                                     headers={"xi-api-key": settings.elevenlabs_api_key,
                                              "Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode()).get("voice_id")
        except Exception:
            return None


@provider("tts", "playht", rank=22, requires="PLAYHT_API_KEY + PLAYHT_USER_ID")
class PlayHTProvider:
    """PlayHT Play 3.0 — low-latency streaming, custom fine-tuned clones."""

    @staticmethod
    def is_available() -> bool:
        return bool(os.environ.get("PLAYHT_API_KEY") and os.environ.get("PLAYHT_USER_ID"))

    def synthesize(self, text: str, voice: VoiceRef, sample_rate: int,
                   emotion: str = "neutral", intensity: float = 1.0,
                   speed: float = 1.0) -> np.ndarray:
        body = json.dumps({
            "text": text,
            "voice": voice.provider_voice_id or "s3://voice-cloning-zero-shot/default",
            "output_format": "wav",
            "sample_rate": sample_rate,
            "speed": speed,
            **({"emotion": emotion} if emotion != "neutral" else {}),
        }).encode()
        req = urllib.request.Request(
            "https://api.play.ht/api/v2/tts/stream", data=body, method="POST",
            headers={"Authorization": os.environ["PLAYHT_API_KEY"],
                     "X-User-ID": os.environ["PLAYHT_USER_ID"],
                     "Content-Type": "application/json", "Accept": "audio/wav"},
        )
        return _wav_bytes_to_array(urllib.request.urlopen(req, timeout=180).read(), sample_rate)


@provider("tts", "cartesia", rank=24, requires="CARTESIA_API_KEY")
class CartesiaProvider:
    """Cartesia Sonic — sub-100ms TTFB, built for real-time agents."""

    @staticmethod
    def is_available() -> bool:
        return bool(os.environ.get("CARTESIA_API_KEY"))

    def synthesize(self, text: str, voice: VoiceRef, sample_rate: int,
                   emotion: str = "neutral", intensity: float = 1.0,
                   speed: float = 1.0) -> np.ndarray:
        body = json.dumps({
            "model_id": "sonic-2",
            "transcript": text,
            "voice": {"mode": "id", "id": voice.provider_voice_id or ""},
            "language": (voice.language or "en")[:2],
            "output_format": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": sample_rate},
        }).encode()
        req = urllib.request.Request(
            "https://api.cartesia.ai/tts/bytes", data=body, method="POST",
            headers={"X-API-Key": os.environ["CARTESIA_API_KEY"], "Cartesia-Version": "2024-06-10",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return np.frombuffer(resp.read(), dtype="<f4").astype(np.float32)


# ==========================================================================
# helpers
# ==========================================================================
def _cuda() -> bool:
    if not module_available("torch"):
        return False
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _wav_bytes_to_array(raw: bytes, sample_rate: int) -> np.ndarray:
    import io
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fh:
        fh.write(raw)
        fh.flush()
        wav, sr = read_wav(fh.name)
    return resample(to_mono(wav), sr, sample_rate)
