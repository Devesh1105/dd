"""Prompt-driven voice design.

Turns a natural-language description ("a low-pitched, stoic male anime rival,
husky and gravelly, slow and confident") into a `VoiceParams` vector that the
local synthesiser renders directly, and that remote providers
(ElevenLabs Voice Design, PlayHT, Cartesia) receive as a prompt string.

The parser follows the five-part prompt structure the platform documents:
  1. age & gender   2. pitch & texture   3. pacing & rhythm
  4. emotion & attitude   5. accent / style
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class VoiceParams:
    """Synthesiser-facing description of a voice."""

    f0: float = 120.0            # base pitch in Hz
    f0_range: float = 0.18       # relative pitch excursion (expressiveness)
    formant_shift: float = 1.0   # vocal-tract length scaling (<1 = larger head)
    rasp: float = 0.08           # aperiodic noise mixed into the glottal source
    breathiness: float = 0.10    # broadband air in voiced frames
    speed: float = 1.0           # words-per-minute multiplier
    energy: float = 1.0          # output level / projection
    vibrato: float = 0.0         # pitch modulation depth
    tremor: float = 0.0          # amplitude irregularity (age, instability)
    brightness: float = 1.0      # high-formant emphasis
    growl: float = 0.0           # sub-harmonic doubling (demonic / guttural)
    attack: float = 1.0          # consonant sharpness

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "VoiceParams":
        data = data or {}
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    def blend(self, other: "VoiceParams", weight: float) -> "VoiceParams":
        w = float(np.clip(weight, 0.0, 1.0))
        out = {}
        for f in self.__dataclass_fields__:
            out[f] = (1 - w) * getattr(self, f) + w * getattr(other, f)
        return VoiceParams(**out)

    def with_emotion(self, emotion: str, intensity: float = 1.0) -> "VoiceParams":
        mod = EMOTIONS.get((emotion or "neutral").lower())
        if not mod:
            return self
        out = VoiceParams(**self.to_dict())
        for key, delta in mod.items():
            setattr(out, key, getattr(out, key) + delta * intensity)
        out.clamp()
        return out

    def clamp(self) -> "VoiceParams":
        self.f0 = float(np.clip(self.f0, 55.0, 420.0))
        self.f0_range = float(np.clip(self.f0_range, 0.02, 0.85))
        self.formant_shift = float(np.clip(self.formant_shift, 0.62, 1.65))
        self.rasp = float(np.clip(self.rasp, 0.0, 0.95))
        self.breathiness = float(np.clip(self.breathiness, 0.0, 0.9))
        self.speed = float(np.clip(self.speed, 0.55, 1.9))
        self.energy = float(np.clip(self.energy, 0.25, 2.0))
        self.vibrato = float(np.clip(self.vibrato, 0.0, 0.6))
        self.tremor = float(np.clip(self.tremor, 0.0, 0.8))
        self.brightness = float(np.clip(self.brightness, 0.45, 1.9))
        self.growl = float(np.clip(self.growl, 0.0, 0.95))
        self.attack = float(np.clip(self.attack, 0.4, 1.8))
        return self


# --------------------------------------------------------------------------
# emotion deltas — the "expressive & dynamic voices" feature
# --------------------------------------------------------------------------
EMOTIONS: dict[str, dict[str, float]] = {
    "neutral": {},
    "happy": {"f0": 18, "f0_range": 0.12, "speed": 0.08, "energy": 0.15, "brightness": 0.15},
    "excited": {"f0": 32, "f0_range": 0.22, "speed": 0.18, "energy": 0.32, "brightness": 0.22},
    "sad": {"f0": -14, "f0_range": -0.06, "speed": -0.16, "energy": -0.25, "breathiness": 0.14},
    "angry": {"f0": 22, "f0_range": 0.16, "energy": 0.42, "rasp": 0.22, "attack": 0.32},
    "shouting": {"f0": 40, "f0_range": 0.2, "energy": 0.6, "rasp": 0.3, "attack": 0.4},
    "whisper": {"f0": -8, "energy": -0.55, "breathiness": 0.55, "rasp": -0.04, "brightness": 0.1},
    "sarcastic": {"f0_range": 0.2, "speed": -0.1, "brightness": -0.08},
    "laughing": {"f0": 26, "f0_range": 0.3, "tremor": 0.25, "energy": 0.2},
    "crying": {"f0": 14, "tremor": 0.35, "breathiness": 0.25, "f0_range": 0.25, "speed": -0.12},
    "fearful": {"f0": 24, "tremor": 0.3, "speed": 0.12, "breathiness": 0.2},
    "calm": {"f0": -6, "f0_range": -0.05, "speed": -0.08, "energy": -0.08},
    "menacing": {"f0": -18, "energy": 0.1, "growl": 0.3, "speed": -0.12, "brightness": -0.15},
}

# --------------------------------------------------------------------------
# prompt vocabulary
# --------------------------------------------------------------------------
_AGE_GENDER: list[tuple[str, dict[str, float]]] = [
    (r"\b(elderly|old man|70s|80s|aged|grandfather)\b", {"f0": 108, "rasp": 0.45, "tremor": 0.32, "speed": -0.12, "brightness": -0.2}),
    (r"\b(old woman|grandmother|elderly female)\b", {"f0": 178, "rasp": 0.35, "tremor": 0.3, "speed": -0.1}),
    (r"\b(child|kid|little (boy|girl)|8 years old|10 years old)\b", {"f0": 265, "formant_shift": 1.28, "speed": 0.12, "brightness": 0.25}),
    (r"\b(teenage female|teen girl|schoolgirl|15|16 years old female)\b", {"f0": 232, "formant_shift": 1.16, "brightness": 0.18}),
    (r"\b(teenage male|teen boy|young male|16 years old|shounen)\b", {"f0": 148, "formant_shift": 1.06, "brightness": 0.1}),
    (r"\b(young adult male|late teens|early 20s|young man)\b", {"f0": 122, "formant_shift": 1.0}),
    (r"\b(adult male|man in his (20s|30s|40s)|male)\b", {"f0": 112, "formant_shift": 0.97}),
    (r"\b(adult female|woman in her (20s|30s|40s)|female|woman)\b", {"f0": 205, "formant_shift": 1.14}),
    (r"\b(non-?human|demon|monster|creature|entity|robot)\b", {"f0": 72, "formant_shift": 0.72, "growl": 0.45, "brightness": -0.25}),
]

_TEXTURE: list[tuple[str, dict[str, float]]] = [
    (r"\b(raspy|hoarse|scratchy|crackly)\b", {"rasp": 0.3}),
    (r"\b(gravelly|gritty|guttural)\b", {"rasp": 0.34, "growl": 0.2, "brightness": -0.1}),
    (r"\b(husky)\b", {"rasp": 0.2, "breathiness": 0.16, "f0": -8}),
    (r"\b(smooth|velvety|silky|eloquent)\b", {"rasp": -0.06, "breathiness": 0.05, "brightness": 0.06}),
    (r"\b(airy|breathy|soft-?spoken|whispery)\b", {"breathiness": 0.3, "energy": -0.12}),
    (r"\b(deep|low-?pitched|lower-?register|bottomless|bass)\b", {"f0": -30, "formant_shift": -0.07, "brightness": -0.12}),
    (r"\b(ultra-?deep|resonant low-?end)\b", {"f0": -18, "formant_shift": -0.08, "growl": 0.2}),
    (r"\b(high-?pitched|shrill|squeaky|bright)\b", {"f0": 42, "brightness": 0.2}),
    (r"\b(medium-?to-?high|crisp)\b", {"f0": 16, "attack": 0.15, "brightness": 0.1}),
    (r"\b(warm|gentle|comforting)\b", {"brightness": -0.06, "breathiness": 0.08, "energy": -0.05}),
    (r"\b(nasal)\b", {"formant_shift": 0.05, "brightness": 0.12}),
    (r"\b(wheezy)\b", {"breathiness": 0.22, "tremor": 0.12}),
    (r"\b(echoey|distorted)\b", {"growl": 0.18, "brightness": -0.08}),
]

_PACING: list[tuple[str, dict[str, float]]] = [
    (r"\b(fast-?paced|rapid ?fire|quick cadence|hyperactive|fast)\b", {"speed": 0.22}),
    (r"\b(slow(ly)?|deliberate|relaxed cadence|slow pacing)\b", {"speed": -0.2}),
    (r"\b(erratic|unpredictable|high-?contrast)\b", {"f0_range": 0.28, "tremor": 0.18}),
    (r"\b(monotone|flat|minimal emotion|detached|aloof)\b", {"f0_range": -0.1}),
    (r"\b(dynamic pitch|dynamic|theatrical|dramatic)\b", {"f0_range": 0.22, "energy": 0.12}),
]

_ATTITUDE: list[tuple[str, dict[str, float]]] = [
    (r"\b(loud|energetic|passionate|intense|battle-?cry|screaming)\b", {"energy": 0.3, "f0_range": 0.14, "attack": 0.2}),
    (r"\b(cold|stoic|composed|calculating|chilling|confident)\b", {"f0_range": -0.06, "speed": -0.06}),
    (r"\b(bubbly|cheerful|sweet|genki|upbeat|enthusiastic)\b", {"f0": 18, "f0_range": 0.18, "speed": 0.1, "brightness": 0.14}),
    (r"\b(seductive|teasing|sultry|ara ara)\b", {"f0": -10, "breathiness": 0.2, "speed": -0.12}),
    (r"\b(flustered|defensive|huffy|tsundere)\b", {"f0_range": 0.24, "attack": 0.2, "tremor": 0.1}),
    (r"\b(menacing|dangerous|terrifying|unhinged|manic)\b", {"growl": 0.22, "f0_range": 0.2}),
    (r"\b(condescending|arrogant|smug)\b", {"f0_range": 0.12, "speed": -0.06, "brightness": 0.05}),
    (r"\b(friendly|eccentric)\b", {"f0_range": 0.1, "brightness": 0.05}),
]

_RULES = _AGE_GENDER + _TEXTURE + _PACING + _ATTITUDE


def design_voice(prompt: str, seed: str | None = None) -> VoiceParams:
    """Parse a free-form voice description into synthesis parameters."""
    text = (prompt or "").lower()
    params = VoiceParams()
    matched_identity = False

    for pattern, deltas in _AGE_GENDER:
        if re.search(pattern, text):
            for key, value in deltas.items():
                if key in ("f0", "formant_shift") and not matched_identity:
                    setattr(params, key, value if key == "f0" else value)
                else:
                    setattr(params, key, getattr(params, key) + value)
            matched_identity = True
            break
    if not matched_identity:
        params.f0, params.formant_shift = 135.0, 1.0

    for pattern, deltas in _TEXTURE + _PACING + _ATTITUDE:
        if re.search(pattern, text):
            for key, value in deltas.items():
                setattr(params, key, getattr(params, key) + value)

    for emotion in EMOTIONS:
        if emotion != "neutral" and re.search(rf"\b{emotion}\b", text):
            params = params.with_emotion(emotion, 0.6)

    # deterministic micro-variation so two similar prompts still differ
    if seed:
        h = hashlib.sha256(seed.encode()).digest()
        jitter = (h[0] / 255.0 - 0.5)
        params.f0 *= 1.0 + 0.05 * jitter
        params.formant_shift *= 1.0 + 0.03 * (h[1] / 255.0 - 0.5)
        params.rasp += 0.03 * (h[2] / 255.0 - 0.5)

    return params.clamp()


def params_from_embedding(embedding: np.ndarray) -> VoiceParams:
    """Derive synthesis parameters from a cloned speaker embedding.

    This is what makes a *cloned* voice sound like its reference in the
    offline engine: pitch, vocal-tract length and texture are read back out
    of the embedding statistics produced by `dsp.speaker_embedding`.
    """
    emb = np.asarray(embedding, dtype=np.float32).ravel()
    if emb.size < 8:
        return VoiceParams().clamp()

    # layout mirrors dsp.speaker_embedding: 20 mean | 20 std | 20 delta | 4 extras
    timbre = emb[:20]
    spread = emb[20:40] if emb.size >= 40 else timbre
    extras = emb[60:64] if emb.size >= 64 else np.zeros(4, dtype=np.float32)

    # the vector is L2-normalised on the way in; the three shape blocks are
    # z-normed so their combined norm is a known sqrt(60), which lets us undo
    # the scaling and read the physical descriptors back out
    block_norm = float(np.linalg.norm(emb[:60]))
    scale = (np.sqrt(60.0) / block_norm) if block_norm > 1e-6 else 1.0
    extras = extras * scale

    f0 = (float(extras[0]) / 3.5 + 1.0) * 200.0
    if not np.isfinite(f0) or f0 < 55.0:  # unvoiced reference — use spectral centroid
        centroid = float(np.argmax(timbre) / max(1, timbre.size))
        f0 = 95.0 + centroid * 190.0

    low = float(np.mean(timbre[:6]))
    high = float(np.mean(timbre[-6:]))
    tilt = float(np.clip((high - low) * 0.35, -1.0, 1.0))

    params = VoiceParams(
        f0=f0,
        f0_range=float(0.10 + 0.5 * np.clip(np.mean(np.abs(spread)) * 4.0, 0.0, 1.0)),
        formant_shift=float(np.clip(1.0 + tilt * 0.35, 0.7, 1.5)),
        rasp=float(np.clip(abs(float(extras[2])) * 0.8, 0.02, 0.5)),
        breathiness=float(np.clip(0.08 + max(0.0, tilt) * 0.4, 0.03, 0.6)),
        brightness=float(np.clip(1.0 + tilt * 0.6, 0.6, 1.6)),
    )
    return params.clamp()


# --------------------------------------------------------------------------
# preset library — studio neural voices + anime archetypes
# --------------------------------------------------------------------------
@dataclass
class Preset:
    id: str
    name: str
    category: str
    language: str
    prompt: str
    tags: list[str] = field(default_factory=list)

    @property
    def params(self) -> VoiceParams:
        return design_voice(self.prompt, seed=self.id)


PRESETS: list[Preset] = [
    # --- studio / narration -------------------------------------------------
    Preset("studio_anchor_m", "Marcus — News Anchor", "studio", "en",
           "An adult male in his 40s. Smooth, authoritative, articulate broadcast delivery. "
           "Medium-low pitch, calm, deliberate pacing, neutral American accent.",
           ["narration", "news", "male"]),
    Preset("studio_anchor_f", "Elena — News Anchor", "studio", "en",
           "An adult female in her 30s. Crisp, clear, professional and warm. Medium pitch, "
           "steady pacing, neutral accent.", ["narration", "news", "female"]),
    Preset("studio_narrator_gentle", "Aria — Gentle Narrator", "studio", "en",
           "An adult female in her late 20s. Soft-spoken, warm, comforting and airy. "
           "Slow relaxed cadence, gentle exhales.", ["audiobook", "calm", "female"]),
    Preset("studio_documentary", "Rhys — Documentary", "studio", "en",
           "An adult male in his 50s. Deep, resonant, warm lower-register. Slow deliberate "
           "pacing, thoughtful and composed.", ["documentary", "male"]),
    Preset("studio_explainer", "Kai — Explainer", "studio", "en",
           "A young adult male, early 20s. Bright, friendly, upbeat and enthusiastic. "
           "Fast-paced, energetic delivery.", ["elearning", "male"]),

    # --- anime archetypes ---------------------------------------------------
    Preset("anime_shounen_lead", "Determined Shounen Lead", "anime", "en",
           "A young male anime protagonist around 16 years old. High energy, loud, passionate, "
           "and fiercely determined. Raspy, slightly hoarse voice texture with intense breath "
           "control. Upbeat and enthusiastic delivery, dynamic pitch variations, screaming "
           "battle-cry potential.", ["shounen", "hero", "male", "teen"]),
    Preset("anime_stoic_rival", "Stoic / Edgy Anti-Hero", "anime", "en",
           "A young adult male, late teens to early 20s. Deep, ultra-calm, low-pitched, and "
           "slightly husky tone. Cold, detached, and aloof delivery with slow pacing. Minimal "
           "emotion, whispery undertones, confident and stoic.", ["rival", "anti-hero", "male"]),
    Preset("anime_smooth_villain", "Arrogant Mastermind", "anime", "en",
           "An adult male in his 30s with a deep, velvety, smooth, and eloquent voice. Highly "
           "theatrical, condescending, and calculating. Rich lower-register, articulate "
           "pronunciation, speaking with a calm, chilling composure.", ["villain", "male"]),
    Preset("anime_manic_villain", "Manic / Unhinged Antagonist", "anime", "en",
           "A male character with an unpredictable, high-contrast vocal delivery. Alternates "
           "rapidly between a high-pitched, playful giggle and a deep, terrifying, gravelly "
           "growl. Unhinged, raspy, theatrical, and erratic pacing.", ["villain", "chaotic", "male"]),
    Preset("anime_genki_girl", "Genki Girl / Bubbly Heroine", "anime", "en",
           "A teenage female anime character. Very high-pitched, bright, hyperactive, and "
           "bubbly. Fast-paced talking speed, sweet tone, filled with squeaks and gasp-like "
           "breaths. Cheerful and energetic.", ["heroine", "genki", "female", "teen"]),
    Preset("anime_tsundere", "Tsundere", "anime", "en",
           "A teenage female character with a crisp, medium-to-high pitch. Sharp, flustered, "
           "aggressive yet defensive delivery. Quick cadence, easily flustered tone, shifting "
           "instantly from huffy and harsh to soft and hesitant.", ["tsundere", "female", "teen"]),
    Preset("anime_ara_ara", "Ara Ara / Mature Older Sister", "anime", "en",
           "An adult female in her late 20s. Deep, smooth, low, and seductive tone. "
           "Soft-spoken, teasing, warm, with a slow, relaxed cadence and gentle exhales. "
           "Comforting yet commanding.", ["onee-san", "female"]),
    Preset("anime_wise_master", "Old Wise Master", "anime", "en",
           "An elderly male in his 70s. Very raspy, gravelly, crackly, and aged vocal texture. "
           "Warm, slightly wheezy, eccentric, and friendly demeanor with sudden bursts of loud, "
           "energetic laughter.", ["mentor", "elder", "male"]),
    Preset("anime_demon_lord", "Demon / Overlord Monstrosity", "anime", "en",
           "A non-human male demonic entity. Bottomless, ultra-deep, gravelly, resonant low-end "
           "bass voice. Menacing, echoey, distorted, slow pacing with a dangerous and guttural "
           "growl.", ["monster", "villain", "non-human"]),
    Preset("anime_kuudere", "Kuudere / Quiet Analyst", "anime", "en",
           "A teenage female character. Flat, monotone, soft and airy delivery with minimal "
           "emotion. Medium pitch, slow deliberate pacing, detached and calm.", ["kuudere", "female"]),
    Preset("anime_loli_mascot", "Mascot / Small Companion", "anime", "en",
           "A child character, very high-pitched and squeaky. Fast-paced, cheerful, playful and "
           "bright with bouncy dynamic pitch.", ["mascot", "child"]),

    # --- gaming / character --------------------------------------------------
    Preset("game_grizzled_soldier", "Grizzled Soldier", "character", "en",
           "An adult male in his 40s. Gravelly, gritty, low-pitched and commanding. "
           "Deliberate pacing, intense and confident.", ["gaming", "male"]),
    Preset("game_ai_companion", "Synthetic Companion", "character", "en",
           "A non-human entity with a smooth, calm, flat monotone delivery. Medium pitch, "
           "precise articulation, minimal emotion.", ["gaming", "robot"]),
]

PRESETS_BY_ID: dict[str, Preset] = {p.id: p for p in PRESETS}
