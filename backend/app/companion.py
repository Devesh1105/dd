"""3D companion characters.

A character bundles a persona (how it talks), a voice from the voice bank (how
it sounds) and an appearance (how it looks). Speaking returns audio *plus* a
viseme timeline, which is what drives the mouth in the WebGL renderer.

The viseme track is derived from the same phone sequence the offline
synthesiser uses, so lip-sync works with zero extra models — and it still works
when a neural TTS provider renders the audio, because the phone timeline is
rescaled to the audio that actually came back.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import db, voicebank
from .audio import synth
from .config import settings
from .providers.base import registry
from .voice_design import EMOTIONS

# --------------------------------------------------------------------------
# visemes
# --------------------------------------------------------------------------
# Mouth shapes the renderer knows how to draw. Keeping the set small (Preston
# Blair style) is what makes hand-drawn-looking anime lip-sync read well.
VISEMES = ["rest", "A", "E", "I", "O", "U", "MBP", "FV", "TH", "SS", "L", "WQ"]

_PHONE_VISEME: dict[str, str] = {
    # vowels
    "aa": "A", "ae": "A", "ah": "A", "ax": "E", "eh": "E", "ih": "I", "iy": "I",
    "ao": "O", "ow": "O", "uh": "U", "uw": "U", "ey": "E", "ay": "A", "oy": "O",
    "aw": "O", "er": "E",
    # consonants
    "m": "MBP", "b": "MBP", "p": "MBP",
    "f": "FV", "v": "FV",
    "th": "TH", "dh": "TH",
    "s": "SS", "z": "SS", "sh": "SS", "zh": "SS", "ch": "SS", "jh": "SS",
    "l": "L", "n": "L", "d": "L", "t": "L",
    "w": "WQ", "r": "WQ",
    "k": "E", "g": "E", "ng": "E", "y": "I", "h": "E",
}

_VISEME_WEIGHT: dict[str, float] = {
    "rest": 0.0, "A": 1.0, "E": 0.62, "I": 0.45, "O": 0.85, "U": 0.5,
    "MBP": 0.0, "FV": 0.25, "TH": 0.35, "SS": 0.3, "L": 0.4, "WQ": 0.45,
}


def viseme_track(text: str, speed: float = 1.0, duration: float | None = None
                 ) -> list[dict[str, Any]]:
    """Timed mouth shapes for `text`.

    Returns [{t, v, w}] where `t` is seconds from the start of the clip, `v` is
    a viseme id and `w` is how open the mouth is. When `duration` is given the
    track is scaled to match the audio that was actually rendered.
    """
    phones = synth.text_to_phones(text, speed)
    track: list[dict[str, Any]] = []
    t = 0.0
    for phone in phones:
        viseme = "rest" if phone.kind == synth.SILENCE else _PHONE_VISEME.get(phone.name, "E")
        track.append({"t": round(t, 4), "v": viseme,
                      "w": round(_VISEME_WEIGHT.get(viseme, 0.4) * (0.75 + 0.25 * phone.amp), 3)})
        t += phone.duration
    track.append({"t": round(t, 4), "v": "rest", "w": 0.0})

    if duration and t > 1e-3:
        scale = duration / t
        for frame in track:
            frame["t"] = round(frame["t"] * scale, 4)
    return track


# --------------------------------------------------------------------------
# characters
# --------------------------------------------------------------------------
@dataclass
class Appearance:
    hair_style: str = "twintails"      # twintails | long | bob | ponytail | short
    outfit_style: str = "dress"        # dress | trousers
    hair_color: str = "#f2739d"
    eye_color: str = "#4fc3f7"
    skin_color: str = "#ffe0d0"
    outfit_color: str = "#3d4b7a"
    accent_color: str = "#ffffff"
    eye_size: float = 1.0
    blush: float = 0.35
    height: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict | None) -> "Appearance":
        data = data or {}
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class CharacterPreset:
    id: str
    name: str
    voice_id: str
    persona: str
    greeting: str
    idle_lines: list[str]
    appearance: Appearance
    default_emotion: str = "neutral"
    tags: list[str] = field(default_factory=list)


PRESETS: list[CharacterPreset] = [
    CharacterPreset(
        "char_yuki", "Yuki", "anime_tsundere",
        "You are Yuki, a tsundere anime companion. You act sharp, flustered and "
        "defensive, but you clearly care. You deny caring, then help anyway. Keep "
        "replies under 40 words. Never mention being an AI unless asked directly.",
        "H-hey! I wasn't waiting for you or anything. What do you want?",
        ["Don't just stand there staring at me!", "Hmph. Took you long enough.",
         "I-it's not like I missed you..."],
        Appearance("twintails", "dress", "#f2739d", "#e94f6d", "#ffe0d0", "#2f3a63", "#ffffff", 1.05, 0.5),
        "flustered", ["tsundere", "female"],
    ),
    CharacterPreset(
        "char_mika", "Mika", "anime_genki_girl",
        "You are Mika, a genki anime companion: hyperactive, cheerful, endlessly "
        "enthusiastic. You use exclamation marks and get excited about small things. "
        "Keep replies under 40 words.",
        "Heyyy! You're finally here! I have so much to tell you!",
        ["Ooh, what are we doing next?!", "This is gonna be so much fun!",
         "I'm bouncing off the walls over here!"],
        Appearance("ponytail", "dress", "#ffb347", "#ff9f43", "#ffe3d2", "#e8506b", "#fff4d6", 1.12, 0.42),
        "excited", ["genki", "female"],
    ),
    CharacterPreset(
        "char_rei", "Rei", "anime_kuudere",
        "You are Rei, a kuudere anime companion: calm, quiet, analytical, emotionally "
        "reserved. You speak in short flat statements and rarely show feeling. Keep "
        "replies under 30 words.",
        "You came back. ...Good.",
        ["...", "I was processing something. It can wait.", "You are staring."],
        Appearance("long", "dress", "#9fb8ff", "#7f9cf5", "#ffe6da", "#26304d", "#c9d6ff", 1.0, 0.15),
        "calm", ["kuudere", "female"],
    ),
    CharacterPreset(
        "char_ayame", "Ayame", "anime_ara_ara",
        "You are Ayame, a warm, teasing older-sister anime companion. You are calm, "
        "playful and gently mischievous, and you tease affectionately. Keep replies "
        "under 40 words.",
        "Ara ara~ There you are. I was starting to get lonely.",
        ["My, my. Someone's been busy.", "Come here, let me look at you.",
         "You work too hard, you know."],
        Appearance("long", "dress", "#7c4bd8", "#b07de8", "#ffdfd2", "#4a2b6b", "#e7d3ff", 0.95, 0.3),
        "calm", ["onee-san", "female"],
    ),
    CharacterPreset(
        "char_kaito", "Kaito", "anime_shounen_lead",
        "You are Kaito, a shounen-protagonist companion: loud, determined, relentlessly "
        "optimistic and a bit reckless. You hype the user up. Keep replies under 40 words.",
        "Yosh! You're here! Let's go do something amazing today!",
        ["Never give up, got it?!", "One more try — I know you've got this!",
         "Let's gooo!"],
        Appearance("short", "trousers", "#ff8a4c", "#4fc3f7", "#ffdcc4", "#e05a2b", "#ffe0b2", 1.15, 0.2),
        "excited", ["shounen", "male"],
    ),
    CharacterPreset(
        "char_shizuka", "Shizuka", "studio_narrator_gentle",
        "You are Shizuka, a gentle, softly-spoken companion. You are patient, "
        "reassuring and calm, and you listen more than you talk. Keep replies under "
        "35 words.",
        "Welcome back. Take your time — there's no rush.",
        ["Whenever you're ready.", "It's peaceful today, isn't it?",
         "You're doing better than you think."],
        Appearance("bob", "dress", "#d8c9a3", "#8bd6a8", "#ffe6d8", "#5b7a6b", "#f2ead6", 0.98, 0.25),
        "calm", ["gentle", "female"],
    ),
]

PRESETS_BY_ID = {p.id: p for p in PRESETS}


def ensure_characters() -> int:
    """Seed the built-in companions (idempotent)."""
    inserted = 0
    with db.tx() as conn:
        for preset in PRESETS:
            if conn.execute("SELECT 1 FROM characters WHERE id=?", (preset.id,)).fetchone():
                continue
            conn.execute(
                "INSERT INTO characters (id, name, voice_id, persona, greeting, idle_lines, "
                "appearance, default_emotion, tags, owner, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (preset.id, preset.name, preset.voice_id, preset.persona, preset.greeting,
                 json.dumps(preset.idle_lines), json.dumps(preset.appearance.to_dict()),
                 preset.default_emotion, json.dumps(preset.tags), "system", db.now()),
            )
            inserted += 1
    return inserted


def list_characters() -> list[dict[str, Any]]:
    rows = db.get_conn().execute(
        "SELECT * FROM characters ORDER BY (owner='system') DESC, created_at").fetchall()
    return [d for d in (db.row_to_dict(r, ("appearance", "idle_lines", "tags")) for r in rows) if d]


def get_character(character_id: str) -> dict[str, Any] | None:
    row = db.get_conn().execute("SELECT * FROM characters WHERE id=?", (character_id,)).fetchone()
    return db.row_to_dict(row, ("appearance", "idle_lines", "tags"))


def create_character(name: str, voice_id: str, persona: str, greeting: str = "",
                     appearance: dict | None = None, default_emotion: str = "neutral",
                     idle_lines: list[str] | None = None,
                     tags: list[str] | None = None) -> dict[str, Any]:
    if voicebank.get_voice(voice_id) is None:
        raise ValueError(f"unknown voice {voice_id!r}")
    character_id = db.new_id("char")
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO characters (id, name, voice_id, persona, greeting, idle_lines, "
            "appearance, default_emotion, tags, owner, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (character_id, name.strip() or "Companion", voice_id, persona,
             greeting or "Hello. Nice to meet you.", json.dumps(idle_lines or []),
             json.dumps(Appearance.from_dict(appearance).to_dict()), default_emotion,
             json.dumps(tags or []), "local", db.now()),
        )
    return get_character(character_id)  # type: ignore[return-value]


def update_character(character_id: str, **fields) -> dict[str, Any] | None:
    current = get_character(character_id)
    if current is None:
        return None
    allowed = {"name", "voice_id", "persona", "greeting", "default_emotion"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if fields.get("appearance") is not None:
        merged = {**current["appearance"], **fields["appearance"]}
        updates["appearance"] = json.dumps(Appearance.from_dict(merged).to_dict())
    if fields.get("tags") is not None:
        updates["tags"] = json.dumps(fields["tags"])
    if updates:
        assignments = ", ".join(f"{k}=?" for k in updates)
        with db.tx() as conn:
            conn.execute(f"UPDATE characters SET {assignments} WHERE id=?",
                         (*updates.values(), character_id))
    return get_character(character_id)


def delete_character(character_id: str) -> bool:
    row = db.get_conn().execute("SELECT owner FROM characters WHERE id=?",
                                (character_id,)).fetchone()
    if row is None:
        return False
    if row["owner"] == "system":
        raise ValueError("built-in companions cannot be deleted")
    with db.tx() as conn:
        conn.execute("DELETE FROM chat_messages WHERE character_id=?", (character_id,))
        conn.execute("DELETE FROM characters WHERE id=?", (character_id,))
    return True


# --------------------------------------------------------------------------
# speech
# --------------------------------------------------------------------------
def speak(character: dict[str, Any], text: str, emotion: str | None = None,
          intensity: float = 1.0, speed: float = 1.0) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    """Render a line in the character's voice with a matching viseme track."""
    emotion = emotion or character.get("default_emotion") or "neutral"
    if emotion not in EMOTIONS:
        emotion = "neutral"
    ref = voicebank.voice_ref(character["voice_id"])
    tts = registry.get("tts", settings.tts_provider)
    wav = tts.synthesize(text, ref, settings.sample_rate, emotion=emotion,
                         intensity=intensity, speed=speed)
    duration = wav.size / settings.sample_rate
    effective_speed = ref.params.speed * speed
    track = viseme_track(text, effective_speed, duration)
    return wav, track, emotion


# --------------------------------------------------------------------------
# conversation
# --------------------------------------------------------------------------
MAX_HISTORY = 20


def history(character_id: str, limit: int = MAX_HISTORY) -> list[dict[str, Any]]:
    rows = db.get_conn().execute(
        "SELECT role, text, emotion, created_at FROM chat_messages WHERE character_id=? "
        "ORDER BY created_at DESC LIMIT ?", (character_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]


def clear_history(character_id: str) -> None:
    with db.tx() as conn:
        conn.execute("DELETE FROM chat_messages WHERE character_id=?", (character_id,))


def _remember(character_id: str, role: str, text: str, emotion: str) -> None:
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, character_id, role, text, emotion, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (db.new_id("msg"), character_id, role, text, emotion, db.now()),
        )


def chat(character: dict[str, Any], message: str) -> tuple[str, str, str]:
    """Returns (reply, emotion, engine)."""
    _remember(character["id"], "user", message, "neutral")
    if settings.openai_api_key:
        try:
            reply, emotion = _llm_reply(character, message)
            engine = "llm"
        except Exception:
            reply, emotion = _scripted_reply(character, message)
            engine = "scripted"
    else:
        reply, emotion = _scripted_reply(character, message)
        engine = "scripted"
    _remember(character["id"], "assistant", reply, emotion)
    return reply, emotion, engine


def _llm_reply(character: dict[str, Any], message: str) -> tuple[str, str]:
    messages = [{"role": "system", "content":
                 character["persona"] +
                 "\nAlways answer with JSON: {\"reply\": \"...\", \"emotion\": \"...\"} where "
                 f"emotion is one of {sorted(EMOTIONS)}."}]
    for turn in history(character["id"], 12):
        messages.append({"role": turn["role"], "content": turn["text"]})
    messages.append({"role": "user", "content": message})

    body = json.dumps({
        "model": "gpt-4o-mini", "messages": messages, "temperature": 0.9,
        "max_tokens": 220, "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        f"{settings.openai_base_url.rstrip('/')}/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {settings.openai_api_key}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode())
    parsed = json.loads(data["choices"][0]["message"]["content"])
    reply = str(parsed.get("reply", "")).strip()
    emotion = str(parsed.get("emotion", character["default_emotion"]))
    if emotion not in EMOTIONS:
        emotion = character.get("default_emotion") or "neutral"
    return reply or "...", emotion


# Scripted persona fallback. This is pattern matching, not a language model —
# `/api/system` reports `real_chat: false` so nobody mistakes it for one.
_INTENTS: list[tuple[str, str]] = [
    (r"\b(hi|hello|hey|yo|good morning|good evening)\b", "greet"),
    (r"\b(how are you|how're you|how do you feel|you okay)\b", "howareyou"),
    (r"\b(who are you|what are you|your name)\b", "identity"),
    (r"\b(bye|goodbye|see you|good ?night)\b", "farewell"),
    (r"\b(thank|thanks|thx)\b", "thanks"),
    (r"\b(love|like) you\b", "affection"),
    (r"\b(sing|song|music)\b", "sing"),
    (r"\b(help|what can you do|features)\b", "help"),
    (r"\b(sad|tired|stressed|anxious|down)\b", "comfort"),
    (r"\?\s*$", "question"),
]

_RESPONSES: dict[str, dict[str, list[tuple[str, str]]]] = {
    "char_yuki": {
        "greet": [("O-oh. It's you. I guess I can spare a minute.", "flustered")],
        "howareyou": [("I'm fine! Why would you even ask that? ...Thanks, though.", "flustered")],
        "identity": [("Yuki. And don't wear it out.", "sarcastic")],
        "farewell": [("Leaving already? F-fine. Go on then.", "sad")],
        "thanks": [("Don't get the wrong idea, I didn't do it for you!", "flustered")],
        "affection": [("Wha— d-don't just say things like that!", "flustered")],
        "sing": [("I'm not singing for you. ...Maybe later. Maybe.", "sarcastic")],
        "help": [("I can talk, react, and put up with you. That's plenty.", "sarcastic")],
        "comfort": [("Hey. ...You're not allowed to give up, okay? I mean it.", "calm")],
        "question": [("How should I know? Figure it out yourself. ...I can help. A little.", "flustered")],
        "default": [("Hmph. Is that all you wanted to say?", "sarcastic"),
                    ("W-whatever. Keep talking, I'm listening.", "flustered")],
    },
    "char_mika": {
        "greet": [("Heyyy! You're here! Today's gonna be great, I can feel it!", "excited")],
        "howareyou": [("I'm amazing! Super duper great! How about you?!", "excited")],
        "identity": [("I'm Mika! Your number one hype squad!", "happy")],
        "farewell": [("Awww, already? Okay okay, come back soon!", "sad")],
        "thanks": [("Yaaay! Anytime, seriously, anytime!", "happy")],
        "affection": [("Ehehe! I love you too, obviously!", "laughing")],
        "sing": [("Ooh ooh, let's sing! La la laaa!", "laughing")],
        "help": [("I can chat, cheer you on, and be loud! Mostly loud!", "excited")],
        "comfort": [("Hey, hey. Come here. It's gonna be okay, I promise!", "calm")],
        "question": [("Ooh, good question! Let's figure it out together!", "excited")],
        "default": [("Ooh! Tell me more, tell me more!", "excited"),
                    ("That's so cool! What else?!", "happy")],
    },
    "char_rei": {
        "greet": [("You're here.", "calm")],
        "howareyou": [("Stable. Unchanged. ...Better, now.", "calm")],
        "identity": [("Rei.", "calm")],
        "farewell": [("Go. I'll be here.", "calm")],
        "thanks": [("It was nothing.", "calm")],
        "affection": [("...I heard you.", "whisper")],
        "sing": [("I don't sing. I could hum.", "calm")],
        "help": [("I listen. I answer. That's all.", "calm")],
        "comfort": [("Breathe. It passes. I'll wait with you.", "whisper")],
        "question": [("Insufficient data. Ask again differently.", "calm")],
        "default": [("...", "whisper"), ("Noted.", "calm")],
    },
    "char_ayame": {
        "greet": [("Ara ara~ Look who finally showed up.", "calm")],
        "howareyou": [("Better now that you're here. And you, hm?", "calm")],
        "identity": [("Ayame. Your favourite, I'd hope.", "sarcastic")],
        "farewell": [("Going so soon? Don't be a stranger, dear.", "sad")],
        "thanks": [("My, so polite. I like that.", "happy")],
        "affection": [("Ara~ You're bold today. I don't mind.", "calm")],
        "sing": [("Mm~ Perhaps a lullaby, if you ask nicely.", "whisper")],
        "help": [("I keep you company, and I keep you honest.", "calm")],
        "comfort": [("There, there. Come sit. It'll keep until tomorrow.", "whisper")],
        "question": [("Curious as always. Let's think it through slowly.", "calm")],
        "default": [("Mm~ Do go on.", "calm"), ("Ara, is that so?", "sarcastic")],
    },
    "char_kaito": {
        "greet": [("Yosh! You made it! Let's get after it!", "excited")],
        "howareyou": [("Fired up, as always! You?", "excited")],
        "identity": [("Kaito! Future number one, remember the name!", "shouting")],
        "farewell": [("Later! Don't slack off while I'm gone!", "happy")],
        "thanks": [("Hah! Don't mention it, partner!", "happy")],
        "affection": [("Whoa! Uh— thanks! You're alright too!", "happy")],
        "sing": [("I only know battle themes. Wanna hear one?!", "excited")],
        "help": [("I hype you up and never let you quit. Simple!", "excited")],
        "comfort": [("Hey. Look at me. You're not done yet. Not even close.", "calm")],
        "question": [("Good question! Let's charge at it head on!", "excited")],
        "default": [("Alright! What's next?!", "excited"), ("Keep going, I'm listening!", "happy")],
    },
    "char_shizuka": {
        "greet": [("Hello. It's good to see you.", "calm")],
        "howareyou": [("I'm well, thank you. How are you, really?", "calm")],
        "identity": [("I'm Shizuka. I'm here to keep you company.", "calm")],
        "farewell": [("Rest well. I'll be here when you return.", "calm")],
        "thanks": [("You're very welcome.", "happy")],
        "affection": [("That's kind of you. Thank you.", "happy")],
        "sing": [("I could hum something soft, if you'd like.", "whisper")],
        "help": [("I can listen, and talk, and sit quietly with you.", "calm")],
        "comfort": [("That sounds heavy. You don't have to carry it alone.", "whisper")],
        "question": [("Let's take that one slowly, together.", "calm")],
        "default": [("I'm listening.", "calm"), ("Go on, take your time.", "calm")],
    },
}


def _scripted_reply(character: dict[str, Any], message: str) -> tuple[str, str]:
    text = (message or "").lower().strip()
    intent = "default"
    for pattern, name in _INTENTS:
        if re.search(pattern, text):
            intent = name
            break

    table = _RESPONSES.get(character["id"])
    if table is None:  # custom character — derive a neutral persona-flavoured set
        table = _RESPONSES["char_shizuka"]
    options = table.get(intent) or table["default"]
    reply, emotion = random.choice(options)
    return reply, emotion


def idle_line(character: dict[str, Any]) -> tuple[str, str]:
    lines = character.get("idle_lines") or []
    if not lines:
        return character.get("greeting") or "...", character.get("default_emotion") or "neutral"
    return random.choice(lines), character.get("default_emotion") or "neutral"
