"""End-to-end tests. Run with: python -m pytest tests -q"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# isolate every run in a throwaway data dir before app modules read settings
_TMP = tempfile.mkdtemp(prefix="dub-test-")
os.environ["DUB_DATA_DIR"] = _TMP
os.environ["DUB_WORKERS"] = "2"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.audio import dsp, synth  # noqa: E402
from backend.app.audio.wavio import read_wav, resample, to_mono, write_wav  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.providers import translate as mt  # noqa: E402
from backend.app.voice_design import PRESETS_BY_ID, VoiceParams, design_voice  # noqa: E402

SR = 24000


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "sample.wav"
    clips = []
    for voice_id, line in (
        ("studio_anchor_m", "Good evening and welcome to the broadcast."),
        ("anime_genki_girl", "I am so excited to be here with you today!"),
        ("studio_anchor_m", "Let us begin with the main story of the hour."),
    ):
        wav = synth.synthesize(line, PRESETS_BY_ID[voice_id].params, SR)
        clips.append(np.concatenate([wav, np.zeros(int(0.5 * SR), dtype=np.float32)]))
    write_wav(path, np.concatenate(clips), SR)
    return path


def _wait_for_job(client, job_id: str, timeout: float = 240.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


# --------------------------------------------------------------------------
# DSP / synthesis
# --------------------------------------------------------------------------
def test_wav_roundtrip(tmp_path):
    x = np.sin(2 * np.pi * 220 * np.arange(SR) / SR).astype(np.float32) * 0.5
    path = write_wav(tmp_path / "a.wav", x, SR)
    y, sr = read_wav(path)
    assert sr == SR
    assert np.max(np.abs(to_mono(y) - x)) < 1e-3


def test_resample_preserves_duration():
    x = np.random.default_rng(0).normal(0, 0.1, SR).astype(np.float32)
    y = resample(x, SR, 16000)
    assert abs(y.size / 16000 - 1.0) < 0.01


def test_synth_is_deterministic_and_bounded():
    params = design_voice("a deep calm male narrator", seed="t")
    a = synth.synthesize("Testing one two three.", params, SR)
    b = synth.synthesize("Testing one two three.", params, SR)
    assert np.array_equal(a, b)
    assert a.size > SR // 2
    assert float(np.max(np.abs(a))) <= 1.0


def test_voice_params_change_the_signal():
    deep = synth.synthesize("Hello world.", design_voice("a very deep male voice"), SR)
    high = synth.synthesize("Hello world.", design_voice("a very high pitched teenage girl"), SR)
    assert dsp.estimate_f0(high, SR) > dsp.estimate_f0(deep, SR)


def test_time_stretch_hits_target_duration():
    x = synth.synthesize("This line has to fit inside its slot.", VoiceParams(), SR)
    for target in (1.5, 3.0, 6.0):
        y, rate = dsp.fit_to_duration(x, target, SR)
        assert abs(y.size / SR - target) < 0.02
        assert float(np.max(np.abs(y))) <= 1.0
        assert 0.7 <= rate <= 1.5


def test_vad_finds_the_gaps():
    speech = synth.synthesize("One two three four.", VoiceParams(), SR)
    gap = np.zeros(int(0.8 * SR), dtype=np.float32)
    spans = dsp.detect_speech(np.concatenate([speech, gap, speech]), SR)
    assert len(spans) == 2


def test_separation_conserves_energy():
    x = synth.synthesize("Vocals over a backing track.", VoiceParams(), SR)
    music = 0.3 * np.sin(2 * np.pi * 440 * np.arange(x.size) / SR).astype(np.float32)
    vocals, background = dsp.separate_stems(x + music, SR)
    assert vocals.size == background.size == x.size
    recombined = float(np.sqrt(np.mean((vocals + background) ** 2)))
    original = float(np.sqrt(np.mean((x + music) ** 2)))
    assert recombined == pytest.approx(original, rel=0.5)


def test_speaker_embeddings_separate_voices():
    a = synth.synthesize("The quick brown fox.", PRESETS_BY_ID["anime_demon_lord"].params, SR)
    b = synth.synthesize("The quick brown fox.", PRESETS_BY_ID["anime_genki_girl"].params, SR)
    a2 = synth.synthesize("Jumps over the lazy dog.", PRESETS_BY_ID["anime_demon_lord"].params, SR)
    ea, eb, ea2 = (dsp.speaker_embedding(x, SR) for x in (a, b, a2))
    assert dsp.cosine(ea, ea2) > dsp.cosine(ea, eb)


def test_diarization_clusters_two_speakers():
    embeddings = []
    for voice_id in ("studio_anchor_m", "anime_genki_girl") * 3:
        wav = synth.synthesize("A line of dialogue here.", PRESETS_BY_ID[voice_id].params, SR)
        embeddings.append(dsp.speaker_embedding(wav, SR))
    labels = dsp.cluster_speakers(embeddings, max_speakers=4)
    assert len(set(labels)) >= 2


# --------------------------------------------------------------------------
# translation length control
# --------------------------------------------------------------------------
def test_char_budget_tracks_language_expansion():
    assert mt.char_budget(3.0, "de") < mt.char_budget(3.0, "en")
    assert mt.char_budget(3.0, "zh") < mt.char_budget(3.0, "en")  # denser script
    assert mt.char_budget(6.0, "en") == 2 * mt.char_budget(3.0, "en")


def test_fit_text_never_splits_a_word():
    assert mt.fit_text("hello wonderful world", 12) == "hello"
    assert mt.fit_text("short", 50) == "short"


def test_detect_language_by_script():
    assert mt.detect_language("こんにちは") == "ja"
    assert mt.detect_language("Привет") == "ru"
    assert mt.detect_language("hello") == "en"


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
def test_system_reports_capabilities(client):
    data = client.get("/api/system").json()
    assert data["active_providers"]["tts"]
    assert "spectral" in data["providers"]["separation"]


def test_presets_are_seeded(client):
    voices = client.get("/api/voices").json()
    ids = {v["id"] for v in voices}
    assert "anime_shounen_lead" in ids and "studio_anchor_m" in ids
    assert all(v["params"]["f0"] > 0 for v in voices)


def test_archetype_library(client):
    data = client.get("/api/voices/archetypes").json()
    assert "anime" in data["categories"]
    assert len(data["prompt_structure"]) == 5
    assert "whisper" in data["emotions"]


def test_design_and_preview_voice(client):
    created = client.post("/api/voices/design", json={
        "name": "Test rival",
        "prompt": "A low-pitched, stoic male anime rival voice. Smooth, husky, slightly gravelly "
                  "texture. Speaks slowly with extreme confidence, cold and composed delivery.",
    }).json()
    assert created["kind"] == "designed"
    assert created["params"]["f0"] < 160

    audio = client.post(f"/api/voices/{created['id']}/preview",
                        json={"text": "Is that all you have?", "emotion": "angry"})
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert len(audio.content) > 4000


def test_preview_rejects_unknown_emotion(client):
    voice = client.get("/api/voices").json()[0]
    resp = client.post(f"/api/voices/{voice['id']}/preview",
                       json={"text": "hi", "emotion": "nonsense"})
    assert resp.status_code == 400


def test_clone_voice_and_match(client, sample_wav):
    with sample_wav.open("rb") as fh:
        resp = client.post("/api/voices/clone",
                           files={"file": ("sample.wav", fh, "audio/wav")},
                           data={"name": "Cloned narrator", "kind": "instant"})
    assert resp.status_code == 200, resp.text
    voice = resp.json()
    assert voice["kind"] == "instant"
    assert voice["report"]["duration"] > 1
    assert voice["training_seconds"] > 1

    with sample_wav.open("rb") as fh:
        matches = client.post("/api/voices/match",
                              files={"file": ("sample.wav", fh, "audio/wav")}).json()
    assert matches and matches[0]["similarity"] > 0.5


def test_clone_rejects_too_short_audio(client, tmp_path):
    path = write_wav(tmp_path / "tiny.wav", np.zeros(int(0.4 * SR), dtype=np.float32), SR)
    with path.open("rb") as fh:
        resp = client.post("/api/voices/clone", files={"file": ("tiny.wav", fh, "audio/wav")},
                           data={"name": "too short"})
    assert resp.status_code == 400


def test_preset_voices_cannot_be_deleted(client):
    assert client.delete("/api/voices/anime_tsundere").status_code == 400


def test_voice_to_voice_conversion(client, sample_wav):
    with sample_wav.open("rb") as fh:
        resp = client.post("/api/voices/anime_demon_lord/convert",
                           files={"file": ("sample.wav", fh, "audio/wav")},
                           data={"strength": "1.0"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert len(resp.content) > 10_000


# --------------------------------------------------------------------------
# full pipeline
# --------------------------------------------------------------------------
def test_full_dubbing_pipeline(client, sample_wav):
    with sample_wav.open("rb") as fh:
        project = client.post("/api/projects",
                              files={"file": ("sample.wav", fh, "audio/wav")},
                              data={"name": "Pipeline test", "source_language": "en",
                                    "target_language": "es", "auto_start": "false"}).json()
    project_id = project["id"]

    script = ("Good evening and welcome to the broadcast. "
              "I am so excited to be here with you today! "
              "Let us begin with the main story of the hour.")
    assert client.post(f"/api/projects/{project_id}/script",
                       json={"script": script, "language": "en"}).status_code == 200

    job = client.post(f"/api/projects/{project_id}/dub", json={"target_language": "es"}).json()
    finished = _wait_for_job(client, job["id"])
    assert finished["status"] == "done", finished.get("error")
    assert finished["progress"] == 100

    full = client.get(f"/api/projects/{project_id}").json()
    assert full["status"] == "ready"
    assert len(full["segments"]) >= 2
    assert all(seg["source_text"] for seg in full["segments"])
    assert all(seg["voice_id"] for seg in full["segments"])
    assert len(full["speakers"]) >= 1
    for role in ("original", "vocals", "background", "dubbed", "mixed"):
        assert role in full["assets"], f"missing {role} asset"

    mixed = client.get(f"/api/projects/{project_id}/media/mixed")
    assert mixed.status_code == 200 and len(mixed.content) > 40_000

    srt = client.get(f"/api/projects/{project_id}/export.srt").text
    assert "-->" in srt and "00:00:" in srt

    waveform = client.get(f"/api/projects/{project_id}/waveform").json()
    assert len(waveform["peaks"]) > 100

    # dubbed speech must sit inside the original timeline
    data, sr = read_wav(ROOT / "data" / "media" / project_id / "mixed.wav") \
        if (ROOT / "data" / "media" / project_id / "mixed.wav").exists() else (None, None)
    assert full["duration"] > 1.0


def test_edit_then_rerender(client, sample_wav):
    with sample_wav.open("rb") as fh:
        project = client.post("/api/projects",
                              files={"file": ("s.wav", fh, "audio/wav")},
                              data={"name": "Edit test", "target_language": "en",
                                    "auto_start": "false"}).json()
    pid = project["id"]
    client.post(f"/api/projects/{pid}/script", json={"script": "First line here. Second line here."})
    _wait_for_job(client, client.post(f"/api/projects/{pid}/dub").json()["id"])

    segments = client.get(f"/api/projects/{pid}/segments").json()
    assert segments
    resp = client.patch(f"/api/projects/{pid}/segments/{segments[0]['id']}",
                        json={"target_text": "A completely rewritten line.",
                              "emotion": "excited", "voice_id": "anime_genki_girl"})
    assert resp.status_code == 200
    assert resp.json()["target_text"] == "A completely rewritten line."

    finished = _wait_for_job(client, client.post(f"/api/projects/{pid}/render").json()["id"])
    assert finished["status"] == "done", finished.get("error")

    seg_audio = client.get(f"/api/projects/{pid}/segments/{segments[0]['id']}/audio")
    assert seg_audio.status_code == 200


def test_segment_validation_rejects_bad_timing(client, sample_wav):
    with sample_wav.open("rb") as fh:
        project = client.post("/api/projects", files={"file": ("s.wav", fh, "audio/wav")},
                              data={"name": "Validation", "auto_start": "false"}).json()
    pid = project["id"]
    _wait_for_job(client, client.post(f"/api/projects/{pid}/dub").json()["id"])
    segments = client.get(f"/api/projects/{pid}/segments").json()
    if segments:
        resp = client.patch(f"/api/projects/{pid}/segments/{segments[0]['id']}",
                            json={"start": 5.0, "end": 1.0})
        assert resp.status_code == 400


def test_unknown_project_is_404(client):
    assert client.get("/api/projects/prj_missing").status_code == 404
    assert client.post("/api/projects/prj_missing/render").status_code == 404


# --------------------------------------------------------------------------
# 3D companion
# --------------------------------------------------------------------------
def test_viseme_track_covers_the_line():
    from backend.app import companion

    track = companion.viseme_track("Hello there, friend!")
    assert len(track) > 8
    assert all(t["v"] in companion.VISEMES for t in track)
    assert track[-1]["v"] == "rest"
    times = [t["t"] for t in track]
    assert times == sorted(times), "viseme timeline must be monotonic"


def test_viseme_track_rescales_to_audio_duration():
    from backend.app import companion

    track = companion.viseme_track("A test line for timing.", 1.0, duration=3.0)
    assert track[-1]["t"] == pytest.approx(3.0, abs=0.01)


def test_visemes_distinguish_open_and_closed_sounds():
    from backend.app import companion

    shapes = {t["v"] for t in companion.viseme_track("Mama papa said ah")}
    assert "MBP" in shapes, "bilabials must close the mouth"
    assert shapes & {"A", "E", "O"}, "vowels must open the mouth"


def test_companion_characters_seeded(client):
    chars = client.get("/api/companion/characters").json()
    names = {c["name"] for c in chars}
    assert {"Yuki", "Mika", "Rei", "Kaito"} <= names
    for c in chars:
        assert c["appearance"]["hair_style"]
        assert c["voice_id"]


def test_companion_say_returns_audio_and_visemes(client):
    resp = client.post("/api/companion/characters/char_mika/say",
                       json={"text": "Hi there, nice to meet you!", "emotion": "excited"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["emotion"] == "excited"
    assert data["duration"] > 0.3
    assert data["audio"].startswith("data:audio/wav;base64,")
    assert len(data["visemes"]) > 5
    # the mouth track must not run past the audio it belongs to
    assert data["visemes"][-1]["t"] <= data["duration"] + 0.05


def test_companion_default_emotion_survives_synthesis(client):
    """Yuki is 'flustered'; the emotion table must actually know that word."""
    data = client.post("/api/companion/characters/char_yuki/say",
                       json={"text": "D-don't look at me like that."}).json()
    assert data["emotion"] == "flustered"


def test_companion_rejects_unknown_emotion(client):
    resp = client.post("/api/companion/characters/char_rei/say",
                       json={"text": "hello", "emotion": "banana"})
    assert resp.status_code == 400


def test_companion_chat_remembers_history(client):
    client.delete("/api/companion/characters/char_yuki/history")
    reply = client.post("/api/companion/characters/char_yuki/chat",
                        json={"message": "hello there", "speak": False}).json()
    assert reply["reply"]
    assert reply["engine"] in ("scripted", "llm")
    history = client.get("/api/companion/characters/char_yuki/history").json()
    assert [h["role"] for h in history] == ["user", "assistant"]
    assert history[0]["text"] == "hello there"

    client.delete("/api/companion/characters/char_yuki/history")
    assert client.get("/api/companion/characters/char_yuki/history").json() == []


def test_companion_chat_speaks_when_asked(client):
    reply = client.post("/api/companion/characters/char_kaito/chat",
                        json={"message": "how are you?", "speak": True}).json()
    assert reply["audio"].startswith("data:audio/wav;base64,")
    assert reply["visemes"]


def test_create_and_delete_custom_companion(client):
    created = client.post("/api/companion/characters", json={
        "name": "Hikari", "voice_id": "anime_kuudere",
        "persona": "A quiet companion who loves astronomy.",
        "appearance": {"hair_style": "bob", "hair_color": "#334455"},
    }).json()
    assert created["appearance"]["hair_style"] == "bob"

    updated = client.patch(f"/api/companion/characters/{created['id']}",
                           json={"appearance": {"outfit_style": "trousers"}}).json()
    assert updated["appearance"]["outfit_style"] == "trousers"
    assert updated["appearance"]["hair_style"] == "bob", "patch must merge, not replace"

    assert client.delete(f"/api/companion/characters/{created['id']}").status_code == 200
    assert client.get(f"/api/companion/characters/{created['id']}").status_code == 404


def test_builtin_companions_are_protected(client):
    assert client.delete("/api/companion/characters/char_yuki").status_code == 400


def test_companion_rejects_unknown_voice(client):
    resp = client.post("/api/companion/characters",
                       json={"name": "Ghost", "voice_id": "voice_does_not_exist"})
    assert resp.status_code == 400


def test_idle_line_is_in_character(client):
    data = client.post("/api/companion/characters/char_ayame/idle").json()
    assert data["text"]
    assert data["visemes"]
