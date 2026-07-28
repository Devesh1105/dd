# AI Dubbing & Voice Platform

A complete, **local-first** dubbing studio: upload media, and it separates the
background music, detects who spoke when, transcribes, translates with length
control, clones or designs the voices, synthesises the dub, fits every line
back into its original timestamp, and re-mixes it under the original score.

It runs on a laptop with **no GPU, no API keys and no model downloads** — every
stage has a working offline implementation. Install Whisper, XTTS, Demucs or
plug in ElevenLabs/PlayHT/Cartesia and the same pipeline transparently upgrades.

```
[source media] ─► ingest ─► stem split ─► VAD + diarization ─► ASR
                                                                │
 [dubbed output] ◄─ render ◄─ mix ◄─ time-fit ◄─ TTS ◄─ translation
```

---

## Quick start

```bash
./run.sh                     # creates .venv, installs deps, starts the server
```

Then open <http://127.0.0.1:8000>.

A demo clip (two speakers over a music bed) is generated at `data/sample.wav`
on first run — upload it, paste the printed transcript into **Known script**,
and press **Create & start dubbing**.

Manual start:

```bash
python -m pip install -r requirements.txt
python scripts/make_sample.py data/sample.wav      # optional demo clip
python -m uvicorn backend.app.main:app --port 8000
```

Tests:

```bash
python -m pytest tests -q                          # 25 tests, ~15s, no network
```

> **ffmpeg is optional.** Without it the platform works on WAV files only.
> Install it to accept MP3/MP4/MOV and to write dubbed video.

---

## What it does

### Voices

| Kind | How it is made | Where |
|---|---|---|
| **Preset / studio** | 18 built-in voices — broadcast, narration, gaming, and the full anime archetype set | Voice bank |
| **Instant (zero-shot)** | 5–30s of reference audio → speaker embedding | `POST /api/voices/clone` |
| **Professional** | Long-form studio audio, same path with a longer reference | `kind=professional` |
| **Cross-lingual** | A cloned voice speaking a language the source speaker never spoke | automatic during dubbing |
| **Designed** | A natural-language prompt → synthesis parameters | `POST /api/voices/design` |
| **Voice-to-voice (RVC-style)** | Re-voice a recording, keeping its exact timing and delivery | `POST /api/voices/{id}/convert` |

**Voice Design** parses prompts using the five-part structure the UI documents
— age & gender, pitch & texture, pacing & rhythm, emotion & attitude, accent &
style:

```json
{
  "name": "Rival",
  "prompt": "A low-pitched, stoic male anime rival voice. Smooth, husky, slightly gravelly texture. Speaks slowly and with extreme confidence, cold and composed delivery."
}
```

Thirteen emotion modifiers (`shouting`, `whisper`, `laughing`, `crying`,
`sarcastic`, `menacing`, …) apply per line without re-creating the voice.

### Pipeline features

- **Stem separation** — vocals vs. background, so the original score survives
  the dub and is re-mixed underneath with side-chain ducking.
- **Diarization** — speaker embeddings + agglomerative clustering assign
  "who spoke when", then each speaker is cast to a voice (auto-cloned by default).
- **Length-aware translation** — every line gets a character budget derived
  from its slot duration and the target language's speaking rate, so the
  translation is asked to *fit* rather than being clipped afterwards.
- **Time-fit / pacing** — WSOLA time-scaling compresses or stretches each
  rendered line into its original timestamp without changing pitch, within
  studio-realistic limits (padding rather than mangling beyond them).
- **Chunked parallelism** — long files are split into ≤12s utterances and
  fanned out across a thread pool.
- **Live progress** — WebSocket (with SSE fallback) reports
  `Step 3/9: Isolating background music…` per job.
- **Editor** — canvas timeline with per-speaker lanes, transcript/translation
  matrix with per-line fit warnings, per-line voice and emotion, mix controls,
  and SRT/JSON/WAV export.

---

## Architecture

```
frontend/                 zero-build SPA (canvas timeline, transcript matrix, WS progress)
backend/app/
  main.py                 FastAPI app, static mount, /api/system capability report
  config.py               env-driven settings
  db.py                   SQLite schema + vector search over voice embeddings
  voicebank.py            presets, cloning, reference hygiene grading, lookup
  voice_design.py         prompt → VoiceParams, emotion deltas, archetype library
  audio/
    wavio.py              stdlib WAV I/O + resampling
    dsp.py                STFT, VAD, embeddings, WSOLA, separation, mix bus
    synth.py              offline formant synthesiser + voice conversion
  providers/              swappable engines behind one interface per capability
  pipeline/orchestrator.py the nine pipeline steps and their job handlers
  core/queue.py           durable async job queue (the local Celery/Temporal stand-in)
  core/events.py          pub/sub feeding WebSocket + SSE
  api/                    projects, voices, jobs routers
```

### Design decisions worth knowing

**One embedding space for every voice.** Cloned voices are embedded from audio;
designed voices are embedded by *synthesising a fixed probe line and running the
same encoder*. Deriving preset vectors analytically would have put them in a
different space, making "find the preset closest to this speaker" meaningless.

**Offline stages never fabricate content.** With no ASR model installed, the
offline provider emits correctly-timed *empty* segments rather than inventing
words, and the UI asks for a script. With no MT model, text passes through
untranslated and un-truncated — silently cutting a line's ending would destroy
meaning the editor cannot recover. `/api/system` reports exactly which
capabilities are real in this installation.

**Background separation uses stationary-bed subtraction, not HPSS.** Classic
harmonic/percussive separation was tried first and performed badly: speech is
both harmonic and transient, so most of the voice landed in the background
stem. Per-bin median over a ~1.6s window models the music bed far better.

**The offline synthesiser is voice-conditioned, not a beep generator.** Pitch,
vocal-tract length, rasp, breathiness, growl and pacing all come from
`VoiceParams`, which are read back out of a clone's embedding — so a cloned
voice audibly tracks its reference. It is intelligible and useful for building
and testing the pipeline; install a neural TTS provider for production audio.

---

## Configuring engines

Copy `.env.example` to `.env`. Every provider is `auto` by default: the best
locally-installed engine wins, otherwise the offline one runs.

| Capability | Providers (best first) | Offline fallback |
|---|---|---|
| ASR | `faster_whisper`, `whisper`, `openai_api` | `offline` (VAD segments + script alignment) |
| Translation | `llm` (any OpenAI-compatible endpoint), `argos` | `passthrough` |
| TTS | `xtts`, `f5`, `chatterbox`, `piper`, `elevenlabs`, `playht`, `cartesia` | `local_formant` |
| Voice conversion | `rvc` | `local_morph` |
| Separation | `demucs` | `spectral` |
| Lip-sync | `wav2lip` | `none` (audio muxed onto the original video) |

```bash
pip install faster-whisper                 # real transcription
pip install TTS                            # Coqui XTTS v2 cross-lingual cloning
pip install demucs                         # proper stem separation
export ELEVENLABS_API_KEY=...              # or a commercial provider
```

See `requirements-optional.txt`. Check what is live at `GET /api/system` or the
**System** tab.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/projects` | upload media, create a project |
| `POST` | `/api/projects/{id}/script` | attach a known transcript (skips ASR) |
| `POST` | `/api/projects/{id}/dub` | run the full pipeline |
| `POST` | `/api/projects/{id}/render` | re-render after edits (skips ASR/MT) |
| `GET` | `/api/projects/{id}` | project + segments + speakers + assets + jobs |
| `PATCH` | `/api/projects/{id}/segments/{sid}` | edit a line's text, timing, voice, emotion |
| `POST` | `/api/projects/{id}/speakers/{spk}/voice/{vid}` | re-cast a speaker |
| `GET` | `/api/projects/{id}/media/{role}` | `original` · `vocals` · `background` · `dubbed` · `mixed` · `output_video` |
| `GET` | `/api/projects/{id}/export.srt` | subtitles |
| `GET` | `/api/voices` · `/archetypes` | voice bank and the prompt library |
| `POST` | `/api/voices/design` · `/clone` · `/match` · `/{id}/preview` · `/{id}/convert` | voice operations |
| `WS` | `/ws/projects/{id}` · `/ws/jobs/{id}` | live progress |
| `GET` | `/api/jobs/{id}/events` | SSE fallback |

Interactive docs at `/docs`.

---

## Scaling beyond one machine

The local components map one-to-one onto their production equivalents:

- `core/queue.py` → Celery / Temporal (keep `JobContext`, swap the executor).
- `core/events.py` → Redis pub/sub (`Broker.publish` is the only change).
- `db.py` voice table → PostgreSQL + pgvector or Qdrant
  (`search_voices_by_embedding` keeps its signature).
- Project media directories → S3 / R2.
- Provider adapters already exist for the GPU engines; point `DUB_TTS` at them
  and run the workers on A10G/L4 nodes.

## Limitations

- The offline ASR cannot transcribe — supply a script or install
  `faster-whisper`. The UI and `/api/system` say so explicitly.
- The offline translator does not translate; it passes text through.
- The built-in synthesiser is intelligible but clearly synthetic. It exists so
  the pipeline is testable end-to-end without downloads.
- Script alignment distributes sentences across detected spans by duration,
  which is an approximation of forced alignment.
- Lip-sync requires a Wav2Lip checkout; without it audio is muxed unchanged.
