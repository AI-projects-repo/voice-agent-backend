# RTC Audio Agent Server

FastAPI + aiortc server that receives browser microphone audio over WebRTC, transcribes it with Vosk, fetches chat replies from an upstream API, and streams synthesized speech back to the client over a WebRTC DataChannel (Piper TTS).

## Features

- **WebRTC audio in** — Browser sends a microphone track; the server resamples to 16 kHz mono for speech recognition.
- **Streaming STT** — Vosk transcribes audio in real time; partial results accumulate per session.
- **Chat upstream** — On `resume_audio`, the server flushes STT, sends the transcript to `GET /chat`, and reads one sentence per line from the response stream.
- **TTS reply** — Each sentence is synthesized with Piper and sent as raw PCM over the DataChannel.
- **Interrupt handling** — `interrupt_audio` stops outbound audio promptly using a per-session reply epoch and task cancellation; the client receives `audio_abort`.
- **Optional recording** — Incoming audio can be written to WAV under `recordings/` (MediaRecorder is wired per peer).

## Architecture

```text
Browser                         Server                           Upstream
  │                               │                                  │
  │── WebRTC audio track ────────►│ Vosk STT → peer_transcripts      │
  │                               │                                  │
  │── DataChannel: resume_audio ─►│ flush STT → GET /chat ──────────►│ chat-api
  │                               │◄── sentence lines ───────────────│
  │◄── audio_start / PCM / end ───│ Piper TTS                        │
  │                               │                                  │
  │── DataChannel: interrupt ────►│ bump reply epoch, cancel task    │
  │◄── audio_abort ───────────────│ stop sending PCM                 │
```

## Requirements

- Python 3.10+
- FFmpeg (required by aiortc/PyAV)
- Vosk and Piper model files under `models/` (not committed; see [Models](#models))
- A reachable **chat-api** service when using voice replies (Docker Compose expects it on the `agent-network` as `chat-api:8000`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file and place STT/TTS models in `models/`.

## Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## Run with Docker

```bash
docker compose up --build
```

The service listens on port **8080**, mounts a `recordings` volume, and joins the external Docker network `agent-network` so it can reach `chat-api`.

## Configuration

Settings are loaded from environment variables via `app/core/config.py`:

| Variable | Description |
|----------|-------------|
| `RECORDINGS_DIR` | Directory for WAV recordings (e.g. `recordings`) |
| `TRANSCRIBER_MODEL_NAME` | Vosk model directory name under `models/` |
| `VOICE_MODEL_NAME` | Piper voice model file name under `models/` |
| `LOG_LEVEL` | Logging level (e.g. `INFO`, `DEBUG`) |
| `CHAT_UPSTREAM_READ_TIMEOUT` | httpx read timeout (seconds) for streaming `/chat` |

## Models

Download and install models into `models/` before starting the server:

- **Vosk** — e.g. [vosk-model-en-us-0.22](https://alphacephei.com/vosk/models)
- **Piper** — e.g. a `.onnx` voice plus its `.onnx.json` config from [rhasspy/piper](https://github.com/rhasspy/piper)

The paths used at startup are `models/<TRANSCRIBER_MODEL_NAME>` and `models/<VOICE_MODEL_NAME>`.

## HTTP API

### `POST /offer`

WebRTC signaling endpoint. The browser (offerer) must create the DataChannel before sending the offer.

**Request body:**

```json
{
  "sdp": "<SDP offer string>",
  "type": "offer"
}
```

**Response:**

```json
{
  "sdp": "<SDP answer string>",
  "type": "answer"
}
```

CORS is enabled for all origins.

## DataChannel protocol

The browser creates the DataChannel. All control messages are JSON text; synthesized audio uses JSON metadata plus binary PCM frames.

### Client → server (signals)

Send JSON text messages:

```json
{ "type": "signal", "action": "resume_audio" }
```

Starts (or restarts) the reply pipeline: flush STT, fetch chat for the current transcript, synthesize and stream audio.

```json
{ "type": "signal", "action": "interrupt_audio" }
```

Stops the current reply: increments the session reply epoch, cancels the reply task, and prevents further PCM from being sent for that reply.

### Server → client (audio stream)

| Message | Format | When |
|---------|--------|------|
| `audio_start` | JSON: `{ "type": "audio_start", "sample_rate", "channels", "sample_width" }` | First PCM chunk of a reply |
| PCM data | Binary (`int16` bytes) | One or more frames per synthesized sentence |
| `audio_end` | JSON: `{ "type": "audio_end" }` | Full reply completed successfully |
| `audio_abort` | JSON: `{ "type": "audio_abort" }` | Reply interrupted or superseded |

**Client responsibilities:**

- On `audio_start`, configure playback using the provided sample format.
- Append binary messages as PCM until `audio_end` or `audio_abort`.
- On `audio_abort`, stop playback and discard buffered PCM until the next `audio_start`.

## Reply interrupt handling

Each session has a **reply epoch** (`peer_reply_epoch`). When the client sends `resume_audio` or `interrupt_audio`, the epoch is bumped. The active reply task captures its epoch at start and checks it before each sentence and PCM chunk; a mismatch stops sending and emits `audio_abort`. The asyncio reply task is also cancelled so the upstream chat stream is closed.

If a reply completes without interruption, the transcript buffer for that session is cleared. On abort, the transcript is kept so a later `resume_audio` can retry with the same user text.

## Project layout

```text
app/
  main.py              FastAPI app, lifespan, /offer
  peer_connector.py    WebRTC peer setup, DataChannel signal handlers
  channel_messanger.py Chat fetch, Piper synthesis, DataChannel send
  transcriber.py       Vosk streaming transcription + STT flush
  core/config.py       Environment settings
models/                Vosk + Piper models (local, gitignored)
recordings/            WAV recordings (gitignored)
```

## Development notes

- Chat upstream URL is currently `http://chat-api:8000/chat` with query param `message=<transcript>`; it must stream one sentence per line.
- STT runs at **16 kHz** mono (`FRAME_RATE` in `main.py`).
- The server is the WebRTC **answerer**; the client must create the DataChannel and offer the microphone track.
