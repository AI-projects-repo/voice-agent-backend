# RTC Audio Server

FastAPI + aiortc server that receives WebRTC audio from browsers and saves it to WAV files.

## Setup

- Python 3.10+
- FFmpeg installed on the system (required by aiortc/PyAV)

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

Open http://localhost:8080 in a browser, allow microphone access, click **Start**. Audio is streamed over WebRTC and saved under `recordings/` as `audio_<uuid>.wav`. Click **Stop** to end the session.

## API

- **GET /** – Serves the test client (HTML).
- **POST /offer** – Body: `{ "sdp": string, "type": "offer" }`. Returns SDP answer `{ "sdp", "type" }` and records incoming audio to a new file in `recordings/`.
