import asyncio
import logging
import os
import uuid
import json

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder, MediaRelay
from aiortc.mediastreams import MediaStreamError
from av import AudioResampler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import vosk

RECORDINGS_DIR = "recordings"
FRAME_RATE = 16000
model = None
logger = logging.getLogger("uvicorn.error")
relay = MediaRelay()

pcs: set[RTCPeerConnection] = set()
peer_recorders: dict[RTCPeerConnection, MediaRecorder] = {}
peer_transcribe_tasks: dict[RTCPeerConnection, asyncio.Task] = {}

app = FastAPI(title="RTC Audio Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    logger.info("Recordings directory ready: %s", RECORDINGS_DIR)
    global model
    model = vosk.Model("vosk-model-small-en-us-0.15")
    logger.info("Vosk model loaded")


async def transcribe_audio_track(track, pc_id: str):
    """
    Consume aiortc audio frames and feed Vosk with 16kHz mono s16 PCM bytes.
    """
    if model is None:
        logger.warning("Vosk model not loaded; skipping transcription for %s", pc_id)
        return

    recognizer = vosk.KaldiRecognizer(model, FRAME_RATE)
    recognizer.SetWords(True)
    resampler = AudioResampler(format="s16", layout="mono", rate=FRAME_RATE)
    logger.warning("Transcription task started for %s", pc_id)

    try:
        while True:
            frame = await track.recv()
            resampled_frames = resampler.resample(frame)
            if resampled_frames is None:
                continue
            if not isinstance(resampled_frames, list):
                resampled_frames = [resampled_frames]

            for audio_frame in resampled_frames:
                pcm_bytes = audio_frame.to_ndarray().tobytes()
                if recognizer.AcceptWaveform(pcm_bytes):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        logger.warning("Recognized (%s): %s", pc_id, text)
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    if partial:
                        logger.warning("Partial (%s): %s", pc_id, partial)
    except MediaStreamError:
        final_text = json.loads(recognizer.FinalResult()).get("text", "").strip()
        if final_text:
            logger.warning("Final (%s): %s", pc_id, final_text)
        else:
            logger.warning("Final (%s): <empty>", pc_id)
    except Exception:
        logger.exception("Transcription task failed for %s", pc_id)


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    sdp = params.get("sdp")
    typ = params.get("type", "offer")
    if not sdp or typ != "offer":
        return JSONResponse(
            status_code=400,
            content={"error": "Missing or invalid body: expected { \"sdp\": string, \"type\": \"offer\" }"},
        )

    offer = RTCSessionDescription(sdp=sdp, type=typ)
    pc = RTCPeerConnection()
    pc_id = str(uuid.uuid4())
    record_path = os.path.join(RECORDINGS_DIR, f"audio_{pc_id}.wav")
    recorder = MediaRecorder(record_path)

    def on_track(track):
        if track.kind == "audio":
            # One source track cannot be consumed by multiple readers directly.
            # Relay creates independent proxy tracks for recorder and transcriber.
            recorder_track = relay.subscribe(track)
            stt_track = relay.subscribe(track)
            recorder.addTrack(recorder_track)
            logger.info("Audio track added for %s", pc_id)
            peer_transcribe_tasks[pc] = asyncio.create_task(transcribe_audio_track(stt_track, pc_id))


        @track.on("ended")
        async def on_ended():
            logger.info("Track ended for %s", pc_id)
            try:
                await recorder.stop()
            except Exception as e:
                logger.warning("Recorder stop error: %s", e)
            task = peer_transcribe_tasks.pop(pc, None)
            if task:
                task.cancel()
            peer_recorders.pop(pc, None)
            pcs.discard(pc)

    @pc.on("track")
    def _on_track(track):
        on_track(track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state %s for %s", pc.connectionState, pc_id)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            rec = peer_recorders.get(pc)
            if rec:
                try:
                    await rec.stop()
                except Exception as e:
                    logger.warning("Recorder stop on state change: %s", e)
                peer_recorders.pop(pc, None)
            task = peer_transcribe_tasks.pop(pc, None)
            if task:
                task.cancel()
            pcs.discard(pc)

    try:
        await pc.setRemoteDescription(offer)
        await recorder.start()
        peer_recorders[pc] = recorder
        pcs.add(pc)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return JSONResponse(
            content={
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        )
    except Exception as e:
        logger.exception("Offer handling failed: %s", e)
        await recorder.stop()
        pcs.discard(pc)
        peer_recorders.pop(pc, None)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.on_event("shutdown")
async def shutdown():
    for pc in list(pcs):
        rec = peer_recorders.get(pc)
        if rec:
            try:
                await rec.stop()
            except Exception as e:
                logger.warning("Recorder stop on shutdown: %s", e)
            peer_recorders.pop(pc, None)
        task = peer_transcribe_tasks.pop(pc, None)
        if task:
            task.cancel()
        await pc.close()
    pcs.clear()
    logger.info("Shutdown: all peer connections closed")


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8080)
