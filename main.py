import asyncio
import logging
import os
import uuid
import json
from httpx import AsyncClient
from contextlib import asynccontextmanager

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
model_name = "vosk-model-small-en-us-0.15"
logger = logging.getLogger("uvicorn.error")
relay = MediaRelay()
async_requests_client = AsyncClient()

pcs: set[RTCPeerConnection] = set()
peer_recorders: dict[RTCPeerConnection, MediaRecorder] = {}
peer_transcribe_tasks: dict[RTCPeerConnection, asyncio.Task] = {}
peer_data_channels: dict[RTCPeerConnection, object] = {}
peer_transcripts: dict[str, str] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    logger.info("Recordings directory ready: %s", RECORDINGS_DIR)
    global model
    model = vosk.Model(model_name)
    logger.info("Vosk model loaded")
    yield
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
        peer_data_channels.pop(pc, None)
        await pc.close()
    pcs.clear()
    peer_data_channels.clear()
    logger.info("Shutdown: all peer connections closed")


app = FastAPI(title="RTC Audio Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


async def fetch_chat_and_reply(pc_id: str, dc: object) -> None:
    """GET chatbot answer from upstream service and send JSON reply on DataChannel."""
    transcript = peer_transcripts.get(pc_id, "").strip()
    logger.info("stop_audio signal [%s]; transcript length=%d", pc_id, len(transcript))
    try:
        res = await async_requests_client.get(
            "http://localhost:8000/chat",
            params={"message": transcript},
        )
        chatbot_message = res.json()["message"]
        logger.info("Chat reply [%s]: %s", pc_id, chatbot_message[:200])
        send_chatbot_data_channel(dc, chatbot_message)
        peer_transcripts[pc_id] = ""
    except Exception:
        logger.exception("Chat fetch failed for session %s", pc_id)


def send_chatbot_data_channel(dc, chatbot_message: str) -> None:
    """Send JSON to the browser over the negotiated RTCDataChannel."""
    payload = json.dumps({"type": "chatbot_reply", "message": chatbot_message})
    try:
        if getattr(dc, "readyState", None) == "open":
            dc.send(payload)
            logger.info("Sent chatbot reply via DataChannel (%d chars)", len(chatbot_message))
        else:
            logger.warning(
                "DataChannel not open (readyState=%s); reply not sent",
                getattr(dc, "readyState", None),
            )
    except Exception as e:
        logger.warning("DataChannel send failed: %s", e)


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
                    peer_transcripts[pc_id] = peer_transcripts.get(pc_id, "") + text
                    logger.info("partial text [%s]: %s", pc_id, peer_transcripts[pc_id])
    except MediaStreamError:
        final_text = json.loads(recognizer.FinalResult()).get("text", "").strip()
        if final_text:
            peer_transcripts[pc_id] = peer_transcripts.get(pc_id, "") + final_text
    except Exception:
        logger.exception("Transcription task failed for %s", pc_id)
    finally:
        final_text = json.loads(recognizer.FinalResult()).get("text", "").strip()
        if final_text:
            prev = peer_transcripts.get(pc_id, "")
            peer_transcripts[pc_id] = (prev + (" " if prev else "") + final_text).strip()


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
            #recorder.addTrack(recorder_track)
            peer_transcribe_tasks[pc] = asyncio.create_task(transcribe_audio_track(stt_track, pc_id))


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
            peer_transcripts.pop(pc_id, None)
            pcs.discard(pc)
        track.on("ended", on_ended)

    pc.on("track", on_track)

    def on_ice_connection_state_change():
        logger.info(
            "ICE connection state %s for %s (peerConnection=%s)",
            pc.iceConnectionState,
            pc_id,
            pc.connectionState,
        )

    pc.on("iceconnectionstatechange", on_ice_connection_state_change)

    async def on_connectionstatechange():
        logger.info("Connection state %s for %s", pc.connectionState, pc_id)
        if pc.connectionState == "connected":
            dc = peer_data_channels.get(pc)
            logger.info(
                "Peer connected [%s]: data_channel readyState=%s sctp=%s",
                pc_id,
                getattr(dc, "readyState", None),
                pc.sctp,
            )
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
            peer_data_channels.pop(pc, None)
            peer_transcripts.pop(pc_id, None)
            pcs.discard(pc)
    pc.on("connectionstatechange", on_connectionstatechange)

    # Offerer (browser) creates `chatbot`; answerer receives it here. Register
    # before setRemoteDescription so the event is never missed.
    def on_datachannel(channel):
        logger.info(
            "Incoming DataChannel [%s] label=%s initial readyState=%s",
            pc_id,
            getattr(channel, "label", ""),
            getattr(channel, "readyState", ""),
        )
        peer_data_channels[pc] = channel

        def on_dc_open():
            logger.info(
                "DataChannel OPEN [%s] label=%s readyState=%s",
                pc_id,
                getattr(channel, "label", ""),
                getattr(channel, "readyState", ""),
            )

        def on_dc_close():
            logger.info("DataChannel CLOSED [%s]", pc_id)

        def on_dc_message(message):
            raw = (
                message.decode("utf-8")
                if isinstance(message, (bytes, bytearray))
                else message
            )
            logger.info("DataChannel message from client [%s]: %s", pc_id, raw)
            if not isinstance(raw, str):
                logger.warning("DataChannel non-text message from client [%s]", pc_id)
                return
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("DataChannel invalid JSON from client [%s]: %s", pc_id, raw[:200])
                return
            if payload.get("type") == "signal" and payload.get("action") == "stop_audio":
                asyncio.create_task(fetch_chat_and_reply(pc_id, channel))
                return
            logger.debug("DataChannel message from client [%s]: %s", pc_id, raw[:500])

        channel.on("open", on_dc_open)
        channel.on("close", on_dc_close)
        channel.on("message", on_dc_message)

    pc.on("datachannel", on_datachannel)

    try:
        await pc.setRemoteDescription(offer)

        await recorder.start()
        peer_recorders[pc] = recorder
        pcs.add(pc)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        _dc = peer_data_channels.get(pc)
        logger.info(
            "SDP answer set [%s]: signalingState=%s ice=%s conn=%s dc=%s sctp=%s",
            pc_id,
            pc.signalingState,
            pc.iceConnectionState,
            pc.connectionState,
            getattr(_dc, "readyState", "") if _dc else "(none yet)",
            pc.sctp,
        )

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
        peer_data_channels.pop(pc, None)
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8080)
