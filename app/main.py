import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

from app.core.config import Settings
settings = Settings()


def _configure_logging() -> None:
    """Ensure app loggers (getLogger(__name__)) emit INFO when run via `uvicorn main:app` (no __main__ block)."""
    level_name = settings.log_level
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root.addHandler(handler)


_configure_logging()

import vosk
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from app.peer_connector import get_peer_connection
from piper import PiperVoice
from httpx import AsyncClient



FRAME_RATE = 16000
transcriber_model = None
voice_model = None
async_requests_client = None
logger = logging.getLogger(__name__)

pcs: set[RTCPeerConnection] = set()
peer_recorders: Dict[RTCPeerConnection, MediaRecorder] = {}
peer_transcribe_tasks: Dict[RTCPeerConnection, asyncio.Task] = {}
peer_data_channels: Dict[RTCPeerConnection, object] = {}
peer_transcripts: Dict[str, str] = {}
# STT flush: DataChannel stop_audio sets the Event; transcriber calls FinalResult then completes the Future.
peer_stt_flush_request: Dict[str, asyncio.Event] = {}
peer_stt_flush_complete: Dict[str, asyncio.Future] = {}
peer_stt_active: Dict[str, bool] = {}
peer_reply_tasks: Dict[str, asyncio.Task] = {}
peer_reply_epoch: Dict[str, int] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.recordings_dir, exist_ok=True)
    logger.info("Recordings directory ready: %s", settings.recordings_dir)
    global transcriber_model
    transcriber_model = vosk.Model(os.path.join("models", settings.transcriber_model_name))
    logger.info("Vosk model loaded")
    global voice_model
    voice_model = PiperVoice.load(os.path.join("models", settings.voice_model_name))
    logger.info("Piper voice model loaded")
    global async_requests_client
    async_requests_client = AsyncClient()
    logger.info("Async requests client initialized")
    yield
    for task in peer_reply_tasks.values():
        if task:
            task.cancel()
    peer_reply_tasks.clear()
    logger.info("Peer reply tasks cleared")
    await async_requests_client.aclose()
    logger.info("Async requests client closed")
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


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    sdp = params.get("sdp")
    type = params.get("type", "offer")
    if not sdp or type != "offer":
        return JSONResponse(
            status_code=400,
            content={"error": "Missing or invalid body: expected { \"sdp\": string, \"type\": \"offer\" }"},
        )

    offer = RTCSessionDescription(sdp=sdp, type=type)
    pc, pc_id, recorder = get_peer_connection(
        transcriber_model,
        voice_model,
        async_requests_client,
        FRAME_RATE,
        peer_data_channels,
        peer_recorders,
        peer_transcribe_tasks,
        peer_transcripts,
        peer_stt_flush_request,
        peer_stt_active,
        peer_stt_flush_complete,
        pcs,
        settings.recordings_dir,
        settings.chat_upstream_read_timeout,
        peer_reply_tasks,
        peer_reply_epoch
    )
    peer_stt_flush_request[pc_id] = asyncio.Event()

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
        peer_stt_flush_request.pop(pc_id, None)
        peer_stt_active.pop(pc_id, None)
        task = peer_reply_tasks.pop(pc_id, None)
        if task:
            task.cancel()
        logger.info("Peer reply task cancelled")
        _f = peer_stt_flush_complete.pop(pc_id, None)
        if _f is not None and not _f.done():
            _f.cancel()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8080)
