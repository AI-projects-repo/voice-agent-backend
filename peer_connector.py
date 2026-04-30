import asyncio
import logging
import uuid
import os
import json
import vosk

from typing import Tuple
from transcriber import transcribe_audio_track
from aiortc.contrib.media import MediaRelay, MediaRecorder
from aiortc import RTCPeerConnection

from channel_messanger import fetch_chat_and_reply

relay = MediaRelay()
RECORDINGS_DIR = "recordings"

logger = logging.getLogger("uvicorn.error")

def get_peer_connection(
    transcriber_model: vosk.Model,
    frame_rate: int,
    peer_data_channels: dict[RTCPeerConnection, object],
    peer_recorders: dict[RTCPeerConnection, MediaRecorder],
    peer_transcribe_tasks: dict[RTCPeerConnection, asyncio.Task],
    peer_transcripts: dict[str, str], peer_stt_flush_request: dict[str, asyncio.Event],
    peer_stt_active: dict[str, bool], peer_stt_flush_complete: dict[str, asyncio.Future],
    pcs: set[RTCPeerConnection],
    recordings_dir: str
) -> Tuple[RTCPeerConnection, str]:
    # create a new peer connection and assign a unique ID
    pc = RTCPeerConnection()
    pc_id = str(uuid.uuid4())
    record_path = os.path.join(recordings_dir, f"audio_{pc_id}.wav")
    recorder = MediaRecorder(record_path)

    def on_ice_connection_state_change():
        logger.info(
            "ICE connection state %s for %s (peerConnection=%s)",
            pc.iceConnectionState,
            pc_id,
            pc.connectionState,
        )

    def on_track(track) -> None: 
        if track.kind == "audio":
            # One source track cannot be consumed by multiple readers directly.
            # Relay creates independent proxy tracks for recorder and transcriber.
            recorder_track = relay.subscribe(track)
            stt_track = relay.subscribe(track)
            #recorder.addTrack(recorder_track)
            peer_transcribe_tasks[pc] = asyncio.create_task(
                transcribe_audio_track(
                    stt_track,
                    pc_id,
                    transcriber_model,
                    frame_rate,
                    peer_stt_flush_request,
                    peer_stt_active,
                    peer_stt_flush_complete,
                    peer_transcripts
                )
            )


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
            peer_stt_flush_request.pop(pc_id, None)
            peer_stt_active.pop(pc_id, None)
            _f = peer_stt_flush_complete.pop(pc_id, None)
            if _f is not None and not _f.done():
                _f.cancel()
            pcs.discard(pc)
        track.on("ended", on_ended)
    
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
            peer_stt_flush_request.pop(pc_id, None)
            peer_stt_active.pop(pc_id, None)
            _f = peer_stt_flush_complete.pop(pc_id, None)
            if _f is not None and not _f.done():
                _f.cancel()
            pcs.discard(pc)

    # Offerer (browser) creates data channel; answerer receives it here. Register
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
                asyncio.create_task(
                    fetch_chat_and_reply(
                        pc_id,
                        channel,
                        peer_stt_flush_request,
                        peer_stt_active,
                        peer_stt_flush_complete,
                        peer_transcripts,
                    )
                )
                return
            logger.debug("DataChannel message from client [%s]: %s", pc_id, raw[:500])

        channel.on("open", on_dc_open)
        channel.on("close", on_dc_close)
        channel.on("message", on_dc_message)

    pc.on("track", on_track)
    pc.on("iceconnectionstatechange", on_ice_connection_state_change)
    pc.on("connectionstatechange", on_connectionstatechange)
    pc.on("datachannel", on_datachannel)
    return pc, pc_id, recorder