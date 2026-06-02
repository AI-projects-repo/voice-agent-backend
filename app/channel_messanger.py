import asyncio
import logging
import json

from collections.abc import Mapping
from typing import Dict, Any
from httpx import AsyncClient, Timeout

from piper import PiperVoice
from aiortc import RTCDataChannel

logger = logging.getLogger(__name__)

def is_reply_active(pc_id: str, peer_reply_epoch: Dict[str, int], epoch: int) -> bool:
    return peer_reply_epoch.get(pc_id) == epoch

def send_message_to_data_channel(data_channel: RTCDataChannel, message: Mapping[str, Any] | bytes) -> None:
    try:
        if getattr(data_channel, "readyState", None) == "open":
            if isinstance(message, bytes):
                data_channel.send(message)
            else:
                data_channel.send(json.dumps(message))
        else:
            logger.warning("DataChannel not open (readyState=%s); message not sent", getattr(data_channel, "readyState", None))
    except Exception as e:
        logger.warning("DataChannel send failed: %s", e)


def send_chatbot_data_channel(
    data_channel: RTCDataChannel,
    voice_model: PiperVoice,
    chatbot_message: str,
    first_chunk: bool = False,
    pc_id: str = None,
    peer_reply_epoch: Dict[str, int] = None,
    epoch: int = 0,
) -> Dict[str, str]:
    """Send audio metadata as JSON and PCM chunks as bytes over the DataChannel."""
    for chunk in voice_model.synthesize(chatbot_message):
        if not is_reply_active(pc_id, peer_reply_epoch, epoch):
            return {"type": "audio_abort"}
        if first_chunk:
            # send chunks metadata
            send_message_to_data_channel(
                data_channel,
                {
                    "type": "audio_start",
                    "sample_rate": chunk.sample_rate,
                    "channels": chunk.sample_channels,
                    "sample_width": chunk.sample_width
                }
            )
            first_chunk = False
        send_message_to_data_channel(data_channel, chunk.audio_int16_bytes)
    logger.info("Sent chatbot reply via DataChannel (%d chars)", len(chatbot_message))
    return {"type": "sentence_done"}


async def fetch_chat_and_reply(
    pc_id: str,
    data_channel: RTCDataChannel,
    voice_model,
    async_requests_client: AsyncClient,
    peer_stt_flush_request: dict[str, asyncio.Event],
    peer_stt_active: dict[str, bool],
    peer_stt_flush_complete: dict[str, asyncio.Future],
    peer_transcripts: dict[str, str],
    chat_upstream_read_timeout: float,
    peer_reply_tasks: dict[str, asyncio.Task],
    peer_reply_epoch: Dict[str, int],
    epoch: int = 0,
) -> None:
    """GET sentence chunks from upstream chat and stream synthesized audio."""
    try:
        ev = peer_stt_flush_request.get(pc_id)
        if ev is not None and peer_stt_active.get(pc_id):
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            peer_stt_flush_complete[pc_id] = fut
            ev.set()
            try:
                await asyncio.wait_for(fut, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("STT flush timed out for session %s", pc_id)
            finally:
                peer_stt_flush_complete.pop(pc_id, None)

        if not is_reply_active(pc_id, peer_reply_epoch, epoch):
            logger.info("Audio abort for session %s", pc_id)
            send_message_to_data_channel(data_channel, {"type": "audio_abort"})
            return

        transcript = peer_transcripts.get(pc_id, "").strip()
        logger.info("resume_audio signal [%s]; transcript length=%d", pc_id, len(transcript))
        CHAT_UPSTREAM_TIMEOUT = Timeout(
            connect=10.0,
            read=chat_upstream_read_timeout,
            write=10.0,
            pool=10.0,
        )
        try:
            async with async_requests_client.stream(
                "GET",
                "http://chat-api:8000/chat",
                params={"message": transcript},
                timeout=CHAT_UPSTREAM_TIMEOUT,
            ) as res:
                res.raise_for_status()

                first_chunk = True
                reset_transcript = False
                audio_abort = False
                # /chat streams one sentence per line, so synthesize each sentence once.
                async for sentence in res.aiter_lines():
                    if not is_reply_active(pc_id, peer_reply_epoch, epoch):
                        logger.info("Audio abort for session %s", pc_id)
                        send_message_to_data_channel(data_channel, {"type": "audio_abort"})
                        audio_abort = True
                        break
                    sentence = sentence.strip()
                    if sentence:
                        result = send_chatbot_data_channel(
                            data_channel,
                            voice_model,
                            sentence,
                            first_chunk,
                            pc_id,
                            peer_reply_epoch,
                            epoch,
                        )
                        if result.get("type") == "audio_abort" or result.get("type") == "error":
                            logger.info("Audio abort for session %s", pc_id)
                            send_message_to_data_channel(data_channel, {"type": "audio_abort"})
                            audio_abort = True
                            break
                        first_chunk = False
                if not audio_abort:
                    send_message_to_data_channel(data_channel, {"type": "audio_end"})
                    reset_transcript = True

            if reset_transcript:
                peer_transcripts[pc_id] = ""
        except Exception:
            logger.exception("Chat fetch failed for session %s", pc_id)
    except asyncio.CancelledError:
        logger.info("Chat fetch cancelled for session %s", pc_id)
    finally:
        current_task = asyncio.current_task()
        if current_task and peer_reply_tasks.get(pc_id, None) is current_task:
            peer_reply_tasks.pop(pc_id, None)