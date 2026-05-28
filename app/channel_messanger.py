import asyncio
import logging
import json

from httpx import AsyncClient, Timeout

logger = logging.getLogger(__name__)


def send_chatbot_data_channel(dc, voice_model, chatbot_message: str, first_chunk: bool = False) -> None:
    """Send audio metadata as JSON and PCM chunks as bytes over the DataChannel."""
    try:
        if getattr(dc, "readyState", None) == "open":
            for chunk in voice_model.synthesize(chatbot_message):
                if first_chunk:
                    # send chunks metadata
                    dc.send(
                        json.dumps({
                            "type": "audio_start",
                            "sample_rate": chunk.sample_rate,
                            "channels": chunk.sample_channels,
                            "sample_width": chunk.sample_width
                        })
                    )
                    first_chunk = False
                dc.send(chunk.audio_int16_bytes)
            logger.info("Sent chatbot reply via DataChannel (%d chars)", len(chatbot_message))
        else:
            logger.warning(
                "DataChannel not open (readyState=%s); reply not sent",
                getattr(dc, "readyState", None),
            )
    except Exception as e:
        logger.warning("DataChannel send failed: %s", e)

async def fetch_chat_and_reply(
    pc_id: str,
    dc: object,
    voice_model,
    async_requests_client: AsyncClient,
    peer_stt_flush_request: dict[str, asyncio.Event],
    peer_stt_active: dict[str, bool],
    peer_stt_flush_complete: dict[str, asyncio.Future],
    peer_transcripts: dict[str, str],
    chat_upstream_read_timeout: float,
    peer_reply_tasks: dict[str, asyncio.Task],
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

        transcript = peer_transcripts.get(pc_id, "").strip()
        logger.info("stop_audio signal [%s]; transcript length=%d", pc_id, len(transcript))
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

                # /chat streams one sentence per line, so synthesize each sentence once.
                async for sentence in res.aiter_lines():
                    sentence = sentence.strip()
                    if sentence:
                        send_chatbot_data_channel(dc, voice_model, sentence, first_chunk)
                        first_chunk = False
                if getattr(dc, "readyState", None) == "open":
                    dc.send(json.dumps({"type": "audio_end"}))

            peer_transcripts[pc_id] = ""
        except Exception:
            logger.exception("Chat fetch failed for session %s", pc_id)
    except asyncio.CancelledError:
        logger.info("Chat fetch cancelled for session %s", pc_id)
    finally:
        current_task = asyncio.current_task()
        if current_task and peer_reply_tasks.get(pc_id, None) is current_task:
            peer_reply_tasks.pop(pc_id, None)