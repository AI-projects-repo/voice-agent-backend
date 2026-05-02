import asyncio
import logging
import json
import os

from httpx import AsyncClient, Timeout

logger = logging.getLogger(__name__)

# Upstream /chat may stream or block on LLMs; default read matches long generation.
CHAT_UPSTREAM_READ_TIMEOUT = float(os.environ.get("CHAT_UPSTREAM_READ_TIMEOUT", "120"))
CHAT_UPSTREAM_TIMEOUT = Timeout(
    connect=10.0,
    read=CHAT_UPSTREAM_READ_TIMEOUT,
    write=10.0,
    pool=10.0,
)

async_requests_client = AsyncClient()


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

async def fetch_chat_and_reply(
    pc_id: str,
    dc: object,
    peer_stt_flush_request: dict[str, asyncio.Event],
    peer_stt_active: dict[str, bool],
    peer_stt_flush_complete: dict[str, asyncio.Future],
    peer_transcripts: dict[str, str]
) -> None:
    """GET chatbot answer from upstream service and send JSON reply on DataChannel."""
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
    try:
        res = await async_requests_client.get(
            "http://chat-api:8000/chat",
            params={"message": transcript},
            timeout=CHAT_UPSTREAM_TIMEOUT,
        )
        chatbot_message = res.json()["message"]
        logger.info("Chat reply [%s]: %s", pc_id, chatbot_message[:200])
        send_chatbot_data_channel(dc, chatbot_message)
        peer_transcripts[pc_id] = ""
    except Exception:
        logger.exception("Chat fetch failed for session %s", pc_id)