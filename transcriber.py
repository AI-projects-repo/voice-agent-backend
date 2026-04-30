import asyncio
import logging
import json
import vosk
import contextlib

from av import AudioResampler


from aiortc.mediastreams import MediaStreamError

logger = logging.getLogger("uvicorn.error")


def merge_recognizer_final(recognizer: vosk.KaldiRecognizer, pc_id: str, peer_transcripts: dict[str, str]) -> None:
    """Append Vosk FinalResult() text to peer_transcripts (must run on the recognizer owner task)."""
    final_text = json.loads(recognizer.FinalResult()).get("text", "").strip()
    if not final_text:
        return
    prev = peer_transcripts.get(pc_id, "")
    peer_transcripts[pc_id] = (prev + (" " if prev else "") + final_text).strip()
    logger.info("flush merged [%s]: %s", pc_id, peer_transcripts[pc_id])

async def transcribe_audio_track(
    track,
    pc_id: str,
    transcriber: vosk.Model,
    frame_rate: int,
    peer_stt_flush_request: dict[str, asyncio.Event],
    peer_stt_active: dict[str, bool],
    peer_stt_flush_complete: dict[str, asyncio.Future],
    peer_transcripts: dict[str, str]
) -> None:
    """
    Consume aiortc audio frames and feed Vosk with 16kHz mono s16 PCM bytes.
    """
    if transcriber is None:
        logger.warning("Vosk model not loaded; skipping transcription for %s", pc_id)
        return

    recognizer = vosk.KaldiRecognizer(transcriber, frame_rate)
    recognizer.SetWords(True)
    resampler = AudioResampler(format="s16", layout="mono", rate=frame_rate)
    flush_ev = peer_stt_flush_request[pc_id]
    peer_stt_active[pc_id] = True
    logger.warning("Transcription task started for %s", pc_id)

    try:
        while True:
            recv_t = asyncio.create_task(track.recv())
            flush_t = asyncio.create_task(flush_ev.wait())
            try:
                done, pending = await asyncio.wait(
                    {recv_t, flush_t},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                for t in pending:
                    with contextlib.suppress(asyncio.CancelledError):
                        await t

                if flush_t in done:
                    with contextlib.suppress(asyncio.CancelledError, MediaStreamError):
                        await recv_t
                    merge_recognizer_final(recognizer, pc_id, peer_transcripts)
                    fut = peer_stt_flush_complete.pop(pc_id, None)
                    if fut is not None and not fut.done():
                        fut.set_result(None)
                    flush_ev.clear()
                    continue

                try:
                    frame = recv_t.result()
                except MediaStreamError:
                    raise
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
            finally:
                for t in (recv_t, flush_t):
                    if not t.done():
                        t.cancel()
                    with contextlib.suppress(asyncio.CancelledError, MediaStreamError):
                        await t
    except MediaStreamError:
        logger.error("MediaStreamError for %s", pc_id)
    except Exception:
        logger.error("Transcription task failed for %s", pc_id)
    finally:
        merge_recognizer_final(recognizer, pc_id, peer_transcripts)
        peer_stt_active.pop(pc_id, None)