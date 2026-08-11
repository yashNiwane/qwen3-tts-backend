"""
Voice Cloning endpoint
======================
POST /api/v1/voice-clone
  — Clone a voice from a reference audio file.

POST /api/v1/voice-clone/batch
  — Batch clone with multiple reference audios.

POST /api/v1/voice-clone/stream
  — Stream generated audio chunks back to the client.
"""

import asyncio
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse

from app.core.model_manager import model_manager
from app.core.audio_utils import save_audio, wav_to_bytes
from app.schemas.common import AudioResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------- #
# Single voice clone
# --------------------------------------------------------------------------- #
@router.post(
    "/voice-clone",
    response_model=AudioResponse,
    summary="Clone a voice from reference audio",
)
async def voice_clone(
    text: str = Form(..., description="Text to synthesise in the cloned voice."),
    language: str = Form("English", description="Target language for synthesis."),
    ref_text: str = Form("", description="Transcript of the reference audio (improves quality)."),
    temperature: float = Form(0.7),
    top_p: float = Form(0.8),
    speed: float = Form(1.0),
    ref_audio: UploadFile = File(..., description="Reference WAV/MP3 audio (3–15 s recommended)."),
):
    """
    Clone the voice captured in `ref_audio` and synthesise `text` with it.

    - **ref_audio**: 3–15 seconds of clear reference speech.
    - **ref_text**: Strongly recommended — transcript of the reference audio.
    - **language**: Target output language.
    """
    try:
        # Save uploaded audio to a temp file
        audio_bytes = await ref_audio.read()
        import tempfile, os
        suffix = os.path.splitext(ref_audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = await model_manager.get_base()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=tmp_path,
                ref_text=ref_text if ref_text.strip() else None,
                temperature=temperature,
                top_p=top_p,
                speed=speed,
            ),
        )
        os.unlink(tmp_path)

        wavs, sr = result
        wav = wavs[0]
        duration = len(wav) / sr
        file_path = save_audio(wav, sr, prefix="voice_clone")

        return AudioResponse(
            success=True,
            file_path=file_path,
            sample_rate=sr,
            duration_sec=round(duration, 3),
            model_used=model_manager._base_id,
            message="Voice cloned successfully.",
        )

    except Exception as e:
        logger.exception("Voice clone failed")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
# Streaming voice clone
# --------------------------------------------------------------------------- #
@router.post(
    "/voice-clone/stream",
    summary="Stream cloned voice audio",
    response_class=StreamingResponse,
)
async def voice_clone_stream(
    text: str = Form(...),
    language: str = Form("English"),
    ref_text: str = Form(""),
    ref_audio: UploadFile = File(...),
):
    """
    Stream the cloned audio as WAV bytes chunks (~97 ms first-packet latency).
    """
    try:
        audio_bytes = await ref_audio.read()
        import tempfile, os
        suffix = os.path.splitext(ref_audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = await model_manager.get_base()

        async def audio_generator():
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(
                None,
                lambda: model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=tmp_path,
                    ref_text=ref_text if ref_text.strip() else None,
                    non_streaming_mode=False,
                ),
            )
            for chunk in stream:
                if isinstance(chunk, np.ndarray):
                    yield wav_to_bytes(chunk, 24000)
                else:
                    yield chunk
            os.unlink(tmp_path)

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    except Exception as e:
        logger.exception("Voice clone stream failed")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
# Batch voice clone
# --------------------------------------------------------------------------- #
@router.post(
    "/voice-clone/batch",
    summary="Batch voice cloning",
)
async def voice_clone_batch(
    texts: str = Form(..., description="JSON array of texts, e.g. '[\"Hello\", \"World\"]'"),
    languages: str = Form("English", description="Single language or JSON array matching texts."),
    ref_text: str = Form(""),
    ref_audio: UploadFile = File(...),
):
    """
    Clone a single voice and synthesise multiple texts in one call.
    Returns a list of AudioResponse objects.
    """
    import json, tempfile, os

    try:
        text_list = json.loads(texts)
        if isinstance(languages, str):
            try:
                lang_list = json.loads(languages)
            except Exception:
                lang_list = [languages] * len(text_list)
        else:
            lang_list = languages

        audio_bytes = await ref_audio.read()
        suffix = os.path.splitext(ref_audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = await model_manager.get_base()

        # Pre-compute the voice clone prompt for efficiency
        loop = asyncio.get_event_loop()
        prompt = await loop.run_in_executor(
            None,
            lambda: model.create_voice_clone_prompt(
                ref_audio=tmp_path,
                ref_text=ref_text if ref_text.strip() else None,
            ),
        )
        os.unlink(tmp_path)

        results = []
        for text, lang in zip(text_list, lang_list):
            result = await loop.run_in_executor(
                None,
                lambda t=text, l=lang: model.generate_voice_clone(
                    text=t,
                    language=l,
                    prompt=prompt,
                ),
            )
            wavs, sr = result
            wav = wavs[0]
            file_path = save_audio(wav, sr, prefix="batch_clone")
            results.append({
                "text": text,
                "language": lang,
                "file_path": file_path,
                "duration_sec": round(len(wav) / sr, 3),
                "sample_rate": sr,
            })

        return {"success": True, "results": results, "count": len(results)}

    except Exception as e:
        logger.exception("Batch voice clone failed")
        raise HTTPException(status_code=500, detail=str(e))
