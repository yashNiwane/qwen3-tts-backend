"""
Custom Voice endpoint
=====================
POST /api/v1/custom-voice
  — Synthesise text with one of the 9 built-in premium voices.

POST /api/v1/custom-voice/stream
  — Same but streamed.

POST /api/v1/custom-voice/batch
  — Batch synthesis across multiple speakers / instructions.

GET  /api/v1/custom-voice/speakers
  — List available preset speakers.
"""

import asyncio
import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.model_manager import model_manager
from app.core.audio_utils import save_audio, wav_to_bytes
from app.schemas.common import AudioResponse, Language

logger = logging.getLogger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------- #
# Request schemas
# --------------------------------------------------------------------------- #
class CustomVoiceRequest(BaseModel):
    text: str = Field(..., description="Text to synthesise.")
    language: Language = Field(Language.english, description="Target output language.")
    speaker: str = Field(
        "Ryan",
        description="Preset speaker name. Call /speakers to list available options.",
    )
    instruction: str = Field(
        "",
        description=(
            "Optional style instruction. "
            "E.g. 'Speak slowly and calmly.' or 'Very excited and happy.'"
        ),
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p:       float = Field(0.8, ge=0.0, le=1.0)
    speed:       float = Field(1.0, ge=0.5, le=2.0)


class CustomVoiceBatchItem(BaseModel):
    text: str
    language: Language = Language.english
    speaker: str = "Ryan"
    instruction: str = ""


class CustomVoiceBatchRequest(BaseModel):
    items: List[CustomVoiceBatchItem] = Field(..., description="List of synthesis requests.")


class CustomVoiceStreamRequest(BaseModel):
    text: str
    language: Language = Language.english
    speaker: str = "Ryan"
    instruction: str = ""


# --------------------------------------------------------------------------- #
# List speakers
# --------------------------------------------------------------------------- #
@router.get("/custom-voice/speakers", summary="List available preset speakers")
async def list_speakers():
    """Return all available preset speaker names with their voice descriptions."""
    speakers = [
        {"name": "aiden",    "description": "Energetic young male voice.",                         "native_language": "English"},
        {"name": "dylan",    "description": "Calm, composed male voice.",                          "native_language": "English"},
        {"name": "eric",     "description": "Friendly, warm male voice.",                          "native_language": "English"},
        {"name": "ono_anna", "description": "Clear, precise female voice.",                        "native_language": "Japanese"},
        {"name": "ryan",     "description": "Deep, professional male narrator voice.",             "native_language": "English"},
        {"name": "serena",   "description": "Warm, gentle young female voice.",                    "native_language": "Chinese"},
        {"name": "sohee",    "description": "Bright, expressive young female voice.",              "native_language": "Korean"},
        {"name": "uncle_fu", "description": "Seasoned, authoritative mature male voice.",          "native_language": "Chinese"},
        {"name": "vivian",   "description": "Bright, slightly edgy young female voice.",           "native_language": "Chinese"},
    ]
    return {
        "speakers": speakers,
        "total": len(speakers),
        "tip": "Each speaker can speak any of the 10 supported languages; native language yields best quality.",
    }


# --------------------------------------------------------------------------- #
# Single custom voice
# --------------------------------------------------------------------------- #
@router.post(
    "/custom-voice",
    response_model=AudioResponse,
    summary="Synthesise with a preset voice",
)
async def custom_voice(req: CustomVoiceRequest):
    """
    Synthesise `text` using one of the 9 premium preset voices.

    Use the optional `instruction` field to control emotion, speed, and tone:
    - `"Speak very slowly and gently."`
    - `"Excited and energetic tone."`
    - `"Whisper softly."`
    """
    try:
        model = await model_manager.get_custom()
        loop = asyncio.get_event_loop()

        generate_kwargs = dict(
            text=req.text,
            language=req.language.value,
            speaker=req.speaker,
            temperature=req.temperature,
            top_p=req.top_p,
            speed=req.speed,
        )
        if req.instruction.strip():
            generate_kwargs["instruction"] = req.instruction

        result = await loop.run_in_executor(
            None,
            lambda: model.generate_custom_voice(**generate_kwargs),
        )

        wavs, sr = result
        wav = wavs[0]
        duration = len(wav) / sr
        file_path = save_audio(wav, sr, prefix=f"custom_{req.speaker.lower()}")

        return AudioResponse(
            success=True,
            file_path=file_path,
            sample_rate=sr,
            duration_sec=round(duration, 3),
            model_used=model_manager._custom_id,
            message=f"Custom voice '{req.speaker}' synthesised successfully.",
        )

    except Exception as e:
        logger.exception("Custom voice failed")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
# Streaming custom voice
# --------------------------------------------------------------------------- #
@router.post(
    "/custom-voice/stream",
    summary="Stream custom voice audio",
    response_class=StreamingResponse,
)
async def custom_voice_stream(req: CustomVoiceStreamRequest):
    """Stream audio chunks for a custom-voice synthesis request."""
    try:
        model = await model_manager.get_custom()

        async def audio_generator():
            loop = asyncio.get_event_loop()
            generate_kwargs = dict(
                text=req.text,
                language=req.language.value,
                speaker=req.speaker,
                non_streaming_mode=False,
            )
            if req.instruction.strip():
                generate_kwargs["instruction"] = req.instruction

            stream = await loop.run_in_executor(
                None,
                lambda: model.generate_custom_voice(**generate_kwargs),
            )
            for chunk in stream:
                if isinstance(chunk, np.ndarray):
                    yield wav_to_bytes(chunk, 24000)
                else:
                    yield chunk

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    except Exception as e:
        logger.exception("Custom voice stream failed")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
# Batch custom voice
# --------------------------------------------------------------------------- #
@router.post(
    "/custom-voice/batch",
    summary="Batch synthesise with multiple speakers / instructions",
)
async def custom_voice_batch(req: CustomVoiceBatchRequest):
    """
    Synthesise multiple texts with different speakers / instructions in one call.
    Returns a list of audio file paths and metadata.
    """
    try:
        model = await model_manager.get_custom()
        loop = asyncio.get_event_loop()

        texts       = [item.text        for item in req.items]
        languages   = [item.language.value for item in req.items]
        speakers    = [item.speaker     for item in req.items]
        instructions = [item.instruction for item in req.items]

        result = await loop.run_in_executor(
            None,
            lambda: model.generate_custom_voice(
                text=texts,
                language=languages,
                speaker=speakers,
                instruction=instructions,
            ),
        )

        wavs, sr = result
        outputs = []
        for i, (wav, item) in enumerate(zip(wavs, req.items)):
            file_path = save_audio(wav, sr, prefix=f"batch_custom_{item.speaker.lower()}")
            outputs.append({
                "index":        i,
                "speaker":      item.speaker,
                "language":     item.language.value,
                "text":         item.text,
                "file_path":    file_path,
                "duration_sec": round(len(wav) / sr, 3),
                "sample_rate":  sr,
            })

        return {"success": True, "results": outputs, "count": len(outputs)}

    except Exception as e:
        logger.exception("Custom voice batch failed")
        raise HTTPException(status_code=500, detail=str(e))
