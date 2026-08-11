"""
Voice Designing endpoint
========================
POST /api/v1/voice-design
  — Generate speech with a brand-new voice described in natural language.

POST /api/v1/voice-design/stream
  — Same but streamed.
"""

import asyncio
import logging

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
# Request schema
# --------------------------------------------------------------------------- #
class VoiceDesignRequest(BaseModel):
    text: str = Field(..., description="Text to synthesise.")
    language: Language = Field(Language.english, description="Target output language.")
    voice_description: str = Field(
        ...,
        description=(
            "Natural language description of the desired voice. "
            "E.g. 'A warm, elderly British gentleman with a slight rasp and calm cadence.'"
        ),
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p:       float = Field(0.8, ge=0.0, le=1.0)
    speed:       float = Field(1.0, ge=0.5, le=2.0)


class VoiceDesignStreamRequest(BaseModel):
    text: str
    language: Language = Language.english
    voice_description: str


# --------------------------------------------------------------------------- #
# Single voice design
# --------------------------------------------------------------------------- #
@router.post(
    "/voice-design",
    response_model=AudioResponse,
    summary="Design a new voice from a text description",
)
async def voice_design(req: VoiceDesignRequest):
    """
    Generate speech using a **brand-new voice** described in natural language —
    no reference audio required.

    The `voice_description` should capture:
    - Gender and age ("young male", "elderly woman")
    - Accent or nationality ("British English", "Mandarin Chinese")
    - Tone qualities ("warm", "raspy", "crisp", "authoritative")
    - Speaking style ("calm", "energetic", "newsreader cadence")
    """
    try:
        model = await model_manager.get_voice_design()
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None,
            lambda: model.generate_voice_design(
                text=req.text,
                language=req.language.value,
                instruct=req.voice_description,
                temperature=req.temperature,
                top_p=req.top_p,
                speed=req.speed,
            ),
        )

        wavs, sr = result
        wav = wavs[0]
        duration = len(wav) / sr
        file_path = save_audio(wav, sr, prefix="voice_design")

        return AudioResponse(
            success=True,
            file_path=file_path,
            sample_rate=sr,
            duration_sec=round(duration, 3),
            model_used=model_manager._voice_design_id,
            message="Voice designed and synthesised successfully.",
        )

    except Exception as e:
        logger.exception("Voice design failed")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
# Streaming voice design
# --------------------------------------------------------------------------- #
@router.post(
    "/voice-design/stream",
    summary="Stream designed voice audio",
    response_class=StreamingResponse,
)
async def voice_design_stream(req: VoiceDesignStreamRequest):
    """Stream audio chunks for a voice-designed synthesis request."""
    try:
        model = await model_manager.get_voice_design()

        async def audio_generator():
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(
                None,
                lambda: model.generate_voice_design(
                    text=req.text,
                    language=req.language.value,
                    instruct=req.voice_description,
                    non_streaming_mode=False,
                ),
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
        logger.exception("Voice design stream failed")
        raise HTTPException(status_code=500, detail=str(e))
