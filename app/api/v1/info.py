"""
Info endpoint — exposes model metadata, available speakers, languages.
"""

from fastapi import APIRouter
from app.core.model_manager import model_manager

router = APIRouter()


@router.get("/info", summary="Model information")
async def info():
    """Returns available speakers, languages, and model metadata."""
    return {
        "models": {
            "voice_cloning": {
                "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            },
            "voice_design": {
                "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            },
            "custom_voice": {
                "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            },
        },
        "supported_languages": model_manager.get_supported_languages(),
        "custom_voice_speakers": model_manager.get_custom_speakers(),
        "capabilities": {
            "streaming": True,
            "batch_inference": True,
            "prompt_caching": True,
        },
    }
