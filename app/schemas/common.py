"""
Shared Pydantic schemas used across multiple endpoints.
"""

from enum import Enum
from pydantic import BaseModel, Field


class Language(str, Enum):
    auto      = "Auto"
    english   = "English"
    chinese   = "Chinese"
    japanese  = "Japanese"
    korean    = "Korean"
    german    = "German"
    french    = "French"
    russian   = "Russian"
    portuguese = "Portuguese"
    spanish   = "Spanish"
    italian   = "Italian"


class GenerationParams(BaseModel):
    """Common LM sampling parameters shared by all generation endpoints."""
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p:       float = Field(0.8, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    speed:       float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier.")
    max_new_tokens: int = Field(2000, ge=100, le=8000, description="Max audio tokens to generate.")


class AudioResponse(BaseModel):
    """Standard audio response returned by all generation endpoints."""
    success:        bool
    file_path:      str  = Field("", description="Absolute path of saved WAV file on the server.")
    audio_url:      str  = Field("", description="Relative URL to fetch the generated audio file.")
    sample_rate:    int  = Field(0,  description="Sample rate of the generated audio (Hz).")
    duration_sec:   float = Field(0.0, description="Duration of generated audio in seconds.")
    model_used:     str  = Field("", description="HuggingFace model ID that was used.")
    message:        str  = Field("", description="Human-readable status or error message.")
