"""
ModelManager — singleton that owns all three Qwen3-TTS model instances.

Three model variants are managed:
  • Base        → Voice Cloning   (Qwen3-TTS-12Hz-{size}-Base)
  • VoiceDesign → Voice Designing (Qwen3-TTS-12Hz-1.7B-VoiceDesign)
  • CustomVoice → Custom Voice    (Qwen3-TTS-12Hz-{size}-CustomVoice)

Models are loaded lazily on the first request or eagerly at startup,
depending on the `EAGER_LOAD` flag.  On Kaggle (16 GB T4) we can
comfortably keep all three 1.7B models loaded simultaneously in bfloat16
because each takes ~6–7 GB and Kaggle provides 16 GB VRAM — meaning we
load just ONE at a time and cache-swap on demand unless a dual-GPU
configuration is available.

Strategy used here: load all three eagerly.  If OOM, fall back to
on-demand loading (controlled by `eager_load=False` in config).
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import torch

from app.core.config import settings

logger = logging.getLogger(__name__)

# HuggingFace model identifiers
_MODEL_IDS = {
    "base_1.7B":         "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "base_0.6B":         "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice_design_1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_1.7B":       "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "custom_0.6B":       "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}


def _get_device_map():
    if settings.use_gpu and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _get_dtype():
    if settings.use_gpu and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _get_attn():
    if (
        settings.use_flash_attention
        and settings.use_gpu
        and torch.cuda.is_available()
    ):
        return "flash_attention_2"
    return "eager"


def _load_model(model_id: str):
    """Load a Qwen3TTSModel with appropriate settings."""
    from qwen_tts import Qwen3TTSModel  # type: ignore

    logger.info(f"⏳  Loading model {model_id} …")
    kwargs = {
        "device_map": _get_device_map(),
        "dtype": _get_dtype(),
        "attn_implementation": _get_attn(),
    }

    # Use local cache dir if specified (avoids re-downloading in Kaggle sessions)
    cache_dir = settings.model_cache_dir
    if cache_dir:
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        os.environ.setdefault("HF_HOME", cache_dir)

    model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
    logger.info(f"✅  {model_id} loaded.")
    return model


class ModelManager:
    """Thread-safe singleton manager for all three TTS model variants."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._base: Optional[object] = None
        self._voice_design: Optional[object] = None
        self._custom: Optional[object] = None
        self._size = settings.model_size  # "1.7B" | "0.6B"

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @property
    def _base_id(self) -> str:
        return _MODEL_IDS[f"base_{self._size}"]

    @property
    def _voice_design_id(self) -> str:
        # VoiceDesign is only released in 1.7B
        return _MODEL_IDS["voice_design_1.7B"]

    @property
    def _custom_id(self) -> str:
        return _MODEL_IDS[f"custom_{self._size}"]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def load_all(self):
        """Load all models concurrently in a thread pool."""
        loop = asyncio.get_event_loop()
        logger.info(f"Loading all Qwen3-TTS models (size={self._size}) …")

        async with self._lock:
            self._base = await loop.run_in_executor(None, _load_model, self._base_id)
            self._voice_design = await loop.run_in_executor(
                None, _load_model, self._voice_design_id
            )
            self._custom = await loop.run_in_executor(None, _load_model, self._custom_id)

    async def unload_all(self):
        """Release GPU memory."""
        import gc

        self._base = None
        self._voice_design = None
        self._custom = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded.")

    # ------------------------------------------------------------------ #
    # Accessors (lazy-load if called before load_all)
    # ------------------------------------------------------------------ #
    async def get_base(self):
        if self._base is None:
            async with self._lock:
                if self._base is None:
                    loop = asyncio.get_event_loop()
                    self._base = await loop.run_in_executor(
                        None, _load_model, self._base_id
                    )
        return self._base

    async def get_voice_design(self):
        if self._voice_design is None:
            async with self._lock:
                if self._voice_design is None:
                    loop = asyncio.get_event_loop()
                    self._voice_design = await loop.run_in_executor(
                        None, _load_model, self._voice_design_id
                    )
        return self._voice_design

    async def get_custom(self):
        if self._custom is None:
            async with self._lock:
                if self._custom is None:
                    loop = asyncio.get_event_loop()
                    self._custom = await loop.run_in_executor(
                        None, _load_model, self._custom_id
                    )
        return self._custom

    # ------------------------------------------------------------------ #
    # Info helpers
    # ------------------------------------------------------------------ #
    def get_custom_speakers(self) -> list[str]:
        if self._custom is not None:
            try:
                return self._custom.get_supported_speakers()
            except Exception:
                pass
        return [
            "Vivian", "Serena", "Uncle_Fu", "Chelsie",
            "Ethan", "Cherry", "Ryan", "Aura", "Daniel",
        ]

    def get_supported_languages(self) -> list[str]:
        return [
            "Chinese", "English", "Japanese", "Korean",
            "German", "French", "Russian", "Portuguese",
            "Spanish", "Italian", "Auto",
        ]

    def status(self) -> dict:
        return {
            "base":         self._base is not None,
            "voice_design": self._voice_design is not None,
            "custom":       self._custom is not None,
            "model_size":   self._size,
            "device":       _get_device_map(),
            "dtype":        str(_get_dtype()),
            "flash_attn":   _get_attn() == "flash_attention_2",
        }


# Singleton instance shared across the app
model_manager = ModelManager()
