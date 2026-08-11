"""
Audio utilities — save audio arrays to disk and return a URL-friendly path.
"""

import io
import os
import time
import uuid
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import settings

logger = logging.getLogger(__name__)


def _ensure_output_dir() -> Path:
    p = Path(settings.audio_output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_audio(wav: np.ndarray, sr: int, prefix: str = "output") -> str:
    """
    Write a waveform to disk.

    Returns the absolute file path as a string.
    """
    out_dir = _ensure_output_dir()
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.wav"
    filepath = out_dir / filename

    sf.write(str(filepath), wav, sr)
    logger.info(f"Audio saved → {filepath}")

    _rotate_old_files(out_dir)
    return str(filepath)


def wav_to_bytes(wav: np.ndarray, sr: int) -> bytes:
    """Encode waveform to WAV bytes (for streaming / inline response)."""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _rotate_old_files(out_dir: Path):
    """Delete oldest files when we exceed `max_audio_files`."""
    files = sorted(out_dir.glob("*.wav"), key=os.path.getmtime)
    excess = len(files) - settings.max_audio_files
    if excess > 0:
        for f in files[:excess]:
            try:
                f.unlink()
                logger.debug(f"Rotated old audio: {f.name}")
            except OSError:
                pass
