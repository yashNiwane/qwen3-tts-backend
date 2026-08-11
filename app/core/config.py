"""
Settings — loaded from environment / .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parents[3] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    host: str = "0.0.0.0"
    port: int = 8000

    # ------------------------------------------------------------------ #
    # ngrok
    # ------------------------------------------------------------------ #
    ngrok_authtoken: str = ""
    ngrok_api_id: str = ""
    ngrok_api_token: str = ""

    # ------------------------------------------------------------------ #
    # Model settings
    # ------------------------------------------------------------------ #
    # Which model sizes to load at startup.
    # Options: "1.7B" | "0.6B" | "both"
    model_size: str = "1.7B"

    # Set to True on Kaggle / GPU machines; False for CPU-only testing
    use_gpu: bool = True

    # Flash-Attention 2 — only works on Ampere+ GPUs
    use_flash_attention: bool = True

    # Cache directory for downloaded model weights
    model_cache_dir: str = "/kaggle/working/models"

    # ------------------------------------------------------------------ #
    # Audio output
    # ------------------------------------------------------------------ #
    audio_output_dir: str = "/kaggle/working/audio_outputs"
    max_audio_files: int = 200   # rotate older files beyond this limit

    # ------------------------------------------------------------------ #
    # Kaggle credentials (passed as env vars in the notebook)
    # ------------------------------------------------------------------ #
    kaggle_api_token: str = ""


settings = Settings()
