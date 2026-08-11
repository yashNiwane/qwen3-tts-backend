"""
Qwen3-TTS Backend — Main Application
=====================================
FastAPI backend exposing Voice Cloning, Voice Designing, and Custom Voice
endpoints powered by the qwen-tts library.  Designed to run inside a
Kaggle notebook (GPU T4) and tunnelled to the internet via ngrok.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.model_manager import model_manager
from app.api.v1 import health, voice_clone, voice_design, custom_voice, info

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load / unload models around the server lifetime
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀  Starting Qwen3-TTS backend …")
    await model_manager.load_all()
    logger.info("✅  All models ready.")
    yield
    logger.info("🛑  Shutting down — releasing model memory …")
    await model_manager.unload_all()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Qwen3-TTS API",
    description=(
        "Production-ready REST API for Qwen3-TTS — supports Voice Cloning, "
        "Voice Designing, and Custom Voice (preset speakers) with streaming support."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins so the frontend (any host) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
PREFIX = "/api/v1"

app.include_router(health.router,        prefix=PREFIX, tags=["Health"])
app.include_router(info.router,          prefix=PREFIX, tags=["Info"])
app.include_router(voice_clone.router,   prefix=PREFIX, tags=["Voice Cloning"])
app.include_router(voice_design.router,  prefix=PREFIX, tags=["Voice Designing"])
app.include_router(custom_voice.router,  prefix=PREFIX, tags=["Custom Voice"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Qwen3-TTS API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }
