"""Health check endpoint."""

from fastapi import APIRouter
from app.core.model_manager import model_manager

router = APIRouter()


@router.get("/health", summary="Health check")
async def health():
    """Returns service status and model loading state."""
    return {
        "status": "ok",
        "models": model_manager.status(),
    }
