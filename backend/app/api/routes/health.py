"""Health and configuration."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import AppServices, get_services
from backend.app.schemas.session import ConfigResponse, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(services: AppServices = Depends(get_services)) -> HealthResponse:
    stats = services.recognition.stats
    return HealthResponse(
        status="ok",
        version=services.settings.app_version,
        model_loaded=True,
        classes=len(services.recognition.classes),
        gemini_enabled=services.suggestions.gemini.available,
        frames_processed=stats.frames,
        average_latency_ms=round(stats.average_ms, 2),
    )


@router.get("/config", response_model=ConfigResponse)
async def config(services: AppServices = Depends(get_services)) -> ConfigResponse:
    settings = services.settings
    gemini_on = services.suggestions.gemini.available
    return ConfigResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        classes=services.recognition.classes,
        gemini_enabled=gemini_on,
        gemini_model=settings.gemini_model if gemini_on else None,
        min_confidence=settings.min_confidence,
        partial_confidence=settings.partial_confidence,
        stable_frames=settings.stable_frames,
        commit_cooldown_ms=settings.commit_cooldown_ms,
        duplicate_suppression_ms=settings.duplicate_suppression_ms,
        release_frames=settings.release_frames,
        max_word_suggestions=settings.max_word_suggestions,
        max_sentence_suggestions=settings.max_sentence_suggestions,
        suggestion_debounce_ms=settings.suggestion_debounce_ms,
    )
