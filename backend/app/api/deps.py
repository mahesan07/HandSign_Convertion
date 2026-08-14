"""The services every request shares.

They are built once during application start-up (see ``main.lifespan``) and
handed out from ``app.state`` -- in particular the ML model is loaded exactly
once per process, never per request.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request, WebSocket

from backend.app.core.config import Settings
from backend.app.services.recognition_service import RecognitionService
from backend.app.services.session import SessionStore
from backend.app.services.sign_translation import SignTranslator
from backend.app.services.suggestion_service import SuggestionService


@dataclass
class AppServices:
    settings: Settings
    recognition: RecognitionService
    sessions: SessionStore
    suggestions: SuggestionService
    signs: SignTranslator

    async def aclose(self) -> None:
        await self.suggestions.aclose()
        self.recognition.close()


def get_services(request: Request) -> AppServices:
    return request.app.state.services


def get_services_ws(websocket: WebSocket) -> AppServices:
    return websocket.app.state.services


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.services.settings


__all__ = ["AppServices", "get_services", "get_services_ws", "get_settings_dep"]
