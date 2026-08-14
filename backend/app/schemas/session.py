"""Contracts for session and text-editing endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from backend.app.schemas.recognition import TextStateModel


class TextCommand(str, Enum):
    """Every way the text buffer can be edited from the UI."""

    SPACE = "space"
    BACKSPACE = "backspace"
    DELETE_WORD = "delete_word"
    CLEAR = "clear"
    ADD_CHARACTER = "add_character"
    ACCEPT_WORD = "accept_word"
    ACCEPT_SENTENCE = "accept_sentence"
    SET_TEXT = "set_text"


class TextCommandRequest(BaseModel):
    command: TextCommand
    session_id: Optional[str] = None
    #: Payload for ADD_CHARACTER / ACCEPT_WORD / ACCEPT_SENTENCE / SET_TEXT.
    value: str = Field(default="", max_length=500)


class SessionResponse(BaseModel):
    session_id: str
    text: TextStateModel = Field(default_factory=TextStateModel)


class ConfigResponse(BaseModel):
    """Everything the frontend needs to configure itself, in one call."""

    app_name: str
    version: str
    classes: list[str]
    gemini_enabled: bool
    gemini_model: Optional[str] = None
    min_confidence: float
    partial_confidence: float
    stable_frames: int
    commit_cooldown_ms: int
    duplicate_suppression_ms: int
    release_frames: int
    max_word_suggestions: int
    max_sentence_suggestions: int
    suggestion_debounce_ms: int
    #: Frames per second the client should send.  Sending faster only burns
    #: bandwidth: the stabilizer counts frames, not seconds.
    recommended_fps: int = 15


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_loaded: bool
    classes: int
    gemini_enabled: bool
    frames_processed: int
    average_latency_ms: float


__all__ = [
    "TextCommand",
    "TextCommandRequest",
    "SessionResponse",
    "ConfigResponse",
    "HealthResponse",
]
