"""Contracts for recognition: the REST predict endpoint and the websocket."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from backend.app.schemas.suggestions import SuggestionResponse


class TextStateModel(BaseModel):
    """The buffer, exactly as the UI should render it."""

    text: str = ""
    words: List[str] = Field(default_factory=list)
    current_word: str = ""

    @classmethod
    def from_state(cls, state: object) -> "TextStateModel":
        """Build from a :class:`~backend.app.services.text_buffer.TextState`."""
        return cls(
            text=state.text,          # type: ignore[attr-defined]
            words=list(state.words),  # type: ignore[attr-defined]
            current_word=state.current_word,  # type: ignore[attr-defined]
        )


class PredictionModel(BaseModel):
    letter: Optional[str] = None
    confidence: float = 0.0
    alternatives: List[Tuple[str, float]] = Field(default_factory=list)


class RecognitionUpdate(BaseModel):
    """Pushed for every frame the backend processes."""

    type: Literal["recognition"] = "recognition"
    session_id: str
    hand_detected: bool
    status: str
    prediction: PredictionModel = Field(default_factory=PredictionModel)
    #: 0.0-1.0 towards committing the held letter.
    progress: float = 0.0
    #: Accumulated evidence for the candidate. Fractional, because a
    #: low-confidence frame contributes less than a whole frame's worth.
    stable_count: float = 0.0
    required_frames: int = 1
    #: Set only on the frame where a letter was actually added.
    committed_letter: Optional[str] = None
    #: Landmarks in normalized [0,1] image space, for the canvas overlay.
    landmarks: List[Tuple[float, float]] = Field(default_factory=list)
    text: TextStateModel = Field(default_factory=TextStateModel)
    #: Pipeline time for this frame, in milliseconds.
    latency_ms: float = 0.0


class SuggestionsUpdate(BaseModel):
    """Pushed whenever the text changes, and again when Gemini answers."""

    type: Literal["suggestions"] = "suggestions"
    session_id: str
    suggestions: SuggestionResponse
    text: TextStateModel = Field(default_factory=TextStateModel)


class TextUpdate(BaseModel):
    """Pushed after an editing command (space, backspace, clear, ...)."""

    type: Literal["text"] = "text"
    session_id: str
    text: TextStateModel = Field(default_factory=TextStateModel)


class ErrorUpdate(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    #: False when the client may keep sending frames regardless.
    fatal: bool = False


class ReadyUpdate(BaseModel):
    type: Literal["ready"] = "ready"
    session_id: str
    classes: List[str] = Field(default_factory=list)
    gemini_enabled: bool = False


# ----------------------------------------------------------------------
# REST
# ----------------------------------------------------------------------


class PredictRequest(BaseModel):
    """A single frame, for clients that do not want a websocket."""

    #: base64-encoded JPEG/PNG; a ``data:`` URL prefix is accepted.
    image: str = Field(min_length=8)
    session_id: Optional[str] = None
    #: Whether the client already mirrored the image (see RecognitionService).
    mirrored: bool = True
    #: When false the prediction is reported but not fed to the stabilizer.
    apply_to_buffer: bool = True


class PredictResponse(BaseModel):
    session_id: str
    hand_detected: bool
    status: str
    prediction: PredictionModel = Field(default_factory=PredictionModel)
    progress: float = 0.0
    committed_letter: Optional[str] = None
    landmarks: List[Tuple[float, float]] = Field(default_factory=list)
    text: TextStateModel = Field(default_factory=TextStateModel)
    latency_ms: float = 0.0


__all__ = [
    "TextStateModel",
    "PredictionModel",
    "RecognitionUpdate",
    "SuggestionsUpdate",
    "TextUpdate",
    "ErrorUpdate",
    "ReadyUpdate",
    "PredictRequest",
    "PredictResponse",
]
