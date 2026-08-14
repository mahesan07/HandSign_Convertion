"""Request/response contracts for the suggestion API."""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


class SuggestionSource(str, Enum):
    """Where a suggestion bundle came from -- surfaced in the UI so the user
    can tell instant local guesses from language-model refinements."""

    LOCAL = "local"
    GEMINI = "gemini"
    CACHE = "cache"


class SuggestionRequest(BaseModel):
    """What the client knows about the text being written."""

    text: str = Field(default="", max_length=500)
    current_word: str = Field(default="", max_length=40)
    context: List[str] = Field(default_factory=list, max_length=64)
    max_words: int = Field(default=4, ge=1, le=8)
    max_sentences: int = Field(default=3, ge=1, le=6)
    #: When false, only the instant local engine runs.
    use_llm: bool = True

    @field_validator("current_word")
    @classmethod
    def _strip_current(cls, value: str) -> str:
        return value.strip()

    @field_validator("context")
    @classmethod
    def _clean_context(cls, value: List[str]) -> List[str]:
        return [item.strip() for item in value if item and item.strip()][-32:]

    def resolved_context(self) -> List[str]:
        """Prefer the explicit context list; fall back to splitting ``text``."""
        if self.context:
            return self.context
        tokens = self.text.split()
        if self.current_word and tokens and tokens[-1] == self.current_word:
            tokens = tokens[:-1]
        return tokens


class SuggestionResponse(BaseModel):
    """The shape the frontend renders.  Both lists are always present."""

    word_suggestions: List[str] = Field(default_factory=list)
    sentence_suggestions: List[str] = Field(default_factory=list)
    source: SuggestionSource = SuggestionSource.LOCAL
    #: True when a Gemini refinement is on its way over the websocket.
    llm_pending: bool = False
    #: Populated when Gemini was asked but could not answer.
    notice: str | None = None


class LLMSuggestionPayload(BaseModel):
    """The *only* shape Gemini is allowed to return.

    Anything else -- prose, markdown, extra keys -- fails validation and the
    caller silently falls back to the local engine.
    """

    word_suggestions: List[str] = Field(
        default_factory=list,
        description="Single words continuing or completing the current word.",
    )
    sentence_suggestions: List[str] = Field(
        default_factory=list,
        description="Complete natural sentences that begin with the text so far.",
    )


__all__ = [
    "SuggestionSource",
    "SuggestionRequest",
    "SuggestionResponse",
    "LLMSuggestionPayload",
]
