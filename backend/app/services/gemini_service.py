"""Gemini access, via LangChain, in exactly one place.

LangChain earns its place here for three concrete things:

* ``ChatPromptTemplate`` keeps the prompt as declared data instead of f-strings
  scattered through the app;
* ``.with_structured_output(LLMSuggestionPayload)`` makes the model return a
  validated Pydantic object rather than prose we would have to parse;
* the ``prompt | model`` runnable gives a native ``ainvoke`` so a slow model
  never blocks the recognition loop.

Nothing else in the codebase imports ``langchain`` or talks to Google.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from backend.app.core.config import Settings
from backend.app.core.logging import get_logger
from backend.app.schemas.suggestions import LLMSuggestionPayload

logger = get_logger(__name__)

_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]{0,19}$")

#: The Gemini API refuses a request deadline shorter than this.
MIN_API_DEADLINE = 10.0


SYSTEM_PROMPT = """\
You are the predictive-text engine inside a sign-language communication app.

The user is Deaf or hard of hearing and spells every letter with a hand sign in \
front of a camera. Each letter costs them several seconds, so your only job is \
to finish their thought in as few letters as possible. You are a keyboard, not \
a chat assistant.

Rules you must never break:
1. Every word in "word_suggestions" MUST start with the exact letters in \
CURRENT WORD (case-insensitive). If CURRENT WORD is empty, predict the most \
likely NEXT word instead.
2. Every entry in "sentence_suggestions" MUST begin with the words in TEXT SO \
FAR, keep them in order, and finish as one complete, grammatical, natural \
sentence with punctuation.
3. Prefer short, common, everyday conversational words. A frequent 3-letter \
word beats a precise 9-letter one.
4. Rank by what this person most plausibly means right now, given the whole \
sentence. Grammar first, then meaning, then frequency.
5. No creativity, no rare or archaic words, no slang, no emoji, no names, no \
explanations, no duplicates, and never repeat a word already completed.
6. If you are unsure, return fewer suggestions rather than invented ones.

Return only the structured fields.\
"""

HUMAN_PROMPT = """\
TEXT SO FAR: {text}
CURRENT WORD: {current_word}
PREVIOUS WORDS: {context}
LOCAL CANDIDATES (fast dictionary guesses, may be wrong): {local_words}

Give at most {max_words} word suggestions and at most {max_sentences} sentence \
suggestions. Keep any local candidate that is genuinely the best fit, reorder \
it by likelihood, and replace the rest with better ones.\
"""


@dataclass(frozen=True, slots=True)
class SuggestionContext:
    """Everything the model is told about the current text."""

    text: str
    current_word: str
    context: Sequence[str]
    local_words: Sequence[str]
    max_words: int
    max_sentences: int

    def cache_key(self) -> str:
        """Identical contexts must reuse the same answer."""
        return "|".join(
            [
                self.text.strip().upper(),
                self.current_word.strip().upper(),
                str(self.max_words),
                str(self.max_sentences),
            ]
        )


class GeminiUnavailable(RuntimeError):
    """Raised when Gemini is not configured or cannot be reached."""


class GeminiSuggestionService:
    """Async, structured, time-boxed access to Gemini.

    The chain is built lazily on first use so that importing the app -- and
    running the test suite -- never requires an API key or a network call.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chain = None
        self._init_failed = False
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return self._settings.gemini_enabled and not self._init_failed

    @property
    def model_name(self) -> str:
        return self._settings.gemini_model

    # ------------------------------------------------------------------

    async def _get_chain(self):
        if self._chain is not None:
            return self._chain
        async with self._lock:
            if self._chain is not None:
                return self._chain
            if not self._settings.gemini_enabled:
                raise GeminiUnavailable("GEMINI_API_KEY is not set")
            try:
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_google_genai import ChatGoogleGenerativeAI

                model = ChatGoogleGenerativeAI(
                    model=self._settings.gemini_model,
                    google_api_key=self._settings.gemini_api_key.get_secret_value(),
                    temperature=self._settings.gemini_temperature,
                    max_output_tokens=self._settings.gemini_max_output_tokens,
                    # The API rejects a deadline below 10s outright, so this
                    # is only a backstop that closes the connection.  The
                    # budget the user actually configured is enforced by the
                    # asyncio.wait_for in `suggest`, which can be any length.
                    timeout=max(self._settings.gemini_timeout_seconds, MIN_API_DEADLINE),
                    max_retries=0,  # we would rather fall back than wait
                )
                prompt = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
                )
                self._chain = prompt | model.with_structured_output(
                    LLMSuggestionPayload
                )
                logger.info(
                    "Gemini suggestions enabled (model=%s)",
                    self._settings.gemini_model,
                )
            except Exception as exc:  # noqa: BLE001 - report and degrade
                self._init_failed = True
                logger.warning("Could not initialise Gemini: %s", exc)
                raise GeminiUnavailable(str(exc)) from exc
        return self._chain

    # ------------------------------------------------------------------

    async def suggest(self, ctx: SuggestionContext) -> LLMSuggestionPayload:
        """Ask Gemini for suggestions.

        Raises :class:`GeminiUnavailable` on any failure -- missing key, bad
        network, timeout or a response that does not validate.  Callers treat
        that as "use the local suggestions" and carry on.
        """
        chain = await self._get_chain()
        try:
            payload = await asyncio.wait_for(
                chain.ainvoke(
                    {
                        "text": ctx.text or "(nothing yet)",
                        "current_word": ctx.current_word or "(none)",
                        "context": ", ".join(ctx.context) or "(none)",
                        "local_words": ", ".join(ctx.local_words) or "(none)",
                        "max_words": ctx.max_words,
                        "max_sentences": ctx.max_sentences,
                    }
                ),
                timeout=self._settings.gemini_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise GeminiUnavailable(
                f"Gemini timed out after {self._settings.gemini_timeout_seconds}s"
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any SDK/parse error degrades
            raise GeminiUnavailable(f"Gemini call failed: {exc}") from exc

        if payload is None or not isinstance(payload, LLMSuggestionPayload):
            raise GeminiUnavailable("Gemini returned an unusable response")

        return LLMSuggestionPayload(
            word_suggestions=self._clean_words(payload.word_suggestions, ctx),
            sentence_suggestions=self._clean_sentences(
                payload.sentence_suggestions, ctx
            ),
        )

    # ------------------------------------------------------------------
    # Never trust the model's output shape
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_words(
        words: Sequence[object], ctx: SuggestionContext
    ) -> List[str]:
        prefix = ctx.current_word.strip().lower()
        seen: set[str] = set()
        cleaned: List[str] = []
        for raw in words or []:
            if not isinstance(raw, str):
                continue
            word = raw.strip().strip(".,!?;:\"'").split()[:1]
            if not word:
                continue
            candidate = word[0]
            if not _WORD_RE.match(candidate):
                continue
            if prefix and not candidate.lower().startswith(prefix):
                continue  # rule 1 violated -- drop it rather than confuse the user
            upper = candidate.upper()
            if upper in seen:
                continue
            seen.add(upper)
            cleaned.append(upper)
            if len(cleaned) >= ctx.max_words:
                break
        return cleaned

    @staticmethod
    def _clean_sentences(
        sentences: Sequence[object], ctx: SuggestionContext
    ) -> List[str]:
        seen: set[str] = set()
        cleaned: List[str] = []
        for raw in sentences or []:
            if not isinstance(raw, str):
                continue
            sentence = " ".join(raw.split())
            if not 2 <= len(sentence) <= 160:
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(sentence)
            if len(cleaned) >= ctx.max_sentences:
                break
        return cleaned


__all__ = [
    "GeminiSuggestionService",
    "SuggestionContext",
    "GeminiUnavailable",
    "SYSTEM_PROMPT",
]
