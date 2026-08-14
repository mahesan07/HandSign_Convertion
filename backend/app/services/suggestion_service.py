"""Orchestrates suggestions: local first, Gemini second, never blocking.

The contract with the rest of the app is deliberately two-step:

``local_suggestions()``
    Returns in microseconds.  The UI renders this immediately, every time.

``schedule_refinement()``
    Queues a debounced Gemini call and pushes the better answer through a
    callback when (and only if) it arrives.  Nothing waits on it.

Everything that keeps the API bill and the latency down lives here: a TTL
cache, in-flight de-duplication, per-session debouncing and cancellation of
requests that the user has already typed past.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from backend.app.core.config import Settings
from backend.app.core.logging import get_logger
from backend.app.schemas.suggestions import (
    LLMSuggestionPayload,
    SuggestionRequest,
    SuggestionResponse,
    SuggestionSource,
)
from backend.app.services.gemini_service import (
    GeminiSuggestionService,
    GeminiUnavailable,
    SuggestionContext,
)
from backend.app.services.local_suggestions import LocalSuggestionEngine

logger = get_logger(__name__)

RefinementCallback = Callable[[SuggestionResponse], Awaitable[None]]


class _TTLCache:
    """Small LRU cache with expiry.  Repeated states cost nothing."""

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self._max_size = max(1, max_size)
        self._ttl = ttl_seconds
        self._store: "OrderedDict[str, Tuple[float, LLMSuggestionPayload]]" = (
            OrderedDict()
        )

    def get(self, key: str) -> Optional[LLMSuggestionPayload]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.monotonic():
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key: str, value: LLMSuggestionPayload) -> None:
        self._store[key] = (time.monotonic() + self._ttl, value)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class SuggestionService:
    def __init__(
        self,
        settings: Settings,
        local_engine: LocalSuggestionEngine | None = None,
        gemini: GeminiSuggestionService | None = None,
    ) -> None:
        self._settings = settings
        self.local_engine = local_engine or LocalSuggestionEngine()
        self.gemini = gemini if gemini is not None else GeminiSuggestionService(settings)
        self._cache = _TTLCache(
            settings.suggestion_cache_size, settings.suggestion_cache_ttl_seconds
        )
        #: key -> the debounce/refine task currently running for it
        self._pending: Dict[str, asyncio.Task] = {}
        #: cache key -> in-flight Gemini call, so two sessions asking the same
        #: thing at the same time make one request
        self._inflight: Dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Step 1: instant
    # ------------------------------------------------------------------

    def local_suggestions(self, request: SuggestionRequest) -> SuggestionResponse:
        local = self.local_engine.suggest(
            request.resolved_context(),
            request.current_word,
            max_words=request.max_words,
            max_sentences=request.max_sentences,
        )
        return SuggestionResponse(
            word_suggestions=local.words,
            sentence_suggestions=local.sentences,
            source=SuggestionSource.LOCAL,
            llm_pending=self.should_use_llm(request),
        )

    def should_use_llm(self, request: SuggestionRequest) -> bool:
        """Gemini is only worth calling once there is something to reason about."""
        if not request.use_llm or not self.gemini.available:
            return False
        context = request.resolved_context()
        # A single letter carries almost no intent, and firing on every letter
        # is exactly the per-keystroke traffic we want to avoid.
        return bool(context) or len(request.current_word) >= 2

    # ------------------------------------------------------------------
    # Step 2: refined
    # ------------------------------------------------------------------

    async def refined_suggestions(
        self, request: SuggestionRequest
    ) -> SuggestionResponse:
        """Local suggestions merged with Gemini's, awaiting the call.

        Used by the REST endpoint and the tests.  The websocket path uses
        :meth:`schedule_refinement` instead so nothing is ever awaited on the
        recognition path.
        """
        base = self.local_suggestions(request)
        if not self.should_use_llm(request):
            return base

        ctx = self._context(request, base.word_suggestions)
        key = ctx.cache_key()

        cached = self._cache.get(key)
        if cached is not None:
            return self._merge(base, cached, SuggestionSource.CACHE)

        try:
            payload = await self._call_gemini(key, ctx)
        except asyncio.CancelledError:
            raise
        except GeminiUnavailable as exc:
            logger.info("Falling back to local suggestions: %s", exc)
            base.llm_pending = False
            base.notice = "Smart suggestions are unavailable right now."
            return base

        self._cache.put(key, payload)
        return self._merge(base, payload, SuggestionSource.GEMINI)

    def schedule_refinement(
        self,
        session_key: str,
        request: SuggestionRequest,
        callback: RefinementCallback,
    ) -> bool:
        """Debounce a Gemini refinement for one session.

        Any refinement still pending for the same session is cancelled first --
        if the user has typed another letter, the previous answer is stale.
        Returns True when a refinement was actually scheduled.
        """
        self.cancel(session_key)
        if not self.should_use_llm(request):
            return False

        task = asyncio.create_task(
            self._debounced(session_key, request, callback)
        )
        self._pending[session_key] = task
        return True

    def cancel(self, session_key: str) -> None:
        task = self._pending.pop(session_key, None)
        if task is not None and not task.done():
            task.cancel()

    async def aclose(self) -> None:
        for key in list(self._pending):
            self.cancel(key)
        for task in list(self._inflight.values()):
            task.cancel()
        self._pending.clear()
        self._inflight.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _debounced(
        self,
        session_key: str,
        request: SuggestionRequest,
        callback: RefinementCallback,
    ) -> None:
        try:
            base = self.local_suggestions(request)
            ctx = self._context(request, base.word_suggestions)
            key = ctx.cache_key()

            # A cache hit needs no quiet period -- answer straight away.
            cached = self._cache.get(key)
            if cached is not None:
                await callback(self._merge(base, cached, SuggestionSource.CACHE))
                return

            await asyncio.sleep(self._settings.suggestion_debounce_ms / 1000.0)

            payload = await self._call_gemini(key, ctx)
            self._cache.put(key, payload)
            await callback(self._merge(base, payload, SuggestionSource.GEMINI))
        except asyncio.CancelledError:
            # The user typed on; this answer is stale by definition.
            raise
        except GeminiUnavailable as exc:
            logger.info("Refinement unavailable: %s", exc)
            stale = self.local_suggestions(request)
            stale.llm_pending = False
            stale.notice = "Smart suggestions are unavailable right now."
            await callback(stale)
        except Exception:  # noqa: BLE001 - a bad refinement must never kill the socket
            logger.exception("Suggestion refinement failed")
        finally:
            if self._pending.get(session_key) is asyncio.current_task():
                self._pending.pop(session_key, None)

    async def _call_gemini(
        self, key: str, ctx: SuggestionContext
    ) -> LLMSuggestionPayload:
        """One request per distinct context, however many callers want it."""
        task = self._inflight.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self.gemini.suggest(ctx))
            self._inflight[key] = task
            task.add_done_callback(
                lambda _t, k=key: self._inflight.pop(k, None)
            )
        # Shielded so that one caller giving up does not cancel the shared call.
        return await asyncio.shield(task)

    def _context(
        self, request: SuggestionRequest, local_words: List[str]
    ) -> SuggestionContext:
        return SuggestionContext(
            text=request.text or " ".join(request.resolved_context()),
            current_word=request.current_word,
            context=request.resolved_context()[-8:],
            local_words=local_words,
            max_words=request.max_words,
            max_sentences=request.max_sentences,
        )

    @staticmethod
    def _merge(
        base: SuggestionResponse,
        payload: LLMSuggestionPayload,
        source: SuggestionSource,
    ) -> SuggestionResponse:
        """Gemini's ranking wins, local fills any gaps.

        The user therefore never ends up with *fewer* options than the local
        engine already offered, even if the model returns an empty list.
        """
        words = list(payload.word_suggestions)
        for word in base.word_suggestions:
            if len(words) >= len(base.word_suggestions) or len(words) >= 8:
                break
            if word not in words:
                words.append(word)

        sentences = list(payload.sentence_suggestions)
        for sentence in base.sentence_suggestions:
            if len(sentences) >= len(base.sentence_suggestions):
                break
            if sentence not in sentences:
                sentences.append(sentence)

        return SuggestionResponse(
            word_suggestions=words,
            sentence_suggestions=sentences,
            source=source,
            llm_pending=False,
        )


__all__ = ["SuggestionService"]
