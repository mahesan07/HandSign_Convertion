"""Suggestions: the local engine, and the service that layers Gemini on top."""

from __future__ import annotations

import asyncio

import pytest

from backend.app.schemas.suggestions import (
    LLMSuggestionPayload,
    SuggestionRequest,
    SuggestionSource,
)
from backend.app.services.gemini_service import (
    GeminiSuggestionService,
    GeminiUnavailable,
    SuggestionContext,
)
from backend.app.services.local_suggestions import LocalSuggestionEngine
from backend.app.services.suggestion_service import SuggestionService
from tests.conftest import StubGemini


# ======================================================================
# Local engine
# ======================================================================


@pytest.fixture(scope="module")
def local() -> LocalSuggestionEngine:
    return LocalSuggestionEngine()


def test_empty_text_still_offers_a_starting_point(local):
    result = local.suggest([], "")
    assert result.words
    assert result.sentences


def test_prefix_completion(local):
    words = local.suggest_words([], "HEL")
    assert "HELLO" in words
    assert all(word.startswith("HEL") for word in words)


def test_context_beats_raw_frequency(local):
    """"HOW ARE" must suggest YOU, which is not the most frequent word."""
    assert local.suggest_words(["HOW", "ARE"], "")[0] == "YOU"


def test_multi_word_context(local):
    words = local.suggest_words(["I", "WANT", "TO"], "")
    assert {"GO", "LEARN"} & set(words)


def test_prefix_and_context_together(local):
    words = local.suggest_words(["THANK"], "Y")
    assert words[0] == "YOU"


def test_sentences_continue_what_was_typed(local):
    sentences = local.suggest_sentences(["HOW", "ARE"], "")
    assert sentences
    assert all(s.lower().startswith("how are") for s in sentences[:2])


def test_unknown_text_still_returns_a_punctuated_sentence(local):
    sentences = local.suggest_sentences(["ZQXJ"], "")
    assert sentences[-1] == "Zqxj."


def test_a_question_gets_a_question_mark(local):
    assert local.suggest_sentences(["WHERE", "IS", "IT"], "")[-1].endswith("?")


def test_nonsense_prefix_returns_nothing_rather_than_junk(local):
    assert local.suggest_words([], "ZQXJV") == []


def test_respects_the_limit(local):
    assert len(local.suggest_words([], "A", limit=2)) <= 2


# ======================================================================
# Orchestration
# ======================================================================


def request(text="HELLO", current_word="", context=None, **kwargs):
    return SuggestionRequest(
        text=text,
        current_word=current_word,
        context=context if context is not None else ["HELLO"],
        **kwargs,
    )


PAYLOAD = LLMSuggestionPayload(
    word_suggestions=["THERE", "AGAIN"],
    sentence_suggestions=["Hello there, how are you?"],
)


def make_service(settings, gemini) -> SuggestionService:
    return SuggestionService(settings, gemini=gemini)


async def test_local_suggestions_never_call_gemini(settings):
    gemini = StubGemini(PAYLOAD)
    service = make_service(settings, gemini)
    response = service.local_suggestions(request())
    assert response.source is SuggestionSource.LOCAL
    assert response.word_suggestions
    assert gemini.calls == 0


async def test_refinement_puts_gemini_first_and_keeps_local_as_backup(settings):
    service = make_service(settings, StubGemini(PAYLOAD))
    response = await service.refined_suggestions(request())
    assert response.source is SuggestionSource.GEMINI
    assert response.word_suggestions[0] == "THERE"
    assert "Hello there, how are you?" in response.sentence_suggestions


async def test_gemini_failure_falls_back_to_local(settings):
    service = make_service(settings, StubGemini(error=GeminiUnavailable("boom")))
    response = await service.refined_suggestions(request())
    assert response.source is SuggestionSource.LOCAL
    assert response.word_suggestions          # the user still gets suggestions
    assert response.notice


async def test_gemini_timeout_falls_back_to_local(settings):
    service = make_service(
        settings, StubGemini(error=GeminiUnavailable("Gemini timed out"))
    )
    response = await service.refined_suggestions(request())
    assert response.word_suggestions
    assert response.source is SuggestionSource.LOCAL


async def test_identical_requests_are_served_from_cache(settings):
    gemini = StubGemini(PAYLOAD)
    service = make_service(settings, gemini)
    first = await service.refined_suggestions(request())
    second = await service.refined_suggestions(request())
    assert first.source is SuggestionSource.GEMINI
    assert second.source is SuggestionSource.CACHE
    assert gemini.calls == 1


async def test_concurrent_identical_requests_make_one_call(settings):
    gemini = StubGemini(PAYLOAD)
    service = make_service(settings, gemini)
    await asyncio.gather(
        *(service.refined_suggestions(request()) for _ in range(5))
    )
    assert gemini.calls == 1


async def test_gemini_is_not_called_for_a_single_letter(settings):
    gemini = StubGemini(PAYLOAD)
    service = make_service(settings, gemini)
    assert not service.should_use_llm(request(text="A", current_word="A", context=[]))
    await service.refined_suggestions(request(text="A", current_word="A", context=[]))
    assert gemini.calls == 0


async def test_use_llm_false_is_respected(settings):
    gemini = StubGemini(PAYLOAD)
    service = make_service(settings, gemini)
    await service.refined_suggestions(request(use_llm=False))
    assert gemini.calls == 0


async def test_scheduled_refinement_pushes_through_the_callback(settings):
    service = make_service(settings, StubGemini(PAYLOAD))
    received = []

    scheduled = service.schedule_refinement(
        "session-1", request(), lambda r: _collect(received, r)
    )
    assert scheduled
    await asyncio.sleep(0.15)
    assert received and received[0].source is SuggestionSource.GEMINI
    await service.aclose()


async def test_a_newer_request_cancels_the_pending_one(settings):
    """Typing on must not leave a stale answer in flight."""
    gemini = StubGemini(PAYLOAD)
    service = make_service(settings, gemini)
    received = []

    for _ in range(5):
        service.schedule_refinement(
            "session-1", request(), lambda r: _collect(received, r)
        )
    await asyncio.sleep(0.15)

    assert gemini.calls == 1     # only the last one survived the debounce
    assert len(received) == 1
    await service.aclose()


async def _collect(sink: list, response) -> None:
    sink.append(response)


# ======================================================================
# Never trust the model's output
# ======================================================================


def context(current_word="HE", local_words=("HELLO",)):
    return SuggestionContext(
        text="HE",
        current_word=current_word,
        context=[],
        local_words=list(local_words),
        max_words=4,
        max_sentences=3,
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        (["hello", "help"], ["HELLO", "HELP"]),
        (["hello.", " help "], ["HELLO", "HELP"]),        # punctuation stripped
        (["hello", "HELLO", "Hello"], ["HELLO"]),          # de-duplicated
        (["hello", "world"], ["HELLO"]),                   # prefix enforced
        ([123, None, {"a": 1}, "help"], ["HELP"]),         # wrong types dropped
        (["hello there friend"], ["HELLO"]),               # multi-word truncated
        ([], []),
    ],
)
def test_malformed_word_lists_are_sanitised(raw, expected):
    assert GeminiSuggestionService._clean_words(raw, context()) == expected


def test_word_list_is_capped_at_max_words():
    words = ["help", "hello", "her", "here", "hers", "hey"]
    cleaned = GeminiSuggestionService._clean_words(words, context())
    assert len(cleaned) == 4


@pytest.mark.parametrize(
    "raw, expected",
    [
        (["Hello there."], ["Hello there."]),
        (["  Hello   there.  "], ["Hello there."]),   # whitespace collapsed
        (["Hi", "hi"], ["Hi"]),                        # case-insensitive dedupe
        (["x" * 500], []),                             # absurd length dropped
        ([None, 42, "Hello."], ["Hello."]),
    ],
)
def test_malformed_sentence_lists_are_sanitised(raw, expected):
    assert GeminiSuggestionService._clean_sentences(raw, context()) == expected


async def test_service_without_an_api_key_reports_unavailable(settings):
    service = GeminiSuggestionService(settings)
    assert service.available is False
    with pytest.raises(GeminiUnavailable):
        await service.suggest(context())
