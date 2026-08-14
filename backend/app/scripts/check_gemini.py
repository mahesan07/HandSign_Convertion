"""Diagnose the Gemini setup without starting the whole app.

    python -m backend.app.scripts.check_gemini
    python -m backend.app.scripts.check_gemini --list

Answers, in order, the questions that actually go wrong:

1. is a key being read from .env at all?
2. does the key work?
3. is the configured model still available? (Google retires them, and a
   retired name returns 404, which reads like a broken key but is not)
4. does a real suggestion come back, correctly shaped, and how fast?

The app never needs this to run -- without a key it uses local suggestions.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from backend.app.core.config import get_settings
from backend.app.services.gemini_service import (
    GeminiSuggestionService,
    GeminiUnavailable,
    SuggestionContext,
)

PROBE = SuggestionContext(
    text="HOW ARE",
    current_word="YO",
    context=["HOW", "ARE"],
    local_words=["YOU", "YOUR"],
    max_words=4,
    max_sentences=3,
)


def list_models(api_key: str) -> list[str]:
    from google import genai

    client = genai.Client(api_key=api_key)
    names = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            names.append(model.name.replace("models/", ""))
    return sorted(names)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="list every model the key can use"
    )
    args = parser.parse_args()

    settings = get_settings()

    print("1. API key")
    if not settings.gemini_enabled:
        print("   not set.")
        print("   The app still works -- suggestions come from the local")
        print("   engine and the UI says so. To enable Gemini, put a key from")
        print("   https://aistudio.google.com/app/apikey in .env:")
        print("       GEMINI_API_KEY=your_key_here")
        print("   (in .env, NOT .env.example -- only .env is git-ignored)")
        return 0

    key = settings.gemini_api_key.get_secret_value()
    print(f"   found: {key[:6]}... ({len(key)} chars)")

    print("\n2. Reaching the API")
    try:
        available = list_models(key)
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED: {exc}")
        print("   Check the key is valid and that you are online.")
        return 1
    print(f"   ok - {len(available)} models available to this key")

    if args.list:
        print()
        for name in available:
            print(f"     {name}")

    print(f"\n3. Configured model: {settings.gemini_model}")
    if settings.gemini_model in available:
        print("   listed as available")
    else:
        print("   NOT in the list.")
        suggestions = [n for n in available if "flash-lite" in n] or [
            n for n in available if "flash" in n
        ]
        if suggestions:
            print(f"   Try one of: {', '.join(suggestions[:5])}")
        print("   Set GEMINI_MODEL in .env, then run this again.")

    print("\n4. A real suggestion request")
    service = GeminiSuggestionService(settings)
    started = time.perf_counter()
    try:
        payload = await service.suggest(PROBE)
    except GeminiUnavailable as exc:
        print(f"   FAILED: {exc}")
        print("\n   The app will keep working on local suggestions.")
        return 1
    elapsed = (time.perf_counter() - started) * 1000

    print(f'   "{PROBE.text}" + "{PROBE.current_word}"  [{elapsed:.0f} ms]')
    print(f"   words     : {payload.word_suggestions}")
    print(f"   sentences : {payload.sentence_suggestions}")

    if not payload.word_suggestions:
        print("\n   The model answered but produced nothing usable. That is")
        print("   usually a model that ignores structured output -- try a")
        print("   different GEMINI_MODEL.")
        return 1
    if elapsed > settings.gemini_timeout_seconds * 1000:
        print(f"\n   Slower than GEMINI_TIMEOUT_SECONDS "
              f"({settings.gemini_timeout_seconds}s); raise it or pick a "
              "lighter model.")

    print("\nGemini is working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
