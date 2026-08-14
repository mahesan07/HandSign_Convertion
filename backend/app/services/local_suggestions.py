"""Instant, offline suggestions.

This engine answers in well under a millisecond and never touches the network,
so the UI always has something to show: it is both the first response the user
sees and the fallback when Gemini is slow, rate-limited or not configured.

It combines three cheap signals:

* **prefix completion** over a frequency-ordered common-word list;
* **n-gram continuation** (trigram, then bigram) learned from a small corpus
  of everyday sentences, so "HOW ARE" really does suggest "YOU";
* **sentence matching** against that same corpus, plus a punctuated version of
  whatever the user has actually typed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WORDS_PATH = _DATA_DIR / "words.txt"
PHRASES_PATH = _DATA_DIR / "phrases.txt"

_TOKEN_RE = re.compile(r"[a-z']+")
_QUESTION_STARTERS = {
    "how", "what", "where", "when", "why", "who", "which", "whose",
    "can", "could", "would", "should", "do", "does", "did", "is",
    "are", "was", "were", "will", "may", "am",
}

# Relative weight of each signal.  Context beats raw frequency, and short
# words win ties -- that is how a predictive keyboard is expected to behave.
_TRIGRAM_WEIGHT = 12.0
_BIGRAM_WEIGHT = 6.0
_FREQUENCY_WEIGHT = 1.0
_LENGTH_PENALTY = 0.06


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True, slots=True)
class LocalSuggestions:
    words: List[str]
    sentences: List[str]


class LocalSuggestionEngine:
    """Loads its data once at startup; every query after that is pure lookup."""

    def __init__(
        self,
        words_path: Path = WORDS_PATH,
        phrases_path: Path = PHRASES_PATH,
    ) -> None:
        self._sentences: List[str] = _read_lines(phrases_path)
        self._sentence_tokens: List[List[str]] = [
            _tokenize(s) for s in self._sentences
        ]

        # --- unigram frequency ------------------------------------------
        vocabulary = _read_lines(words_path)
        self._frequency: Dict[str, float] = {}
        total = max(len(vocabulary), 1)
        for rank, word in enumerate(vocabulary):
            word = word.lower()
            # Smooth 1/rank so the head of the list does not dwarf the tail.
            self._frequency.setdefault(word, (total - rank) / total)

        # Words that only appear in the phrase corpus still deserve a score.
        for tokens in self._sentence_tokens:
            for word in tokens:
                self._frequency.setdefault(word, 0.15)

        # --- prefix index (a two-character bucketed trie) -----------------
        self._by_prefix: Dict[str, List[str]] = defaultdict(list)
        for word in sorted(
            self._frequency, key=lambda w: -self._frequency[w]
        ):
            self._by_prefix[word[:1]].append(word)
            if len(word) >= 2:
                self._by_prefix[word[:2]].append(word)

        # --- n-gram continuations ----------------------------------------
        self._bigram: Dict[str, Counter] = defaultdict(Counter)
        self._trigram: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self._starters: Counter = Counter()
        for tokens in self._sentence_tokens:
            if tokens:
                self._starters[tokens[0]] += 1
            for i in range(len(tokens) - 1):
                self._bigram[tokens[i]][tokens[i + 1]] += 1
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i + 1])
                self._trigram[key][tokens[i + 2]] += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vocabulary_size(self) -> int:
        return len(self._frequency)

    def suggest(
        self,
        context: Sequence[str],
        current_word: str,
        *,
        max_words: int = 4,
        max_sentences: int = 3,
    ) -> LocalSuggestions:
        """Suggest completions for ``current_word`` given the preceding words."""
        return LocalSuggestions(
            words=self.suggest_words(context, current_word, limit=max_words),
            sentences=self.suggest_sentences(
                context, current_word, limit=max_sentences
            ),
        )

    def suggest_words(
        self,
        context: Sequence[str],
        current_word: str,
        *,
        limit: int = 4,
    ) -> List[str]:
        prefix = current_word.strip().lower()
        history = [w.lower() for w in context if w.strip()]

        if not prefix:
            return self._next_words(history, limit)

        candidates = self._candidates_for_prefix(prefix)
        if not candidates:
            return []

        trigram_key = tuple(history[-2:]) if len(history) >= 2 else None
        trigram = self._trigram.get(trigram_key, Counter()) if trigram_key else Counter()
        bigram = self._bigram.get(history[-1], Counter()) if history else Counter()

        scored: List[Tuple[float, str]] = []
        for word in candidates:
            score = _FREQUENCY_WEIGHT * self._frequency.get(word, 0.05)
            if word in trigram:
                score += _TRIGRAM_WEIGHT * trigram[word]
            if word in bigram:
                score += _BIGRAM_WEIGHT * bigram[word]
            score -= _LENGTH_PENALTY * len(word)
            if word == prefix:
                # Exact match: useful confirmation, but never the only option.
                score += 0.4
            scored.append((score, word))

        scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        return [word.upper() for _, word in scored[:limit]]

    def suggest_sentences(
        self,
        context: Sequence[str],
        current_word: str,
        *,
        limit: int = 3,
    ) -> List[str]:
        history = [w.lower() for w in context if w.strip()]
        prefix = current_word.strip().lower()
        if not history and not prefix:
            return self._sentences[:limit]

        matches: List[str] = []
        for sentence, tokens in zip(self._sentences, self._sentence_tokens):
            if self._matches_prefix(tokens, history, prefix):
                matches.append(sentence)
            if len(matches) >= limit:
                break

        if len(matches) < limit:
            for sentence, tokens in zip(self._sentences, self._sentence_tokens):
                if sentence in matches:
                    continue
                if history and self._contains_sequence(tokens, history):
                    matches.append(sentence)
                if len(matches) >= limit:
                    break

        completed = self._punctuate(history, prefix)
        if completed and completed not in matches:
            matches.append(completed)
        return matches[:limit]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _candidates_for_prefix(self, prefix: str) -> List[str]:
        bucket = self._by_prefix.get(prefix[:2] if len(prefix) >= 2 else prefix[:1], [])
        if len(prefix) <= 2:
            return bucket
        return [word for word in bucket if word.startswith(prefix)]

    def _next_words(self, history: Sequence[str], limit: int) -> List[str]:
        if not history:
            return [w.upper() for w, _ in self._starters.most_common(limit)]

        ordered: List[str] = []
        if len(history) >= 2:
            key = (history[-2], history[-1])
            ordered += [w for w, _ in self._trigram.get(key, Counter()).most_common(limit)]
        ordered += [
            w for w, _ in self._bigram.get(history[-1], Counter()).most_common(limit)
        ]

        seen: set[str] = set()
        result: List[str] = []
        for word in ordered:
            if word not in seen:
                seen.add(word)
                result.append(word.upper())
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _matches_prefix(
        tokens: Sequence[str], history: Sequence[str], prefix: str
    ) -> bool:
        """True when the sentence starts with the words typed so far."""
        if len(tokens) < len(history) + (1 if prefix else 0):
            return False
        if list(tokens[: len(history)]) != list(history):
            return False
        if prefix:
            return tokens[len(history)].startswith(prefix)
        return True

    @staticmethod
    def _contains_sequence(tokens: Sequence[str], needle: Sequence[str]) -> bool:
        if not needle or len(needle) > len(tokens):
            return False
        window = len(needle)
        return any(
            list(tokens[i : i + window]) == list(needle)
            for i in range(len(tokens) - window + 1)
        )

    @staticmethod
    def _punctuate(history: Sequence[str], prefix: str) -> str:
        """Turn the raw typed text into a presentable sentence."""
        words = [*history, prefix] if prefix else list(history)
        if not words:
            return ""
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:]
        ending = "?" if words[0] in _QUESTION_STARTERS else "."
        return sentence + ending


__all__ = ["LocalSuggestionEngine", "LocalSuggestions"]
