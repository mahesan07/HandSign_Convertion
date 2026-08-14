"""The authoritative text state for a session.

The UI never owns the text -- it renders whatever this buffer reports.  That
keeps the websocket stream, the REST endpoints and the suggestion engine all
looking at exactly the same state.

Fingerspelled input is stored uppercase, because that is how the letters are
recognised and how they read back to the user.  A sentence suggestion, which
is finished natural language, is stored with its own casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True)
class TextState:
    """An immutable snapshot of the buffer, safe to serialise straight out."""

    text: str            # everything, e.g. "HELLO HOW ARE YO"
    words: List[str]     # completed words, e.g. ["HELLO", "HOW", "ARE"]
    current_word: str    # the word being spelled, e.g. "YO"

    @property
    def is_empty(self) -> bool:
        return not self.words and not self.current_word


class TextBuffer:
    """Characters in, sentences out."""

    def __init__(self) -> None:
        self._words: List[str] = []
        self._current: str = ""

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @property
    def words(self) -> List[str]:
        return list(self._words)

    @property
    def current_word(self) -> str:
        return self._current

    @property
    def text(self) -> str:
        parts = list(self._words)
        if self._current:
            parts.append(self._current)
        return " ".join(parts)

    def snapshot(self) -> TextState:
        return TextState(
            text=self.text,
            words=list(self._words),
            current_word=self._current,
        )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_character(self, character: str) -> TextState:
        """Append one recognised character to the word being spelled."""
        if len(character) != 1:
            raise ValueError("add_character expects exactly one character")
        if character.isspace():
            return self.space()
        self._current += character.upper()
        return self.snapshot()

    def space(self) -> TextState:
        """Finish the current word.  A no-op when there is nothing to finish,
        so a stray space can never produce a double gap."""
        if self._current:
            self._words.append(self._current)
            self._current = ""
        return self.snapshot()

    def backspace(self) -> TextState:
        """Delete one character, exactly like a keyboard: once the current
        word is empty this removes the preceding space and re-opens the
        previous word for editing."""
        if self._current:
            self._current = self._current[:-1]
        elif self._words:
            self._current = self._words.pop()
        return self.snapshot()

    def delete_word(self) -> TextState:
        """Drop the word being spelled, or the last completed one."""
        if self._current:
            self._current = ""
        elif self._words:
            self._words.pop()
        return self.snapshot()

    def clear(self) -> TextState:
        """Erase the whole sentence."""
        self._words.clear()
        self._current = ""
        return self.snapshot()

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def accept_word(self, word: str) -> TextState:
        """Replace the word being spelled with a suggestion and complete it."""
        word = word.strip()
        if not word:
            return self.snapshot()
        self._words.append(word.upper())
        self._current = ""
        return self.snapshot()

    def accept_sentence(self, sentence: str) -> TextState:
        """Replace everything with a finished sentence, keeping its casing."""
        return self.set_text(sentence, preserve_case=True)

    def set_text(self, text: str, *, preserve_case: bool = False) -> TextState:
        """Replace the buffer contents.

        A trailing space means "the last word is finished"; without one, the
        last token stays open for more letters.
        """
        normalised = text if preserve_case else text.upper()
        tokens = normalised.split()
        if not tokens:
            return self.clear()
        if normalised[-1:].isspace():
            self._words = tokens
            self._current = ""
        else:
            self._words = tokens[:-1]
            self._current = tokens[-1]
        return self.snapshot()


__all__ = ["TextBuffer", "TextState"]
