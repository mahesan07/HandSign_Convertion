"""English text -> a sequence of hand-sign tokens.

The service knows nothing about image files beyond one base URL, so swapping
the illustrations (for photographs, say) is a change to
``ml/scripts/generate_sign_assets.py`` and this one constant -- never to the
frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence

#: Where the generated assets are served from, relative to the web root.
SIGN_ASSET_BASE = "/signs"


class TokenKind(str, Enum):
    LETTER = "letter"              # has an illustration
    SPACE = "space"                # word break
    DIGIT = "digit"                # shown as a character tile
    PUNCTUATION = "punctuation"    # shown as a character tile
    UNSUPPORTED = "unsupported"    # no sign, shown with a clear notice


@dataclass(frozen=True, slots=True)
class SignToken:
    character: str
    kind: TokenKind
    #: URL of the illustration, or ``None`` for anything not fingerspelled.
    asset: str | None
    #: Screen-reader text.
    label: str


@dataclass(frozen=True, slots=True)
class SignWord:
    text: str
    signs: List[SignToken]


@dataclass(frozen=True, slots=True)
class SignTranslation:
    text: str
    words: List[SignWord]
    #: Distinct characters we had no sign for, so the UI can say so once.
    unsupported: List[str]

    @property
    def sign_count(self) -> int:
        return sum(
            1 for word in self.words for token in word.signs
            if token.kind is TokenKind.LETTER
        )


class SignTranslator:
    def __init__(
        self,
        supported_letters: Sequence[str],
        asset_base: str = SIGN_ASSET_BASE,
    ) -> None:
        self._asset_base = asset_base.rstrip("/")
        self._assets: Dict[str, str] = {
            letter.upper(): f"{self._asset_base}/{letter.upper()}.svg"
            for letter in supported_letters
        }

    @property
    def supported_letters(self) -> List[str]:
        return sorted(self._assets)

    def asset_for(self, letter: str) -> str | None:
        return self._assets.get(letter.upper())

    def catalog(self) -> Dict[str, str]:
        """The full letter -> asset mapping, served to the frontend."""
        return dict(sorted(self._assets.items()))

    # ------------------------------------------------------------------

    def translate(self, text: str) -> SignTranslation:
        words: List[SignWord] = []
        unsupported: List[str] = []

        for raw_word in text.split():
            tokens: List[SignToken] = []
            for character in raw_word:
                token = self._token(character)
                if token.kind is TokenKind.UNSUPPORTED:
                    upper = character.upper()
                    if upper not in unsupported:
                        unsupported.append(upper)
                tokens.append(token)
            if tokens:
                words.append(SignWord(text=raw_word, signs=tokens))

        return SignTranslation(text=text, words=words, unsupported=unsupported)

    def _token(self, character: str) -> SignToken:
        upper = character.upper()
        asset = self._assets.get(upper)
        if asset is not None:
            return SignToken(
                character=upper,
                kind=TokenKind.LETTER,
                asset=asset,
                label=f"Letter {upper}",
            )
        if character.isspace():
            return SignToken(character=" ", kind=TokenKind.SPACE, asset=None, label="Space")
        if character.isdigit():
            return SignToken(
                character=character,
                kind=TokenKind.DIGIT,
                asset=None,
                label=f"Number {character}",
            )
        if not character.isalnum():
            return SignToken(
                character=character,
                kind=TokenKind.PUNCTUATION,
                asset=None,
                label=f"Punctuation {character}",
            )
        return SignToken(
            character=upper,
            kind=TokenKind.UNSUPPORTED,
            asset=None,
            label=f"No sign available for {upper}",
        )


__all__ = [
    "SignTranslator",
    "SignTranslation",
    "SignWord",
    "SignToken",
    "TokenKind",
    "SIGN_ASSET_BASE",
]
