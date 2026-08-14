"""English -> hand signs."""

from __future__ import annotations

import json

import pytest

from backend.app.services.sign_translation import SignTranslator, TokenKind
from ml.paths import SIGN_ASSETS_DIR

ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


@pytest.fixture
def translator() -> SignTranslator:
    return SignTranslator(ALPHABET)


def test_letters_map_to_assets(translator):
    translation = translator.translate("HELLO")
    tokens = translation.words[0].signs
    assert [t.character for t in tokens] == list("HELLO")
    assert all(t.kind is TokenKind.LETTER for t in tokens)
    assert tokens[0].asset == "/signs/H.svg"


def test_lowercase_input_is_handled(translator):
    assert translator.translate("hi").words[0].signs[0].character == "H"


def test_words_are_grouped_and_spaces_do_not_become_tiles(translator):
    translation = translator.translate("HELLO HOW ARE YOU")
    assert [word.text for word in translation.words] == [
        "HELLO", "HOW", "ARE", "YOU",
    ]
    assert translation.sign_count == len("HELLOHOWAREYOU")


def test_punctuation_is_kept_but_has_no_asset(translator):
    tokens = translator.translate("Hi!").words[0].signs
    assert tokens[-1].kind is TokenKind.PUNCTUATION
    assert tokens[-1].asset is None


def test_digits_are_marked_as_digits(translator):
    tokens = translator.translate("A1").words[0].signs
    assert tokens[1].kind is TokenKind.DIGIT


def test_unsupported_characters_are_reported_once(translator):
    translation = translator.translate("CAFÉ ÉCLAIR")
    assert translation.unsupported == ["É"]
    flat = [t for word in translation.words for t in word.signs]
    assert any(t.kind is TokenKind.UNSUPPORTED for t in flat)


def test_a_letter_the_model_does_not_know_has_no_sign():
    """The catalog follows the model, not the alphabet."""
    limited = SignTranslator(["A", "B", "C"])
    translation = limited.translate("ABZ")
    assert translation.words[0].signs[2].kind is TokenKind.UNSUPPORTED
    assert limited.supported_letters == ["A", "B", "C"]


def test_empty_text_is_empty(translator):
    translation = translator.translate("   ")
    assert translation.words == []
    assert translation.sign_count == 0


def test_every_token_has_a_label_for_screen_readers(translator):
    for word in translator.translate("Hi 1!").words:
        for token in word.signs:
            assert token.label


def test_catalog_covers_every_supported_letter(translator):
    catalog = translator.catalog()
    assert sorted(catalog) == ALPHABET
    assert all(path.endswith(".svg") for path in catalog.values())


# ----------------------------------------------------------------------
# The generated assets themselves
# ----------------------------------------------------------------------


def test_an_asset_file_exists_for_every_letter():
    missing = [
        letter for letter in ALPHABET
        if not (SIGN_ASSETS_DIR / f"{letter}.svg").exists()
    ]
    assert not missing, (
        f"missing sign artwork for {missing}; "
        "run `python -m ml.scripts.generate_sign_assets`"
    )


def test_assets_are_self_contained_svg():
    """No external references, so the UI works offline."""
    for letter in ALPHABET:
        content = (SIGN_ASSETS_DIR / f"{letter}.svg").read_text(encoding="utf-8")
        assert content.startswith("<svg")
        assert f"aria-label" in content
        assert "http://www.w3.org/2000/svg" in content
        assert "<image" not in content


def test_manifest_matches_the_files_on_disk():
    manifest = json.loads(
        (SIGN_ASSETS_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert sorted(manifest["signs"]) == ALPHABET
