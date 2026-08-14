"""The text buffer and the editing commands built on it."""

from __future__ import annotations

import pytest

from backend.app.core.config import Settings
from backend.app.schemas.session import TextCommand
from backend.app.services.session import SessionStore
from backend.app.services.text_buffer import TextBuffer
from backend.app.services.text_commands import InvalidCommand, apply_command


@pytest.fixture
def buffer() -> TextBuffer:
    return TextBuffer()


def spell(buffer: TextBuffer, word: str) -> None:
    for character in word:
        buffer.add_character(character)


# ----------------------------------------------------------------------
# Building text
# ----------------------------------------------------------------------


def test_starts_empty(buffer):
    state = buffer.snapshot()
    assert state.text == ""
    assert state.is_empty


def test_spelling_builds_a_word(buffer):
    spell(buffer, "HELLO")
    assert buffer.text == "HELLO"
    assert buffer.current_word == "HELLO"
    assert buffer.words == []


def test_space_completes_the_word(buffer):
    spell(buffer, "HELLO")
    buffer.space()
    assert buffer.words == ["HELLO"]
    assert buffer.current_word == ""
    assert buffer.text == "HELLO"


def test_building_a_sentence(buffer):
    for word in ["HELLO", "HOW", "ARE"]:
        spell(buffer, word)
        buffer.space()
    spell(buffer, "YOU")
    assert buffer.text == "HELLO HOW ARE YOU"
    assert buffer.current_word == "YOU"


def test_characters_are_stored_uppercase(buffer):
    buffer.add_character("h")
    assert buffer.text == "H"


def test_repeated_space_does_not_create_a_gap(buffer):
    spell(buffer, "HI")
    buffer.space()
    buffer.space()
    buffer.space()
    assert buffer.text == "HI"
    assert buffer.words == ["HI"]


def test_add_character_rejects_a_string(buffer):
    with pytest.raises(ValueError):
        buffer.add_character("AB")


# ----------------------------------------------------------------------
# Deleting
# ----------------------------------------------------------------------


def test_backspace_removes_one_letter(buffer):
    spell(buffer, "HELLO")
    buffer.backspace()
    assert buffer.text == "HELL"


def test_backspace_reopens_the_previous_word(buffer):
    spell(buffer, "HELLO")
    buffer.space()
    buffer.backspace()
    assert buffer.words == []
    assert buffer.current_word == "HELLO"
    assert buffer.text == "HELLO"


def test_backspace_on_empty_is_harmless(buffer):
    buffer.backspace()
    assert buffer.text == ""


def test_delete_word_drops_the_word_being_spelled(buffer):
    spell(buffer, "HELLO")
    buffer.space()
    spell(buffer, "WOR")
    buffer.delete_word()
    assert buffer.text == "HELLO"


def test_delete_word_then_drops_the_last_completed_word(buffer):
    spell(buffer, "HELLO")
    buffer.space()
    buffer.delete_word()
    assert buffer.text == ""


def test_clear_empties_everything(buffer):
    spell(buffer, "HELLO")
    buffer.space()
    spell(buffer, "THERE")
    buffer.clear()
    assert buffer.snapshot().is_empty


# ----------------------------------------------------------------------
# Suggestions
# ----------------------------------------------------------------------


def test_accepting_a_word_replaces_what_was_spelled(buffer):
    spell(buffer, "HE")
    buffer.accept_word("hello")
    assert buffer.words == ["HELLO"]
    assert buffer.current_word == ""


def test_accepting_a_sentence_keeps_its_casing(buffer):
    spell(buffer, "HOW")
    buffer.accept_sentence("How are you?")
    assert buffer.text == "How are you?"


def test_set_text_leaves_the_last_word_open(buffer):
    buffer.set_text("hello how are yo")
    assert buffer.words == ["HELLO", "HOW", "ARE"]
    assert buffer.current_word == "YO"


def test_set_text_with_a_trailing_space_closes_the_last_word(buffer):
    buffer.set_text("hello how ")
    assert buffer.words == ["HELLO", "HOW"]
    assert buffer.current_word == ""


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


@pytest.fixture
def session():
    return SessionStore(Settings(_env_file=None)).create()


def test_commands_cover_the_whole_editing_flow(session):
    for character in "HELLO":
        apply_command(session, TextCommand.ADD_CHARACTER, character)
    apply_command(session, TextCommand.SPACE)
    for character in "WORLD":
        apply_command(session, TextCommand.ADD_CHARACTER, character)
    assert session.buffer.text == "HELLO WORLD"

    apply_command(session, TextCommand.BACKSPACE)
    assert session.buffer.text == "HELLO WORL"

    apply_command(session, TextCommand.DELETE_WORD)
    assert session.buffer.text == "HELLO"

    state = apply_command(session, TextCommand.CLEAR)
    assert state.is_empty


def test_clear_also_resets_the_stabilizer(session):
    session.stabilizer.update("A", 0.95)
    apply_command(session, TextCommand.CLEAR)
    update = session.stabilizer.update("A", 0.95)
    assert update.stable_count == 1


@pytest.mark.parametrize(
    "command",
    [TextCommand.ADD_CHARACTER, TextCommand.ACCEPT_WORD, TextCommand.ACCEPT_SENTENCE],
)
def test_commands_that_need_a_value_say_so(session, command):
    with pytest.raises(InvalidCommand):
        apply_command(session, command, "")
