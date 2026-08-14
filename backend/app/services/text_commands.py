"""One implementation of the editing commands, shared by REST and websocket.

Both transports funnel through :func:`apply_command`, so a keyboard shortcut,
an on-screen button and a future voice command can never drift apart.
"""

from __future__ import annotations

from backend.app.schemas.session import TextCommand
from backend.app.services.session import Session
from backend.app.services.text_buffer import TextState


class InvalidCommand(ValueError):
    """The command needs a value it was not given."""


def apply_command(
    session: Session, command: TextCommand, value: str = ""
) -> TextState:
    if command is TextCommand.SPACE:
        return session.buffer.space()

    if command is TextCommand.BACKSPACE:
        return session.buffer.backspace()

    if command is TextCommand.DELETE_WORD:
        return session.buffer.delete_word()

    if command is TextCommand.CLEAR:
        # Also forget a half-held sign, so the next letter starts clean.
        return session.reset()

    if command is TextCommand.ADD_CHARACTER:
        if len(value) != 1:
            raise InvalidCommand("add_character needs exactly one character")
        return session.buffer.add_character(value)

    if command is TextCommand.ACCEPT_WORD:
        if not value.strip():
            raise InvalidCommand("accept_word needs a word")
        # The user picked this word deliberately, so stop tracking the pose
        # that was being held for the letter they no longer need to spell.
        session.stabilizer.reset()
        return session.buffer.accept_word(value)

    if command is TextCommand.ACCEPT_SENTENCE:
        if not value.strip():
            raise InvalidCommand("accept_sentence needs a sentence")
        session.stabilizer.reset()
        return session.buffer.accept_sentence(value)

    if command is TextCommand.SET_TEXT:
        session.stabilizer.reset()
        return session.buffer.set_text(value)

    raise InvalidCommand(f"unknown command: {command}")


__all__ = ["apply_command", "InvalidCommand"]
