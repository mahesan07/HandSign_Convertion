"""Per-user recognition state.

A session bundles the two things that must stay in step -- the stabilizer that
decides when a letter is real, and the text buffer that letter goes into -- so
that the websocket, the REST endpoints and the tests all mutate the same
state through one door.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.app.core.config import Settings
from backend.app.services.stabilizer import (
    PredictionStabilizer,
    StabilizerConfig,
    StabilizerUpdate,
)
from backend.app.services.text_buffer import TextBuffer, TextState


@dataclass
class Session:
    """One user's conversation state."""

    session_id: str
    buffer: TextBuffer
    stabilizer: PredictionStabilizer
    created_at: float
    last_seen_at: float

    def touch(self) -> None:
        self.last_seen_at = time.time()

    def reset(self) -> TextState:
        """Clear the sentence and forget any half-held sign."""
        self.stabilizer.reset()
        return self.buffer.clear()

    def apply(self, update: StabilizerUpdate) -> Optional[TextState]:
        """Commit a stabilized letter, if there is one."""
        if update.committed_letter is None:
            return None
        return self.buffer.add_character(update.committed_letter)


class SessionStore:
    """In-memory session registry with a TTL.

    Deliberately not a database: this is a single-user desktop-style app and
    the state is worthless once the tab closes.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def _stabilizer_config(self) -> StabilizerConfig:
        s = self._settings
        return StabilizerConfig(
            min_confidence=s.min_confidence,
            partial_confidence=s.partial_confidence,
            stable_frames=s.stable_frames,
            decay=s.prediction_decay,
            commit_cooldown_ms=s.commit_cooldown_ms,
            duplicate_suppression_ms=s.duplicate_suppression_ms,
            release_frames=s.release_frames,
        )

    def create(self, session_id: str | None = None) -> Session:
        now = time.time()
        session = Session(
            session_id=session_id or uuid.uuid4().hex,
            buffer=TextBuffer(),
            stabilizer=PredictionStabilizer(self._stabilizer_config()),
            created_at=now,
            last_seen_at=now,
        )
        with self._lock:
            self._prune_locked()
            if len(self._sessions) >= self._settings.max_sessions:
                oldest = min(
                    self._sessions.values(), key=lambda s: s.last_seen_at
                )
                self._sessions.pop(oldest.session_id, None)
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.touch()
            return session

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        return self.create(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def active_ids(self) -> List[str]:
        with self._lock:
            return list(self._sessions)

    def _prune_locked(self) -> None:
        cutoff = time.time() - self._settings.session_ttl_seconds
        for sid in [
            sid for sid, s in self._sessions.items() if s.last_seen_at < cutoff
        ]:
            self._sessions.pop(sid, None)


__all__ = ["Session", "SessionStore"]
