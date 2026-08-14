"""Session lifecycle and text editing over REST.

The websocket carries these same commands; these endpoints exist so the text
state is reachable without a socket (and so it is easy to test).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.api.deps import AppServices, get_services
from backend.app.schemas.recognition import TextStateModel
from backend.app.schemas.session import SessionResponse, TextCommandRequest
from backend.app.services.text_commands import InvalidCommand, apply_command

router = APIRouter(prefix="/session", tags=["session"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    services: AppServices = Depends(get_services),
) -> SessionResponse:
    session = services.sessions.create()
    return SessionResponse(
        session_id=session.session_id,
        text=TextStateModel.from_state(session.buffer.snapshot()),
    )


@router.post("/reset", response_model=SessionResponse)
async def reset_session(
    payload: TextCommandRequest | None = None,
    services: AppServices = Depends(get_services),
) -> SessionResponse:
    """Clear the sentence.  Creates the session first if it does not exist."""
    session_id = payload.session_id if payload else None
    session = services.sessions.get_or_create(session_id)
    state = session.reset()
    services.suggestions.cancel(session.session_id)
    return SessionResponse(
        session_id=session.session_id, text=TextStateModel.from_state(state)
    )


@router.post("/text", response_model=SessionResponse)
async def text_command(
    payload: TextCommandRequest,
    services: AppServices = Depends(get_services),
) -> SessionResponse:
    session = services.sessions.get_or_create(payload.session_id)
    try:
        state = apply_command(session, payload.command, payload.value)
    except InvalidCommand as exc:
        raise HTTPException(
            status_code=422, detail=str(exc)
        ) from exc
    return SessionResponse(
        session_id=session.session_id, text=TextStateModel.from_state(state)
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    services: AppServices = Depends(get_services),
) -> SessionResponse:
    session = services.sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session"
        )
    return SessionResponse(
        session_id=session.session_id,
        text=TextStateModel.from_state(session.buffer.snapshot()),
    )
