"""Live recognition: one websocket per camera, plus a REST fallback.

The websocket is the low-latency path.  Its shape matters:

* frames are held in a **single-slot mailbox**, so if the client outruns the
  backend we always work on the newest frame instead of building a queue that
  makes the overlay lag further behind reality with every second;
* recognition runs in a worker thread, so decoding a JPEG never blocks the
  loop that is delivering suggestions;
* Gemini refinements are *scheduled*, never awaited -- a slow or dead model
  cannot stall, slow down or stop letter recognition.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from backend.app.api.deps import AppServices, get_services, get_services_ws
from backend.app.core.logging import get_logger
from backend.app.schemas.recognition import (
    ErrorUpdate,
    PredictionModel,
    PredictRequest,
    PredictResponse,
    ReadyUpdate,
    RecognitionUpdate,
    SuggestionsUpdate,
    TextStateModel,
    TextUpdate,
)
from backend.app.schemas.session import TextCommand
from backend.app.schemas.suggestions import SuggestionRequest, SuggestionResponse
from backend.app.services.recognition_service import (
    FrameDecodeError,
    landmarks_payload,
)
from backend.app.services.session import Session
from backend.app.services.stabilizer import RecognitionStatus, StabilizerUpdate
from backend.app.services.text_commands import InvalidCommand, apply_command

logger = get_logger(__name__)

#: REST endpoints, mounted under /api.
router = APIRouter(tags=["recognition"])

#: The live channel, mounted at the root so its URL is /ws/recognition.
ws_router = APIRouter()

#: Guard against a client streaming absurd payloads over the socket.
MAX_MESSAGE_CHARS = 4_000_000


# ======================================================================
# REST
# ======================================================================


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    services: AppServices = Depends(get_services),
) -> PredictResponse:
    """Recognise one frame.

    Useful for testing and for clients that cannot hold a websocket open; the
    live UI uses ``/ws/recognition`` instead.
    """
    session = services.sessions.get_or_create(payload.session_id)
    try:
        result = await services.recognition.recognize_encoded(
            payload.image, mirrored=payload.mirrored
        )
    except FrameDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=str(exc)
        ) from exc

    if payload.apply_to_buffer:
        update = session.stabilizer.update(result.letter, result.confidence)
        session.apply(update)
    else:
        # Report only -- do not disturb the stabilizer state of a session that
        # may have a live websocket attached.
        update = StabilizerUpdate(
            status=(
                RecognitionStatus.DETECTING
                if result.hand_detected
                else RecognitionStatus.IDLE
            ),
            candidate=result.letter,
            confidence=result.confidence,
        )

    return PredictResponse(
        session_id=session.session_id,
        hand_detected=result.hand_detected,
        status=update.status.value,
        prediction=PredictionModel(
            letter=result.letter,
            confidence=round(result.confidence, 4),
            alternatives=[
                (letter, round(prob, 4)) for letter, prob in result.alternatives
            ],
        ),
        progress=round(update.progress, 3),
        committed_letter=update.committed_letter,
        landmarks=landmarks_payload(result),
        text=TextStateModel.from_state(session.buffer.snapshot()),
        latency_ms=round(result.elapsed_ms, 2),
    )


# ======================================================================
# WebSocket
# ======================================================================


class RecognitionConnection:
    """Owns one browser connection for its lifetime."""

    def __init__(
        self,
        websocket: WebSocket,
        services: AppServices,
        session: Session,
    ) -> None:
        self._ws = websocket
        self._services = services
        self._session = session
        self._frame: Optional[tuple[str, bool]] = None   # the single slot
        self._frame_ready = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._dropped_frames = 0

    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self._send(
            ReadyUpdate(
                session_id=self._session.session_id,
                classes=self._services.recognition.classes,
                gemini_enabled=self._services.suggestions.gemini.available,
            )
        )
        # Give the user something to pick from before they sign anything.
        await self._push_suggestions()

        worker = asyncio.create_task(self._process_frames())
        try:
            await self._receive_loop()
        finally:
            self._closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            self._services.suggestions.cancel(self._session.session_id)
            if self._dropped_frames:
                logger.debug(
                    "session %s dropped %d stale frames",
                    self._session.session_id,
                    self._dropped_frames,
                )

    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        while True:
            raw = await self._ws.receive_text()
            if len(raw) > MAX_MESSAGE_CHARS:
                await self._send_error("message_too_large", "Frame is too large.")
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error("bad_json", "Message was not valid JSON.")
                continue
            if not isinstance(message, dict):
                await self._send_error("bad_message", "Message must be an object.")
                continue
            await self._dispatch(message)

    async def _dispatch(self, message: dict) -> None:
        kind = message.get("type")

        if kind == "frame":
            image = message.get("image")
            if not isinstance(image, str) or not image:
                await self._send_error("bad_frame", "Frame had no image data.")
                return
            if self._frame is not None:
                self._dropped_frames += 1  # newest frame wins
            self._frame = (image, bool(message.get("mirrored", True)))
            self._frame_ready.set()
            return

        if kind == "command":
            await self._handle_command(message)
            return

        if kind == "ping":
            await self._send_raw({"type": "pong"})
            return

        await self._send_error("unknown_type", f"Unsupported message type: {kind!r}")

    async def _handle_command(self, message: dict) -> None:
        try:
            command = TextCommand(message.get("command"))
        except ValueError:
            await self._send_error(
                "unknown_command", f"Unsupported command: {message.get('command')!r}"
            )
            return

        try:
            state = apply_command(
                self._session, command, str(message.get("value", ""))
            )
        except InvalidCommand as exc:
            await self._send_error("invalid_command", str(exc))
            return

        await self._send(
            TextUpdate(
                session_id=self._session.session_id,
                text=TextStateModel.from_state(state),
            )
        )
        await self._push_suggestions()

    # ------------------------------------------------------------------

    async def _process_frames(self) -> None:
        """Recognise the most recent frame, forever."""
        while not self._closed:
            await self._frame_ready.wait()
            self._frame_ready.clear()
            pending, self._frame = self._frame, None
            if pending is None:
                continue

            image, mirrored = pending
            try:
                result = await self._services.recognition.recognize_encoded(
                    image, mirrored=mirrored
                )
            except FrameDecodeError as exc:
                await self._send_error("bad_frame", str(exc))
                continue
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad frame must not kill the socket
                logger.exception("Recognition failed for one frame")
                await self._send_error(
                    "recognition_failed", "That frame could not be processed."
                )
                continue

            # Everything below is inside the guard too. A failure here -- a
            # schema mismatch, a serialisation error -- would otherwise kill
            # this task silently and the socket would simply stop responding
            # forever, which is far harder to diagnose than a dropped frame.
            try:
                update = self._session.stabilizer.update(
                    result.letter, result.confidence
                )
                committed_state = self._session.apply(update)

                await self._send(
                    RecognitionUpdate(
                        session_id=self._session.session_id,
                        hand_detected=result.hand_detected,
                        status=update.status.value,
                        prediction=PredictionModel(
                            letter=result.letter,
                            confidence=round(result.confidence, 4),
                            alternatives=[
                                (letter, round(prob, 4))
                                for letter, prob in result.alternatives
                            ],
                        ),
                        progress=round(update.progress, 3),
                        stable_count=round(update.stable_count, 3),
                        required_frames=update.required_frames,
                        committed_letter=update.committed_letter,
                        landmarks=landmarks_payload(result),
                        text=TextStateModel.from_state(
                            self._session.buffer.snapshot()
                        ),
                        latency_ms=round(result.elapsed_ms, 2),
                    )
                )

                # Only a *new letter* is worth re-suggesting for, not every frame.
                if committed_state is not None:
                    await self._push_suggestions()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("Failed to publish a recognition update")
                await self._send_error(
                    "recognition_failed", "That frame could not be reported."
                )

    # ------------------------------------------------------------------

    async def _push_suggestions(self) -> None:
        """Send local suggestions now; queue a Gemini refinement for later."""
        state = self._session.buffer.snapshot()
        request = SuggestionRequest(
            text=state.text,
            current_word=state.current_word,
            context=state.words,
            max_words=self._services.settings.max_word_suggestions,
            max_sentences=self._services.settings.max_sentence_suggestions,
        )

        instant = self._services.suggestions.local_suggestions(request)
        await self._send(
            SuggestionsUpdate(
                session_id=self._session.session_id,
                suggestions=instant,
                text=TextStateModel.from_state(state),
            )
        )

        self._services.suggestions.schedule_refinement(
            self._session.session_id, request, self._on_refined
        )

    async def _on_refined(self, refined: SuggestionResponse) -> None:
        if self._closed:
            return
        await self._send(
            SuggestionsUpdate(
                session_id=self._session.session_id,
                suggestions=refined,
                text=TextStateModel.from_state(self._session.buffer.snapshot()),
            )
        )

    # ------------------------------------------------------------------

    async def _send(self, model) -> None:
        await self._send_raw(model.model_dump(mode="json"))

    async def _send_raw(self, payload: dict) -> None:
        if self._closed:
            return
        try:
            async with self._send_lock:
                await self._ws.send_text(json.dumps(payload))
        except (WebSocketDisconnect, RuntimeError):
            self._closed = True

    async def _send_error(self, code: str, message: str, fatal: bool = False) -> None:
        await self._send(ErrorUpdate(code=code, message=message, fatal=fatal))


@ws_router.websocket("/ws/recognition")
async def recognition_socket(
    websocket: WebSocket,
    session_id: str | None = None,
    services: AppServices = Depends(get_services_ws),
) -> None:
    await websocket.accept()
    session = services.sessions.get_or_create(session_id)
    connection = RecognitionConnection(websocket, services, session)
    try:
        await connection.run()
    except WebSocketDisconnect:
        pass
    except ValidationError:
        logger.exception("Invalid websocket payload")
    except Exception:  # noqa: BLE001
        logger.exception("Recognition socket failed")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except RuntimeError:
            pass
