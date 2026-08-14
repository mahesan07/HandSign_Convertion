"""Word and sentence suggestions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import AppServices, get_services
from backend.app.schemas.suggestions import SuggestionRequest, SuggestionResponse

router = APIRouter(tags=["suggestions"])


@router.post("/suggestions", response_model=SuggestionResponse)
async def suggestions(
    payload: SuggestionRequest,
    services: AppServices = Depends(get_services),
    wait_for_llm: bool = Query(
        default=True,
        description=(
            "Await the Gemini refinement. Set false for a guaranteed-instant "
            "local answer; the websocket uses the push model instead."
        ),
    ),
) -> SuggestionResponse:
    """Suggest completions for the text the user is building.

    With ``wait_for_llm=false`` this never touches the network, so it is safe
    to call on every keystroke.
    """
    if not wait_for_llm:
        return services.suggestions.local_suggestions(payload)
    return await services.suggestions.refined_suggestions(payload)
