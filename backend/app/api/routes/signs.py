"""English -> sign language."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import AppServices, get_services
from backend.app.schemas.signs import (
    SignCatalogResponse,
    SignTokenModel,
    SignWordModel,
    TranslateToSignRequest,
    TranslateToSignResponse,
)
from backend.app.services.sign_translation import SIGN_ASSET_BASE

router = APIRouter(tags=["signs"])


@router.get("/signs", response_model=SignCatalogResponse)
async def sign_catalog(
    services: AppServices = Depends(get_services),
) -> SignCatalogResponse:
    """The letter -> illustration mapping.  The frontend hard-codes no paths."""
    return SignCatalogResponse(
        asset_base=SIGN_ASSET_BASE,
        signs=services.signs.catalog(),
    )


@router.post("/translate-to-sign", response_model=TranslateToSignResponse)
async def translate_to_sign(
    payload: TranslateToSignRequest,
    services: AppServices = Depends(get_services),
) -> TranslateToSignResponse:
    translation = services.signs.translate(payload.text)
    return TranslateToSignResponse(
        text=translation.text,
        words=[
            SignWordModel(
                text=word.text,
                signs=[
                    SignTokenModel(
                        character=token.character,
                        kind=token.kind.value,
                        asset=token.asset,
                        label=token.label,
                    )
                    for token in word.signs
                ],
            )
            for word in translation.words
        ],
        unsupported=translation.unsupported,
        sign_count=translation.sign_count,
    )
