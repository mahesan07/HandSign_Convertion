"""Contracts for the English -> sign direction."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SignTokenModel(BaseModel):
    character: str
    kind: str
    asset: Optional[str] = None
    label: str


class SignWordModel(BaseModel):
    text: str
    signs: List[SignTokenModel] = Field(default_factory=list)


class TranslateToSignRequest(BaseModel):
    text: str = Field(default="", max_length=500)


class TranslateToSignResponse(BaseModel):
    text: str
    words: List[SignWordModel] = Field(default_factory=list)
    #: Characters with no available sign, listed once each.
    unsupported: List[str] = Field(default_factory=list)
    sign_count: int = 0


class SignCatalogResponse(BaseModel):
    """The single source of truth for letter -> image, used by the frontend."""

    asset_base: str
    signs: Dict[str, str] = Field(default_factory=dict)


__all__ = [
    "SignTokenModel",
    "SignWordModel",
    "TranslateToSignRequest",
    "TranslateToSignResponse",
    "SignCatalogResponse",
]
