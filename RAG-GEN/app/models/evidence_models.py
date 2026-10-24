
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OntologySystem(str):
    """
    Coding-system identifier for catalog entries used by ArabicNormalizer.

    Preserves ``OntologySystem(value).value`` as used in normalization.py.
    """

    @property
    def value(self) -> str:
        return str(self)


class OntologyCode(BaseModel):
    """A single ontology code attached to a catalog surface."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    system: OntologySystem
    code: str
    display: str = ""

    @field_validator("system", mode="before")
    @classmethod
    def coerce_system(cls, value: Any) -> OntologySystem:
        if isinstance(value, OntologySystem):
            return value
        return OntologySystem(str(value))


class SourceDocument(BaseModel):
    source_id: str
    title: str
    text: str
    media_type: str = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageDocument(BaseModel):
    source_id: str
    title: str
    image_path: str
    media_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceChunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
