from __future__ import annotations
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    source_id: str
    title: str
    text: str
    media_type: str = "text/plain"
    metadata: dict = Field(default_factory=dict)


class ImageDocument(BaseModel):
    source_id: str
    title: str
    image_path: str
    media_type: str
    metadata: dict = Field(default_factory=dict)


class EvidenceChunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    metadata: dict = Field(default_factory=dict)