from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.evidence_models import (
    EvidenceChunk,
    SourceDocument,
)


_PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?؟])(?:\s+|$)"
)


@dataclass(frozen=True)
class ChunkingLimits:
    """Configuration limits for parent-child chunking."""

    parent_chunk_size: int = 1200
    child_chunk_size: int = 400
    max_chunks_per_document: int = 10_000

    def __post_init__(self) -> None:
        if self.parent_chunk_size <= 0:
            raise ValueError(
                "parent_chunk_size must be greater than 0."
            )

        if self.child_chunk_size <= 0:
            raise ValueError(
                "child_chunk_size must be greater than 0."
            )

        if self.child_chunk_size >= self.parent_chunk_size:
            raise ValueError(
                "child_chunk_size must be smaller than "
                "parent_chunk_size."
            )

        if self.max_chunks_per_document <= 0:
            raise ValueError(
                "max_chunks_per_document must be greater than 0."
            )


class ParentChildChunker:
    """
    Split source documents into parent/child evidence chunks.

    Responsibilities:
    - Preserve document identity and provenance.
    - Build deterministic parent/child relationships.
    - Produce bounded child chunks for retrieval/indexing.
    - Preserve source metadata.

    Forbidden responsibilities:
    - Cleaning / normalization.
    - Embedding generation.
    - Vector indexing.
    - Retrieval.
    - Reranking.
    - OCR / vision processing.
    - LLM generation.
    - Business logic.
    """

    def __init__(
        self,
        limits: ChunkingLimits | None = None,
    ) -> None:
        self._limits = limits or ChunkingLimits()

    @property
    def parent_chunk_size(self) -> int:
        return self._limits.parent_chunk_size

    @property
    def child_chunk_size(self) -> int:
        return self._limits.child_chunk_size

    def chunk_document(
        self,
        document: SourceDocument,
    ) -> list[EvidenceChunk]:
        if not isinstance(document, SourceDocument):
            raise TypeError(
                "document must be a SourceDocument."
            )

        text = document.text.strip()

        if not text:
            raise ValueError(
                f"Cannot chunk empty document: "
                f"{document.source_id}"
            )

        paragraphs = self._split_paragraphs(text)
        parents = self._create_parents(paragraphs)

        chunks: list[EvidenceChunk] = []

        for parent_index, parent_text in enumerate(parents):
            parent_id = (
                f"{document.source_id}"
                f":parent:{parent_index:05d}"
            )

            children = self._create_children(parent_text)

            for child_index, child_text in enumerate(children):
                if len(chunks) >= self._limits.max_chunks_per_document:
                    raise ValueError(
                        f"Document exceeds maximum chunk count: "
                        f"{document.source_id}"
                    )

                chunk_index = len(chunks)

                metadata = dict(document.metadata or {})

                metadata.update(
                    {
                        "source_title": document.title,
                        "media_type": document.media_type,
                        "chunk_index": chunk_index,
                        "chunk_size": len(child_text),
                        "chunking_strategy": "parent_child",
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                        "child_index": child_index,
                    }
                )

                chunks.append(
                    EvidenceChunk(
                        chunk_id=(
                            f"{document.source_id}:"
                            f"chunk:{chunk_index:06d}"
                        ),
                        source_id=document.source_id,
                        text=child_text,
                        metadata=metadata,
                    )
                )

        return chunks

    def chunk_documents(
        self,
        documents: list[SourceDocument],
    ) -> list[EvidenceChunk]:
        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list."
            )

        all_chunks: list[EvidenceChunk] = []

        for document in documents:
            all_chunks.extend(
                self.chunk_document(document)
            )

        return all_chunks

    def _split_paragraphs(
        self,
        text: str,
    ) -> list[str]:
        return [
            paragraph.strip()
            for paragraph in _PARAGRAPH_SEPARATOR.split(text)
            if paragraph.strip()
        ]

    def _create_parents(
        self,
        paragraphs: list[str],
    ) -> list[str]:
        parents: list[str] = []
        current_parent = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.parent_chunk_size:
                if current_parent:
                    parents.append(
                        current_parent.strip()
                    )
                    current_parent = ""

                parents.extend(
                    self._split_large_text(
                        paragraph,
                        self.parent_chunk_size,
                    )
                )
                continue

            if not current_parent:
                current_parent = paragraph
                continue

            candidate = (
                f"{current_parent}\n\n{paragraph}"
            )

            if len(candidate) <= self.parent_chunk_size:
                current_parent = candidate
            else:
                parents.append(
                    current_parent.strip()
                )
                current_parent = paragraph

        if current_parent:
            parents.append(
                current_parent.strip()
            )

        return parents

    def _create_children(
        self,
        parent_text: str,
    ) -> list[str]:
        parent_text = parent_text.strip()

        if not parent_text:
            return []

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(parent_text)
            if sentence.strip()
        ]

        children: list[str] = []
        current_child = ""

        for sentence in sentences:
            if len(sentence) > self.child_chunk_size:
                if current_child:
                    children.append(
                        current_child.strip()
                    )
                    current_child = ""

                children.extend(
                    self._split_long_text(sentence)
                )
                continue

            if not current_child:
                current_child = sentence
                continue

            candidate = (
                f"{current_child} {sentence}"
            )

            if len(candidate) <= self.child_chunk_size:
                current_child = candidate
            else:
                children.append(
                    current_child.strip()
                )
                current_child = sentence

        if current_child:
            children.append(
                current_child.strip()
            )

        return children

    def _split_large_text(
        self,
        text: str,
        max_size: int,
    ) -> list[str]:
        text = text.strip()

        if not text:
            return []

        chunks: list[str] = []
        remaining = text

        while len(remaining) > max_size:
            split_at = self._find_sentence_split(
                remaining,
                max_size,
            )

            if split_at <= 0:
                split_at = remaining.rfind(
                    " ",
                    0,
                    max_size + 1,
                )

            if split_at <= 0:
                split_at = max_size
            else:
                split_at += 1

            chunk = remaining[:split_at].strip()

            if chunk:
                chunks.append(chunk)

            remaining = remaining[split_at:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    def _split_long_text(
        self,
        text: str,
    ) -> list[str]:
        chunks: list[str] = []
        remaining = text.strip()

        while len(remaining) > self.child_chunk_size:
            split_at = remaining.rfind(
                " ",
                0,
                self.child_chunk_size + 1,
            )

            if split_at <= 0:
                split_at = self.child_chunk_size

            chunk = remaining[:split_at].strip()

            if chunk:
                chunks.append(chunk)

            remaining = remaining[split_at:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    @staticmethod
    def _find_sentence_split(
        text: str,
        max_size: int,
    ) -> int:
        candidates = [
            text.rfind(".", 0, max_size + 1),
            text.rfind("!", 0, max_size + 1),
            text.rfind("?", 0, max_size + 1),
            text.rfind("؟", 0, max_size + 1),
        ]

        return max(candidates)