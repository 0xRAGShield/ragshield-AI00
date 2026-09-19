from __future__ import annotations

import re

from app.models.evidence_models import (
    SourceDocument,
    EvidenceChunk,
)


class ParentChildChunker:

    def __init__(
        self,
        parent_chunk_size: int = 1200,
        child_chunk_size: int = 400,
    ):
        if parent_chunk_size <= 0:
            raise ValueError(
                "parent_chunk_size must be greater than 0"
            )

        if child_chunk_size <= 0:
            raise ValueError(
                "child_chunk_size must be greater than 0"
            )

        if child_chunk_size >= parent_chunk_size:
            raise ValueError(
                "child_chunk_size must be smaller than "
                "parent_chunk_size"
            )

        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

    def chunk_document(
        self,
        document: SourceDocument,
    ) -> list[EvidenceChunk]:

        text = document.text.strip()

        if not text:
            raise ValueError(
                f"Cannot chunk empty document: "
                f"{document.source_id}"
            )

        paragraphs = self._split_paragraphs(text)
        parents = self._create_parents(paragraphs)

        chunks = []

        for parent_index, parent_text in enumerate(parents):

            parent_id = (
                f"{document.source_id}"
                f"_parent_{parent_index:05d}"
            )

            children = self._create_children(parent_text)

            for child_index, child_text in enumerate(children):

                chunk_id = (
                    f"{document.source_id}:"
                    f"{len(chunks)}"
                )

                metadata = {
                    "source_title": document.title,
                    "media_type": document.media_type,
                    "chunk_index": len(chunks),
                    "chunk_size": len(child_text),
                    "chunking_strategy": "parent_child",
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "child_index": child_index,
                }

                chunks.append(
                    EvidenceChunk(
                        chunk_id=chunk_id,
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

        all_chunks = []

        for document in documents:

            chunks = self.chunk_document(document)

            all_chunks.extend(chunks)

        return all_chunks

    def _split_paragraphs(
        self,
        text: str,
    ) -> list[str]:

        paragraphs = re.split(
            r"\n\s*\n",
            text,
        )

        return [
            paragraph.strip()
            for paragraph in paragraphs
            if paragraph.strip()
        ]

    def _create_parents(
        self,
        paragraphs: list[str],
    ) -> list[str]:

        parents = []
        current_parent = ""

        for paragraph in paragraphs:

            if len(paragraph) > self.parent_chunk_size:

                if current_parent:
                    parents.append(
                        current_parent.strip()
                    )
                    current_parent = ""

                large_parts = self._split_large_text(
                    paragraph,
                    self.parent_chunk_size,
                )

                parents.extend(large_parts)

                continue

            if not current_parent:

                current_parent = paragraph

                continue

            candidate = (
                current_parent
                + "\n\n"
                + paragraph
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

        sentences = re.split(
            r"(?<=[.!?؟])\s+",
            parent_text,
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        children = []
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
                current_child + " " + sentence
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

        chunks = []
        remaining = text

        while len(remaining) > max_size:

            sentence_positions = [
                remaining.rfind(".", 0, max_size + 1),
                remaining.rfind("!", 0, max_size + 1),
                remaining.rfind("?", 0, max_size + 1),
                remaining.rfind("؟", 0, max_size + 1),
            ]

            split_at = max(sentence_positions)

            if split_at <= 0:

                split_at = remaining.rfind(" ", 0, max_size + 1, )

            if split_at <= 0:

                split_at = max_size

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

        chunks = []
        remaining = text.strip()

        while len(remaining) > self.child_chunk_size:

            split_at = remaining.rfind(" ", 0, self.child_chunk_size + 1,)

            if split_at <= 0:

                split_at = self.child_chunk_size

            chunk = remaining[:split_at].strip()

            if chunk:
                chunks.append(chunk)

            remaining = remaining[split_at:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

