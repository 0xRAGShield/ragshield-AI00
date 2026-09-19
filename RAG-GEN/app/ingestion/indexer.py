from __future__ import annotations

from app.embeddings.embedding_model import BGE_M3_Embedding
from app.models.evidence_models import EvidenceChunk
from app.vector_store.vector_store import QdrantStore


class Indexer:
    """
    Coordinates embedding generation and vector-store indexing.

    Responsibilities:
    - Validate indexing input.
    - Generate embeddings for evidence chunks.
    - Persist chunks and vectors through the vector-store contract.
    - Preserve one-to-one chunk/vector ordering.

    Forbidden responsibilities:
    - Cleaning.
    - Chunking.
    - Retrieval.
    - Reranking.
    - Prompt construction.
    - LLM generation.
    - Business logic.
    """

    def __init__(
        self,
        embedder: BGE_M3_Embedding,
        store: QdrantStore,
    ) -> None:
        if embedder is None:
            raise ValueError(
                "embedder must not be None."
            )

        if store is None:
            raise ValueError(
                "store must not be None."
            )

        self._embedder = embedder
        self._store = store

    def index(
        self,
        chunks: list[EvidenceChunk],
    ) -> int:
        if not isinstance(chunks, list):
            raise TypeError(
                "chunks must be a list."
            )

        if not chunks:
            return 0

        self._validate_chunks(chunks)

        texts = [
            chunk.text
            for chunk in chunks
        ]

        vectors = self._embedder.embed(texts)

        self._validate_vectors(
            vectors=vectors,
            expected_count=len(chunks),
        )

        self._store.upsert(
            chunks=chunks,
            vectors=vectors,
        )

        return len(chunks)

    @staticmethod
    def _validate_chunks(
        chunks: list[EvidenceChunk],
    ) -> None:
        seen_ids: set[str] = set()

        for chunk in chunks:
            if not isinstance(chunk, EvidenceChunk):
                raise TypeError(
                    "All items in chunks must be "
                    "EvidenceChunk instances."
                )

            if not chunk.chunk_id:
                raise ValueError(
                    "Chunk ID must not be empty."
                )

            if chunk.chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID detected: "
                    f"{chunk.chunk_id}"
                )

            seen_ids.add(chunk.chunk_id)

            if not chunk.source_id:
                raise ValueError(
                    f"Chunk source_id must not be empty: "
                    f"{chunk.chunk_id}"
                )

            if not chunk.text or not chunk.text.strip():
                raise ValueError(
                    f"Chunk text must not be empty: "
                    f"{chunk.chunk_id}"
                )

    @staticmethod
    def _validate_vectors(
        vectors: object,
        expected_count: int,
    ) -> None:
        if vectors is None:
            raise ValueError(
                "Embedding model returned no vectors."
            )

        try:
            vector_count = len(vectors)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                "Embedding model returned an invalid "
                "vector collection."
            ) from error

        if vector_count != expected_count:
            raise ValueError(
                "Embedding count does not match chunk count: "
                f"expected {expected_count}, "
                f"received {vector_count}."
            )

        for index, vector in enumerate(vectors):
            if vector is None:
                raise ValueError(
                    f"Embedding at index {index} is None."
                )

            try:
                dimension = len(vector)
            except TypeError as error:
                raise ValueError(
                    f"Embedding at index {index} is invalid."
                ) from error

            if dimension <= 0:
                raise ValueError(
                    f"Embedding at index {index} is empty."
                )