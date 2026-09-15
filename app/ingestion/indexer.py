from __future__ import annotations

from app.embeddings.embedding_model import BGE_M3_Embedding
from app.models.evidence_models import EvidenceChunk
from app.vector_store.vector_store import QdrantStore


class Indexer:

    def __init__(
        self,
        embedder: BGE_M3_Embedding,
        store: QdrantStore,
    ) -> None:

        self._embedder = embedder
        self._store = store

    def index(
        self,
        chunks: list[EvidenceChunk],
    ) -> int:

        if not chunks:
            return 0

        vectors = self._embedder.embed(
            [chunk.text for chunk in chunks]
        )

        self._store.upsert(
            chunks=chunks,
            vectors=vectors,
        )

        return len(chunks)