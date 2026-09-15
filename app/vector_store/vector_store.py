

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_DNS, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.models.evidence_models import EvidenceChunk


class QdrantStore:

    def __init__(
        self,
        collection_name: str = "ragshield_corpus",
        host: str = "localhost",
        port: int = 6333,
        vector_size: int | None = None,
    ) -> None:

        if not collection_name.strip():
            raise ValueError(
                "collection_name cannot be empty."
            )

        if port <= 0:
            raise ValueError(
                "port must be greater than zero."
            )

        if vector_size is not None and vector_size <= 0:
            raise ValueError(
                "vector_size must be greater than zero."
            )

        self.collection_name = collection_name

        self.client = QdrantClient(
            host=host,
            port=port,
        )

        if vector_size is not None:
            self.create_collection(vector_size)

    def create_collection(
        self,
        vector_size: int,
    ) -> None:

        if vector_size <= 0:
            raise ValueError(
                "vector_size must be greater than zero."
            )

        collections = self.client.get_collections()

        collection_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name in collection_names:
            collection_info = self.client.get_collection(
                self.collection_name
            )

            existing_vector_size = (
                collection_info.config.params.vectors.size
            )

            if existing_vector_size != vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' "
                    f"expects vectors of size "
                    f"{existing_vector_size}, "
                    f"but received {vector_size}."
                )

            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    @staticmethod
    def _make_point_id(
        chunk_id: str,
    ) -> str:

        if not chunk_id.strip():
            raise ValueError(
                "chunk_id cannot be empty."
            )

        return str(
            uuid5(
                NAMESPACE_DNS,
                chunk_id,
            )
        )

    def upsert(
        self,
        chunks: list[EvidenceChunk],
        vectors: list[list[float]],
    ) -> None:

        if not chunks and not vectors:
            return

        if len(chunks) != len(vectors):
            raise ValueError(
                "chunks and vectors must have the same length."
            )

        if not vectors:
            return

        vector_size = len(vectors[0])

        if vector_size == 0:
            raise ValueError(
                "Vectors cannot be empty."
            )

        if any(
            len(vector) != vector_size
            for vector in vectors
        ):
            raise ValueError(
                "All vectors must have the same dimension."
            )

        self.create_collection(vector_size)

        points: list[PointStruct] = []

        for chunk, vector in zip(chunks, vectors):

            point_id = self._make_point_id(
                chunk.chunk_id
            )

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:

        if not query_vector:
            raise ValueError(
                "Query vector cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        collection_info = self.client.get_collection(
            self.collection_name
        )

        vector_size = (
            collection_info.config.params.vectors.size
        )

        if len(query_vector) != vector_size:
            raise ValueError(
                f"Query vector has dimension "
                f"{len(query_vector)}, "
                f"but collection expects "
                f"{vector_size}."
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id": result.payload.get(
                    "chunk_id"
                ),
                "source_id": result.payload.get(
                    "source_id"
                ),
                "text": result.payload.get(
                    "text"
                ),
                "metadata": result.payload.get(
                    "metadata",
                    {},
                ),
                "score": result.score,
            }
            for result in results
        ]

    def count(self) -> int:

        collection_info = self.client.get_collection(
            self.collection_name
        )

        return collection_info.points_count or 0

    def delete_collection(self) -> None:

        self.client.delete_collection(
            collection_name=self.collection_name
        )