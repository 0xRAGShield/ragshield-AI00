from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_DNS, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.models.evidence_models import EvidenceChunk


class QdrantStore:
    """
    Qdrant vector-store adapter.

    Responsibilities:
    - Own the Qdrant client.
    - Create and validate the configured collection.
    - Persist chunk/vector pairs.
    - Execute dense vector similarity search.
    - Return stored evidence payloads.
    - Expose collection statistics.

    Forbidden responsibilities:
    - Embedding generation.
    - Chunking.
    - Cleaning.
    - Retrieval orchestration.
    - Reranking.
    - Prompt construction.
    - LLM generation.
    - Business logic.
    """

    def __init__(
        self,
        collection_name: str = "ragshield_corpus",
        host: str = "localhost",
        port: int = 6333,
        vector_size: int | None = None,
    ) -> None:
        if not isinstance(collection_name, str):
            raise TypeError("collection_name must be a string.")

        collection_name = collection_name.strip()

        if not collection_name:
            raise ValueError("collection_name cannot be empty.")

        if not isinstance(host, str):
            raise TypeError("host must be a string.")

        host = host.strip()

        if not host:
            raise ValueError("host cannot be empty.")

        if not isinstance(port, int) or isinstance(port, bool):
            raise TypeError("port must be an integer.")

        if port <= 0 or port > 65535:
            raise ValueError("port must be between 1 and 65535.")

        if vector_size is not None:
            if not isinstance(vector_size, int) or isinstance(vector_size, bool):
                raise TypeError("vector_size must be an integer.")

            if vector_size <= 0:
                raise ValueError("vector_size must be greater than zero.")

        self._collection_name = collection_name
        self._host = host
        self._port = port
        self._configured_vector_size = vector_size

        try:
            self._client = QdrantClient(
                host=self._host,
                port=self._port,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to initialize Qdrant client."
            ) from error

        if vector_size is not None:
            self.create_collection(vector_size)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def vector_size(self) -> int | None:
        return self._configured_vector_size

    def create_collection(
        self,
        vector_size: int,
    ) -> None:
        """
        Create the collection if it does not exist.

        If it already exists, its configured vector dimension
        must match the requested dimension.
        """
        self._validate_vector_size(vector_size)

        try:
            collections = self._client.get_collections()
        except Exception as error:
            raise RuntimeError(
                "Failed to inspect Qdrant collections."
            ) from error

        collection_exists = any(
            collection.name == self._collection_name
            for collection in collections.collections
        )

        if collection_exists:
            existing_vector_size = self._get_collection_vector_size()

            if existing_vector_size != vector_size:
                raise ValueError(
                    f"Collection '{self._collection_name}' expects "
                    f"vectors of size {existing_vector_size}, "
                    f"but received {vector_size}."
                )

            return

        try:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to create Qdrant collection "
                f"'{self._collection_name}'."
            ) from error

    def upsert(
        self,
        chunks: list[EvidenceChunk],
        vectors: list[list[float]],
    ) -> None:
        """
        Persist chunks and their corresponding vectors.

        Ordering is positional:
        chunks[i] <-> vectors[i].
        """
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list.")

        if not isinstance(vectors, list):
            raise TypeError("vectors must be a list.")

        if not chunks and not vectors:
            return

        if not chunks or not vectors:
            raise ValueError(
                "chunks and vectors must either both be empty "
                "or both contain items."
            )

        if len(chunks) != len(vectors):
            raise ValueError(
                "chunks and vectors must have the same length."
            )

        self._validate_chunks(chunks)

        vector_size = self._validate_vectors(vectors)

        configured_size = self._configured_vector_size

        if configured_size is not None and vector_size != configured_size:
            raise ValueError(
                f"Received vectors of size {vector_size}, "
                f"but configured vector size is {configured_size}."
            )

        self.create_collection(vector_size)

        points: list[PointStruct] = []

        for chunk, vector in zip(chunks, vectors):
            points.append(
                PointStruct(
                    id=self._make_point_id(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "source_id": chunk.source_id,
                        "text": chunk.text,
                        "metadata": dict(chunk.metadata or {}),
                    },
                )
            )

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to upsert {len(points)} points into "
                f"Qdrant collection '{self._collection_name}'."
            ) from error

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Execute dense cosine-similarity search.
        """
        if not isinstance(query_vector, list):
            raise TypeError("query_vector must be a list.")

        if not query_vector:
            raise ValueError("query_vector cannot be empty.")

        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer.")

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        query_size = self._validate_vector(query_vector)

        collection_vector_size = self._get_collection_vector_size()

        if query_size != collection_vector_size:
            raise ValueError(
                f"Query vector has dimension {query_size}, "
                f"but collection expects {collection_vector_size}."
            )

        try:
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            ).points
        except Exception as error:
            raise RuntimeError(
                f"Failed to search Qdrant collection "
                f"'{self._collection_name}'."
            ) from error

        return [
            {
                "chunk_id": payload.get("chunk_id"),
                "source_id": payload.get("source_id"),
                "text": payload.get("text"),
                "metadata": payload.get("metadata") or {},
                "score": result.score,
            }
            for result in results
            for payload in [result.payload or {}]
        ]

    def count(self) -> int:
        try:
            collection_info = self._client.get_collection(
                self._collection_name
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to read Qdrant collection "
                f"'{self._collection_name}'."
            ) from error

        return int(collection_info.points_count or 0)

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(
                collection_name=self._collection_name
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to delete Qdrant collection "
                f"'{self._collection_name}'."
            ) from error

    @staticmethod
    def _make_point_id(chunk_id: str) -> str:
        if not isinstance(chunk_id, str):
            raise TypeError("chunk_id must be a string.")

        chunk_id = chunk_id.strip()

        if not chunk_id:
            raise ValueError("chunk_id cannot be empty.")

        return str(
            uuid5(
                NAMESPACE_DNS,
                chunk_id,
            )
        )

    def _get_collection_vector_size(self) -> int:
        try:
            collection_info = self._client.get_collection(
                self._collection_name
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to read Qdrant collection "
                f"'{self._collection_name}'."
            ) from error

        vectors_config = collection_info.config.params.vectors

        try:
            vector_size = vectors_config.size
        except AttributeError as error:
            raise RuntimeError(
                "Configured Qdrant collection does not expose "
                "a single dense vector dimension."
            ) from error

        self._validate_vector_size(vector_size)

        return vector_size

    @staticmethod
    def _validate_vector_size(vector_size: int) -> None:
        if not isinstance(vector_size, int) or isinstance(vector_size, bool):
            raise TypeError("vector_size must be an integer.")

        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero.")

    @classmethod
    def _validate_vector(
        cls,
        vector: list[float],
    ) -> int:
        if not isinstance(vector, list):
            raise TypeError("Each vector must be a list.")

        if not vector:
            raise ValueError("Vectors cannot be empty.")

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            for value in vector
        ):
            raise ValueError(
                "Vectors must contain only numeric values."
            )

        vector_size = len(vector)
        cls._validate_vector_size(vector_size)

        return vector_size

    @classmethod
    def _validate_vectors(
        cls,
        vectors: list[list[float]],
    ) -> int:
        if not vectors:
            raise ValueError("Vectors cannot be empty.")

        vector_size = cls._validate_vector(vectors[0])

        for index, vector in enumerate(vectors[1:], start=1):
            current_size = cls._validate_vector(vector)

            if current_size != vector_size:
                raise ValueError(
                    f"Vector at index {index} has dimension "
                    f"{current_size}; expected {vector_size}."
                )

        return vector_size

    @staticmethod
    def _validate_chunks(
        chunks: list[EvidenceChunk],
    ) -> None:
        seen_ids: set[str] = set()

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, EvidenceChunk):
                raise TypeError(
                    f"chunks[{index}] must be an EvidenceChunk."
                )

            if not chunk.chunk_id or not chunk.chunk_id.strip():
                raise ValueError(
                    f"chunks[{index}].chunk_id cannot be empty."
                )

            if chunk.chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk_id detected: '{chunk.chunk_id}'."
                )

            seen_ids.add(chunk.chunk_id)

            if not chunk.source_id or not chunk.source_id.strip():
                raise ValueError(
                    f"chunks[{index}].source_id cannot be empty."
                )

            if not chunk.text or not chunk.text.strip():
                raise ValueError(
                    f"chunks[{index}].text cannot be empty."
                )