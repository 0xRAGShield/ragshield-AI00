from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

from app.core.settings import RetrievalSettings
from app.embeddings.embedding_model import BGE_M3_Embedding
from app.models.rag_models import (
    RetrievalCandidate,
    RetrievalMetrics,
    RetrievalQuery,
    RetrievalResult,
)
from app.vector_store.vector_store import QdrantStore


logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class RetrievalError(RuntimeError):
    """Base exception for retrieval failures."""


class InvalidQueryError(RetrievalError):
    """Raised when a retrieval query is invalid."""


class RetrievalConfigurationError(RetrievalError):
    """Raised when retrieval configuration is invalid."""


class EmbeddingFailureError(RetrievalError):
    """Raised when query embedding generation fails."""


class SearchFailureError(RetrievalError):
    """Raised when vector-store search fails."""


class InvalidCandidateError(RetrievalError):
    """Raised when a retrieved candidate violates the contract."""


# ============================================================
# Retriever
# ============================================================

class Retriever:
    """
    Production dense retrieval orchestrator.

    Runtime flow:

        RetrievalQuery
            ↓
        BGE-M3 embed_query()
            ↓
        QdrantStore.search()
            ↓
        Candidate validation
            ↓
        RetrievalResult
            ↓
        Reranker

    Responsibilities:
    - Validate retrieval requests.
    - Generate query embeddings.
    - Execute dense vector search.
    - Validate returned candidates.
    - Enforce retrieval limits and timeouts.
    - Report retrieval degradation.
    - Manage its owned executor lifecycle.

    Forbidden:
    - Chunking.
    - Cleaning.
    - Embedding-model loading.
    - Qdrant client creation.
    - Reranking.
    - Prompt construction.
    - LLM generation.
    - Hybrid retrieval.
    - Sparse retrieval.
    """

    def __init__(
        self,
        embedding_model: BGE_M3_Embedding,
        vector_store: QdrantStore,
        settings: RetrievalSettings | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        if embedding_model is None:
            raise RetrievalConfigurationError(
                "embedding_model cannot be None."
            )

        if vector_store is None:
            raise RetrievalConfigurationError(
                "vector_store cannot be None."
            )

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._settings = settings or RetrievalSettings()

        self._owns_executor = executor is None

        self._executor = executor or ThreadPoolExecutor(
            max_workers=self._settings.executor_workers,
            thread_name_prefix="ragshield-retrieval",
        )

        self._closed = False

        self._validate_backend_contract()

    # ========================================================
    # Properties
    # ========================================================

    @property
    def collection_name(self) -> str:
        return self._settings.collection_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_model.dimension

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model.model_name

    # ========================================================
    # Public retrieval
    # ========================================================

    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        """
        Execute dense retrieval.
        """

        self._ensure_open()

        if not isinstance(query, RetrievalQuery):
            raise InvalidQueryError(
                "query must be a RetrievalQuery instance."
            )

        trace_id = uuid.uuid4().hex
        started_at = time.perf_counter()

        top_k = self._resolve_top_k(query.top_k)

        metrics = RetrievalMetrics(
            trace_id=trace_id,
            started_at=started_at,
            requested_top_k=top_k,
        )

        dense_vector = await self._generate_query_embedding(
            query=query,
            metrics=metrics,
            trace_id=trace_id,
        )

        raw_candidates = await self._execute_search(
            query_vector=dense_vector,
            top_k=top_k,
            metrics=metrics,
            trace_id=trace_id,
        )

        candidates = self._validate_candidates(
            raw_candidates=raw_candidates,
            metrics=metrics,
        )

        metrics.returned_candidates = len(candidates)

        metrics.total_latency_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        if metrics.dropped_candidates > 0:
            metrics.is_degraded = True

            if not self._settings.allow_degraded_results:
                raise SearchFailureError(
                    "Retrieval result was degraded because one or more "
                    "candidates failed validation."
                )

        return RetrievalResult(
            query=query,
            candidates=tuple(candidates),
            metrics=metrics,
        )

    # ========================================================
    # Query embedding
    # ========================================================

    async def _generate_query_embedding(
        self,
        query: RetrievalQuery,
        metrics: RetrievalMetrics,
        trace_id: str,
    ) -> list[float]:
        started_at = time.perf_counter()

        try:
            vector = await self._run_with_timeout(
                self._embed_query,
                query.text,
                timeout=self._settings.embedding_timeout_seconds,
            )

        except asyncio.TimeoutError as error:
            logger.error(
                "Query embedding timed out. trace_id=%s",
                trace_id,
            )

            raise EmbeddingFailureError(
                "Query embedding timed out."
            ) from error

        except EmbeddingFailureError:
            raise

        except Exception as error:
            logger.exception(
                "Query embedding failed. trace_id=%s",
                trace_id,
            )

            raise EmbeddingFailureError(
                "Failed to generate query embedding."
            ) from error

        metrics.embedding_latency_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        return vector

    def _embed_query(
        self,
        text: str,
    ) -> list[float]:
        try:
            vector = self._embedding_model.embed_query(text)
        except Exception as error:
            raise EmbeddingFailureError(
                "BGE-M3 failed to generate the query embedding."
            ) from error

        self._validate_dense_vector(vector)

        return vector

    # ========================================================
    # Vector search
    # ========================================================

    async def _execute_search(
        self,
        query_vector: list[float],
        top_k: int,
        metrics: RetrievalMetrics,
        trace_id: str,
    ) -> list[dict[str, Any]]:
        started_at = time.perf_counter()

        try:
            raw_candidates = await self._run_with_timeout(
                self._search,
                query_vector,
                top_k,
                timeout=self._settings.search_timeout_seconds,
            )

        except asyncio.TimeoutError as error:
            logger.error(
                "Qdrant search timed out. trace_id=%s",
                trace_id,
            )

            raise SearchFailureError(
                "Vector search timed out."
            ) from error

        except SearchFailureError:
            raise

        except Exception as error:
            logger.exception(
                "Qdrant search failed. trace_id=%s",
                trace_id,
            )

            raise SearchFailureError(
                "Vector search failed."
            ) from error

        metrics.search_latency_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        return raw_candidates

    def _search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        try:
            return self._vector_store.search(
                query_vector=query_vector,
                limit=top_k,
            )
        except Exception as error:
            raise SearchFailureError(
                "Qdrant dense search failed."
            ) from error

    # ========================================================
    # Candidate validation
    # ========================================================

    def _validate_candidates(
        self,
        raw_candidates: Sequence[Mapping[str, Any]],
        metrics: RetrievalMetrics,
    ) -> list[RetrievalCandidate]:
        if (
            isinstance(raw_candidates, (str, bytes))
            or not isinstance(raw_candidates, Sequence)
        ):
            raise InvalidCandidateError(
                "Vector store returned an invalid candidate collection."
            )

        candidates: list[RetrievalCandidate] = []
        seen_chunk_ids: set[str] = set()

        for index, raw in enumerate(raw_candidates):
            try:
                if not isinstance(raw, Mapping):
                    raise InvalidCandidateError(
                        f"Candidate at index {index} must be a mapping."
                    )

                candidate = RetrievalCandidate(
                    chunk_id=raw["chunk_id"],
                    source_id=raw["source_id"],
                    text=raw["text"],
                    score=raw["score"],
                    metadata=raw.get("metadata") or {},
                )

                if candidate.chunk_id in seen_chunk_ids:
                    raise InvalidCandidateError(
                        f"Duplicate chunk_id: {candidate.chunk_id}"
                    )

                seen_chunk_ids.add(candidate.chunk_id)
                candidates.append(candidate)

            except (
                KeyError,
                TypeError,
                ValueError,
                InvalidCandidateError,
            ) as error:
                metrics.dropped_candidates += 1

                logger.warning(
                    "Invalid retrieval candidate dropped. "
                    "trace_id=%s index=%d reason=%s",
                    metrics.trace_id,
                    index,
                    error,
                )

        return candidates

    # ========================================================
    # Backend contract validation
    # ========================================================

    def _validate_backend_contract(self) -> None:
        store_collection = getattr(
            self._vector_store,
            "collection_name",
            None,
        )

        if store_collection is not None:
            if store_collection != self._settings.collection_name:
                raise RetrievalConfigurationError(
                    "Retrieval collection does not match "
                    "the vector-store collection."
                )

        store_dimension = getattr(
            self._vector_store,
            "vector_size",
            None,
        )

        if (
            store_dimension is not None
            and store_dimension != self._embedding_model.dimension
        ):
            raise RetrievalConfigurationError(
                "Embedding dimension does not match "
                "the vector-store configured dimension."
            )

    def _validate_dense_vector(
        self,
        vector: Any,
    ) -> None:
        if not isinstance(vector, list):
            raise EmbeddingFailureError(
                "Embedding model returned an invalid vector type."
            )

        if len(vector) != self._embedding_model.dimension:
            raise EmbeddingFailureError(
                "Embedding dimension does not match "
                "the BGE-M3 dimension."
            )

        for index, value in enumerate(vector):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise EmbeddingFailureError(
                    f"Embedding vector contains an invalid value "
                    f"at index {index}."
                )

    # ========================================================
    # Top-k
    # ========================================================

    def _resolve_top_k(
        self,
        requested_top_k: int | None,
    ) -> int:
        top_k = (
            self._settings.default_top_k
            if requested_top_k is None
            else requested_top_k
        )

        if top_k > self._settings.max_top_k:
            raise InvalidQueryError(
                f"top_k cannot exceed "
                f"{self._settings.max_top_k}."
            )

        return top_k

    # ========================================================
    # Async executor
    # ========================================================

    async def _run_with_timeout(
        self,
        func: Any,
        *args: Any,
        timeout: float,
    ) -> Any:
        loop = asyncio.get_running_loop()

        future = loop.run_in_executor(
            self._executor,
            func,
            *args,
        )

        return await asyncio.wait_for(
            future,
            timeout=timeout,
        )

    # ========================================================
    # Lifecycle
    # ========================================================

    def _ensure_open(self) -> None:
        if self._closed:
            raise RetrievalError(
                "Retriever is closed."
            )

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        if self._owns_executor:
            self._executor.shutdown(wait=True)

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True

        if self._owns_executor:
            loop = asyncio.get_running_loop()

            await loop.run_in_executor(
                None,
                self._executor.shutdown,
                True,
            )

    def __enter__(self) -> "Retriever":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    async def __aenter__(self) -> "Retriever":
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        await self.aclose()
