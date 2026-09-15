"""Production retrieval component.

Pipeline:

    Query contract → Embedding component → Qdrant search → Candidate validation → RetrievalResult

Self-contained until teammate ``settings.py`` / ``rag_models.py`` exist: typed
exceptions, dummy Settings, and Pydantic contracts live at the top of this file.

This module does not import Reranker, Context Builder, Prompt Builder, LLM,
Arabic/Egyptian normalization, chunking, citations, or generation. Fusion of
dense+sparse scores (if hybrid) is performed by Qdrant, not here.

Teammate defaults consumed as identity, not reimplemented:
collection ``ragshielded_chunks``, embedding backend ``hashed_ngram`` dim 256,
cosine (higher score is better), payload ``chunk_id`` / ``source_id`` / ``text``
/ ``loe`` / ``ontology_codes``.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

try:
    from qdrant_client import AsyncQdrantClient, QdrantClient
    from qdrant_client.http import models as qmodels
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )

    _QDRANT_IMPORT_ERROR: BaseException | None = None
except Exception as _qdrant_exc:  # optional runtime dependency
    AsyncQdrantClient = None  # type: ignore[assignment, misc]
    QdrantClient = None  # type: ignore[assignment, misc]
    qmodels = None  # type: ignore[assignment]
    ResponseHandlingException = Exception  # type: ignore[misc, assignment]
    UnexpectedResponse = Exception  # type: ignore[misc, assignment]
    _QDRANT_IMPORT_ERROR = _qdrant_exc


logger = logging.getLogger("rag.retriever")

__all__ = (
    "RetrievalError", "EmptyQueryError", "InvalidQueryError", "InvalidTopKError",
    "MissingCollectionError", "InvalidEmbeddingError", "EmbeddingFailureError",
    "DimensionMismatchError", "QdrantUnavailableError", "QdrantTimeoutError",
    "SearchFailureError", "InvalidPayloadError", "MalformedQdrantResultError",
    "InvalidConfigurationError", "DataQualityFailureError", "IndexConsistencyError",
    "Settings", "ScoreSemanticsConfig", "get_settings", "compute_embedding_fingerprint",
    "Query", "QueryMetadata", "Candidate", "RetrievalResult", "RetrievalMetrics",
    "ScoreSemantics", "SparseEmbedding", "SearchHit", "CollectionInfo",
    "EmbeddingComponent", "VectorStore", "QdrantVectorStore", "Retriever",
    "current_trace_id", "current_retrieval_metrics", "query_hash",
    "validate_index_consistency",
)

DistanceMetric = Literal["cosine", "euclid", "dot"]
CollectionPolarity = Literal["positive", "negative", "default"]

_DEFAULT_PAYLOAD_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "source_id",
    "text",
    "loe",
    "ontology_codes",
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_QUERY_HASH_BYTES = 12


# ===========================================================================
# Typed exceptions — failures are traceable, never swallowed into []
# ===========================================================================


class RetrievalError(Exception):
    """Base class for retrieval failures."""

    error_code: str = "retrieval_error"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause


class EmptyQueryError(RetrievalError):
    error_code = "empty_query"


class InvalidQueryError(RetrievalError):
    error_code = "invalid_query"


class InvalidTopKError(RetrievalError):
    error_code = "invalid_top_k"


class MissingCollectionError(RetrievalError):
    error_code = "missing_collection"


class InvalidEmbeddingError(RetrievalError):
    error_code = "invalid_embedding"


class EmbeddingFailureError(RetrievalError):
    error_code = "embedding_failure"


class DimensionMismatchError(RetrievalError):
    error_code = "dimension_mismatch"


class QdrantUnavailableError(RetrievalError):
    error_code = "qdrant_unavailable"


class QdrantTimeoutError(RetrievalError):
    error_code = "qdrant_timeout"


class SearchFailureError(RetrievalError):
    error_code = "search_failure"


class InvalidPayloadError(RetrievalError):
    error_code = "invalid_payload"


class MalformedQdrantResultError(RetrievalError):
    error_code = "malformed_qdrant_result"


class InvalidConfigurationError(RetrievalError):
    error_code = "invalid_configuration"


class DataQualityFailureError(RetrievalError):
    """All hits malformed. Distinct from a legitimate empty retrieval."""

    error_code = "data_quality_failure"


class IndexConsistencyError(RetrievalError):
    """Startup mismatch between embedding config and the indexed collection."""

    error_code = "index_consistency"


# ===========================================================================
# Dummy Settings — stand-in for teammate settings.py
# All retrieve-time knobs come from here, not from hardcoded logic.
# ===========================================================================


def compute_embedding_fingerprint(
    model_id: str,
    dimension: int,
    *,
    algo: str = "hashed_ngram_v1",
) -> str:
    payload = f"{algo}|{model_id}|{dimension}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ScoreSemanticsConfig(BaseModel):
    """How raw vector-store scores must be interpreted for this strategy."""

    model_config = ConfigDict(extra="forbid")

    metric: DistanceMetric = "cosine"
    higher_is_better: bool = True
    threshold_mode: Literal["min_similarity", "max_distance"] = "min_similarity"
    fusion_owner: Literal["vector_store", "none"] = "vector_store"
    description: str = (
        "Cosine similarity from the teammate embedding (hashed_ngram identity, dim 256). "
        "Higher is better. Hybrid fusion, if enabled, is owned by the Qdrant layer."
    )


class Settings(BaseModel):
    """In-file stand-in for teammate ``settings.py``.

    Defaults match ``ragshield-AI-phase-3-evaluation-handoff/app/core/settings.py``
    where that file exists (collection name, hashed_ngram, dim 256).
    """

    model_config = ConfigDict(extra="forbid")

    collection_name: str = "ragshielded_chunks"
    top_k: int = 8
    max_top_k: int = 50
    score_threshold: float | None = 0.0
    embedding_backend: str = "hashed_ngram"
    embedding_model: str = "hashed_ngram"
    embedding_model_name: str = "NOT_FINAL_DEV_DEFAULT"
    embedding_dimension: int = 256
    distance_metric: DistanceMetric = "cosine"
    timeout_seconds: float = 5.0
    embedding_timeout_seconds: float = 10.0
    default_filters: dict[str, Any] = Field(default_factory=dict)
    embedding_fingerprint: str = ""
    hybrid_search_enabled: bool = False
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"
    use_named_vectors: bool = False
    expected_payload_fields: tuple[str, ...] = _DEFAULT_PAYLOAD_FIELDS
    score_semantics: ScoreSemanticsConfig = Field(default_factory=ScoreSemanticsConfig)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    index_meta_point_id: str = "__index_meta__"
    executor_max_workers: int = 4
    collection_polarity: CollectionPolarity = "default"

    @model_validator(mode="after")
    def _fill_fingerprint_and_semantics(self) -> Settings:
        if not self.embedding_fingerprint:
            self.embedding_fingerprint = compute_embedding_fingerprint(
                self.embedding_model,
                self.embedding_dimension,
            )
        metric = self.distance_metric
        higher = metric in ("cosine", "dot")
        mode: Literal["min_similarity", "max_distance"] = (
            "min_similarity" if higher else "max_distance"
        )
        semantics = self.score_semantics
        if semantics.metric != metric or semantics.higher_is_better != higher:
            self.score_semantics = semantics.model_copy(
                update={
                    "metric": metric,
                    "higher_is_better": higher,
                    "threshold_mode": mode,
                }
            )
        return self


def get_settings() -> Settings:
    """Fresh dummy Settings. Retriever copies config at construction; no shared mutable state."""

    return Settings()


# ===========================================================================
# Pydantic contracts — stand-in for teammate rag_models.py
# ===========================================================================


class QueryMetadata(BaseModel):
    """Metadata produced *before* retrieval. The retriever does not rewrite medical meaning."""

    model_config = ConfigDict(extra="allow")

    filters: dict[str, Any] = Field(default_factory=dict)
    ontology_codes: list[str] = Field(default_factory=list)
    patient_id: str | None = None
    language: str | None = None
    top_k: int | None = None
    score_threshold: float | None = None
    collection_polarity: CollectionPolarity | None = None
    source_types: list[str] = Field(default_factory=list)
    request_id: str | None = None


class Query(BaseModel):
    """Processed query contract. Text is consumed as-is (no re-normalization)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    text: str
    metadata: QueryMetadata = Field(default_factory=QueryMetadata)
    request_id: str | None = None

    def __repr__(self) -> str:
        return f"Query(length={len(self.text)}, request_id={self.request_id!r})"

    def __str__(self) -> str:
        return self.__repr__()


class Candidate(BaseModel):
    """A single retrieved chunk. Identifiers aligned with teammate payload."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str
    source_id: str | None = None
    loe: str | None = None
    ontology_codes: list[str] = Field(default_factory=list)
    point_id: str | None = None


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_time: float = 0.0
    vector_search_time: float = 0.0
    candidate_validation_time: float = 0.0
    total_retrieval_time: float = 0.0
    candidate_count: int = 0
    valid_candidate_count: int = 0
    dropped_candidate_count: int = 0


class ScoreSemantics(BaseModel):
    """Documented on every result. Positive/negative fusion is not this component's job."""

    model_config = ConfigDict(extra="forbid")

    metric: DistanceMetric
    higher_is_better: bool
    threshold_applied: float | None = None
    threshold_mode: Literal["min_similarity", "max_distance"]
    fusion_owner: Literal["vector_store", "none"]
    polarity: CollectionPolarity = "default"
    description: str = ""


class RetrievalResult(BaseModel):
    """Typed retrieval output. Never a raw dict.

    Empty vs data-quality failure:
      - candidates=[] and is_degraded=False → legitimate no hits
      - candidates=[] and is_degraded=True and data_quality_failure=True
        and dropped_candidates_count>0 → all hits malformed
    """

    model_config = ConfigDict(extra="forbid")

    query: Query
    candidates: list[Candidate] = Field(default_factory=list)
    is_degraded: bool = False
    dropped_candidates_count: int = 0
    data_quality_failure: bool = False
    error_code: str | None = None
    metrics: RetrievalMetrics
    score_semantics: ScoreSemantics
    trace_id: str
    dropped_reasons: list[str] = Field(default_factory=list)


class SparseEmbedding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indices: list[int]
    values: list[float]


class SearchHit(BaseModel):
    """Normalized vector-store hit *before* candidate field validation."""

    model_config = ConfigDict(extra="allow")

    id: Any
    score: float
    payload: dict[str, Any] | None = None


class CollectionInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    exists: bool
    vector_size: int | None = None
    distance: DistanceMetric | None = None
    embedding_model: str | None = None
    embedding_fingerprint: str | None = None
    payload_schema_fields: list[str] = Field(default_factory=list)
    hybrid_enabled: bool = False
    sparse_vector_name: str | None = None
    dense_vector_name: str | None = None


# ===========================================================================
# Observability — request-scoped contextvars; never log raw medical query text
# ===========================================================================

_trace_id_var: ContextVar[str] = ContextVar("rag_retriever_trace_id", default="")
_metrics_var: ContextVar[RetrievalMetrics | None] = ContextVar(
    "rag_retriever_metrics",
    default=None,
)


def current_trace_id() -> str:
    return _trace_id_var.get()


def current_retrieval_metrics() -> RetrievalMetrics | None:
    return _metrics_var.get()


def query_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_QUERY_HASH_BYTES]


def _log_event(event: str, **fields: Any) -> None:
    logger.info("%s", {"event": event, "trace_id": current_trace_id(), **fields})


# ===========================================================================
# Embedding + vector-store boundaries
# ===========================================================================


@runtime_checkable
class EmbeddingComponent(Protocol):
    """Injected teammate embedding. Same model / dim / fingerprint as the index.

    This file does not implement a search engine or hashed-ngram indexer.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    def embed_dense(self, text: str) -> Sequence[float]:
        """Synchronous dense embedding. Retriever offloads this off the event loop."""
        ...

    def embed_sparse(self, text: str) -> SparseEmbedding | None:
        """Optional. Required only when Settings.hybrid_search_enabled is True."""
        ...


class VectorStore(ABC):
    """Search + health + collection metadata. Qdrant adapter implements this."""

    @abstractmethod
    async def search(
        self,
        *,
        collection_name: str,
        dense: Sequence[float],
        sparse: SparseEmbedding | None,
        top_k: int,
        filters: Mapping[str, Any] | None,
        score_threshold: float | None,
        timeout_seconds: float,
        hybrid: bool,
    ) -> list[SearchHit]: ...

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> CollectionInfo: ...

    @abstractmethod
    async def health(self) -> bool: ...


# ===========================================================================
# Validation helpers
# ===========================================================================


def normalize_distance(value: str | None) -> DistanceMetric:
    if value is None:
        raise InvalidConfigurationError("distance metric is required")
    key = value.strip().lower().replace(" ", "")
    aliases: dict[str, DistanceMetric] = {
        "cosine": "cosine",
        "cos": "cosine",
        "euclid": "euclid",
        "euclidean": "euclid",
        "l2": "euclid",
        "dot": "dot",
        "dotproduct": "dot",
    }
    mapped = aliases.get(key)
    if mapped is None:
        raise InvalidConfigurationError(f"unsupported distance metric: {value!r}")
    return mapped


def merge_filters(
    default_filters: Mapping[str, Any],
    query_filters: Mapping[str, Any],
    ontology_codes: Sequence[str] | None = None,
    source_types: Sequence[str] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(default_filters)
    merged.update(query_filters)
    if ontology_codes:
        merged.setdefault("ontology_codes", list(ontology_codes))
    if source_types:
        merged.setdefault("source_types", list(source_types))
    return merged


def validate_dense_vector(vector: Sequence[float], expected_dim: int) -> list[float]:
    if vector is None:
        raise InvalidEmbeddingError("embedding is None")
    try:
        values = [float(x) for x in vector]
    except (TypeError, ValueError) as exc:
        raise InvalidEmbeddingError("embedding contains non-numeric values") from exc
    if not values:
        raise InvalidEmbeddingError("embedding vector is empty")
    if len(values) != expected_dim:
        raise DimensionMismatchError(
            f"embedding dimension {len(values)} != expected {expected_dim}"
        )
    if any(not math.isfinite(x) for x in values):
        raise InvalidEmbeddingError("embedding contains NaN or Inf")
    if all(x == 0.0 for x in values):
        raise InvalidEmbeddingError("embedding vector is all zeros")
    return values


def validate_sparse_vector(sparse: SparseEmbedding) -> SparseEmbedding:
    if len(sparse.indices) != len(sparse.values):
        raise InvalidEmbeddingError("sparse embedding indices/values length mismatch")
    if not sparse.indices:
        raise InvalidEmbeddingError("sparse embedding is empty")
    if any(not math.isfinite(v) for v in sparse.values):
        raise InvalidEmbeddingError("sparse embedding contains NaN or Inf")
    if any(i < 0 for i in sparse.indices):
        raise InvalidEmbeddingError("sparse embedding has negative indices")
    return sparse


def _embedding_supports_sparse(embedding: EmbeddingComponent) -> bool:
    flag = getattr(embedding, "supports_sparse", None)
    if isinstance(flag, bool):
        return flag
    return callable(getattr(embedding, "embed_sparse", None))


def _validate_settings_object(settings: Settings) -> None:
    checks = (
        (bool(str(settings.collection_name).strip()), "collection_name must be a non-empty string"),
        (settings.top_k > 0, "top_k must be > 0"),
        (settings.max_top_k > 0, "max_top_k must be > 0"),
        (settings.top_k <= settings.max_top_k, "top_k exceeds max_top_k"),
        (settings.timeout_seconds > 0, "timeout_seconds must be > 0"),
        (settings.embedding_timeout_seconds > 0, "embedding_timeout_seconds must be > 0"),
        (settings.embedding_dimension > 0, "embedding_dimension must be > 0"),
        (bool(settings.embedding_model), "embedding_model is required"),
        (bool(settings.embedding_fingerprint), "embedding_fingerprint is required"),
        (settings.executor_max_workers > 0, "executor_max_workers must be > 0"),
    )
    for ok, message in checks:
        if not ok:
            raise InvalidConfigurationError(message)
    normalize_distance(settings.distance_metric)


def _validate_query(query: Query) -> None:
    if not isinstance(query.text, str):
        raise InvalidQueryError("query text must be a string")
    if query.text == "" or query.text.isspace():
        raise EmptyQueryError("query text is empty")
    if "\x00" in query.text:
        raise InvalidQueryError("query text contains NUL bytes")
    if _CONTROL_CHARS.search(query.text):
        raise InvalidQueryError("query text contains disallowed control characters")


def _resolve_top_k(query: Query, settings: Settings, override: int | None) -> int:
    value = settings.top_k
    if query.metadata.top_k is not None:
        value = query.metadata.top_k
    if override is not None:
        value = override
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidTopKError(f"top_k must be a positive int, got {type(value).__name__}")
    if value <= 0:
        raise InvalidTopKError("top_k must be > 0")
    if value > settings.max_top_k:
        raise InvalidTopKError(f"top_k {value} exceeds max_top_k {settings.max_top_k}")
    return value


def _coerce_query(query: Query | Mapping[str, Any]) -> Query:
    if isinstance(query, Query):
        return query
    if isinstance(query, Mapping):
        try:
            return Query.model_validate(query)
        except ValidationError as exc:
            raise InvalidQueryError("query contract validation failed") from exc
    raise InvalidQueryError(f"query must be Query or mapping, got {type(query).__name__}")


def _payload_str(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    return None


def _payload_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Mapping) and "code" in item:
                out.append(str(item["code"]))
            else:
                out.append(str(item))
        return out
    return [str(value)]


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_point_id(raw: Any) -> str | None:
    if isinstance(raw, SearchHit):
        return str(raw.id) if raw.id is not None else None
    if isinstance(raw, Mapping) and "id" in raw:
        return str(raw["id"])
    return None


# ===========================================================================
# Qdrant adapter — optional import; fusion stays in Qdrant when hybrid
# ===========================================================================


def _translate_qdrant_exception(exc: BaseException, *, timeout_seconds: float) -> RetrievalError:
    if isinstance(exc, RetrievalError):
        return exc
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return QdrantTimeoutError(
            f"Qdrant search exceeded timeout of {timeout_seconds}s",
            cause=exc,
        )
    name = type(exc).__name__
    message = str(exc)
    lowered = message.lower()
    if "timeout" in lowered or name in {"TimeoutException", "ReadTimeout", "ConnectTimeout"}:
        return QdrantTimeoutError(f"Qdrant timeout: {name}", cause=exc)
    if isinstance(exc, UnexpectedResponse) or "404" in message or "not found" in lowered:
        if "collection" in lowered or "404" in message:
            return MissingCollectionError(f"Qdrant collection missing: {message}", cause=exc)
    if isinstance(exc, ResponseHandlingException) or name in {
        "ConnectError",
        "ConnectTimeout",
        "HTTPError",
        "ConnectionError",
        "RemoteProtocolError",
    }:
        return QdrantUnavailableError(f"Qdrant unavailable: {name}", cause=exc)
    return SearchFailureError(f"Qdrant search failed: {name}: {message}", cause=exc)


def _parse_qdrant_vectors(info: Any) -> tuple[int | None, DistanceMetric | None, bool, bool]:
    try:
        vectors = info.config.params.vectors
    except Exception as exc:
        raise InvalidPayloadError("collection info missing vector params") from exc
    hybrid = False
    named = False
    size = None
    distance = None
    if isinstance(vectors, Mapping):
        named = True
        dense = vectors.get("dense") or next(iter(vectors.values()), None)
        hybrid = any(
            getattr(cfg, "modifier", None) is not None or name in {"bm25", "sparse"}
            for name, cfg in vectors.items()
        )
        if dense is not None:
            size = getattr(dense, "size", None)
            raw = getattr(dense, "distance", None)
            if raw is not None:
                distance = normalize_distance(str(raw))
    else:
        size = getattr(vectors, "size", None)
        raw_distance = getattr(vectors, "distance", None)
        if raw_distance is not None:
            distance = normalize_distance(str(raw_distance))
    return size, distance, named, hybrid


def _parse_qdrant_payload_schema(info: Any) -> list[str]:
    schema = getattr(info, "payload_schema", None) or {}
    if isinstance(schema, Mapping):
        return list(schema.keys())
    return []


def _to_qdrant_filter(filters: Mapping[str, Any] | None) -> Any:
    if not filters:
        return None
    if qmodels is None:
        raise SearchFailureError("cannot apply payload filters: qdrant_client models unavailable")
    must: list[Any] = []
    for key, value in filters.items():
        if isinstance(value, list):
            must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=value)))
        else:
            must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value)))
    return qmodels.Filter(must=must) if must else None


def _hits_from_qdrant_points(points: Any) -> list[SearchHit]:
    if points is None:
        raise MalformedQdrantResultError("Qdrant returned no result object")
    if hasattr(points, "points"):
        points = points.points
    if not isinstance(points, list):
        raise MalformedQdrantResultError(
            f"Qdrant result is not a list: {type(points).__name__}"
        )
    hits: list[SearchHit] = []
    for item in points:
        payload = getattr(item, "payload", None)
        score = getattr(item, "score", None)
        point_id = getattr(item, "id", None)
        if isinstance(item, Mapping):
            payload = item.get("payload")
            score = item.get("score")
            point_id = item.get("id")
        if score is None:
            raise MalformedQdrantResultError("Qdrant hit missing score")
        try:
            hits.append(SearchHit(id=point_id, score=float(score), payload=payload))
        except (TypeError, ValueError, ValidationError) as exc:
            raise MalformedQdrantResultError("Qdrant hit could not be normalized") from exc
    return hits


class QdrantVectorStore(VectorStore):
    """Async Qdrant adapter. Hybrid score fusion stays inside Qdrant.

    The module still imports if ``qdrant_client`` is missing; constructing this
    adapter without an injected client then raises ``QdrantUnavailableError``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        sync_client: Any | None = None,
    ) -> None:
        if (
            AsyncQdrantClient is None
            and QdrantClient is None
            and client is None
            and sync_client is None
        ):
            detail = f": {_QDRANT_IMPORT_ERROR}" if _QDRANT_IMPORT_ERROR else ""
            raise QdrantUnavailableError(
                f"qdrant_client is not installed{detail}"
            )
        self._settings = settings or get_settings()
        if not self._settings.qdrant_url and client is None and sync_client is None:
            raise InvalidConfigurationError("qdrant_url is required for QdrantVectorStore")
        self._client = client
        self._sync_client = sync_client
        self._lock = asyncio.Lock()

    def _connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "url": self._settings.qdrant_url,
            "timeout": self._settings.timeout_seconds,
        }
        if self._settings.qdrant_api_key:
            kwargs["api_key"] = self._settings.qdrant_api_key
        return kwargs

    async def _ensure_async_client(self) -> Any:
        if self._client is not None:
            return self._client
        if AsyncQdrantClient is None:
            return None
        async with self._lock:
            if self._client is None:
                self._client = AsyncQdrantClient(**self._connect_kwargs())
        return self._client

    def _ensure_sync_client(self) -> Any:
        if self._sync_client is not None:
            return self._sync_client
        if QdrantClient is None:
            raise QdrantUnavailableError("qdrant_client is not available")
        self._sync_client = QdrantClient(**self._connect_kwargs())
        return self._sync_client

    async def _call(self, async_name: str, sync_name: str, **kwargs: Any) -> Any:
        client = await self._ensure_async_client()
        if client is not None and hasattr(client, async_name):
            method = getattr(client, async_name)
            result = method(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        sync_client = self._ensure_sync_client()
        method = getattr(sync_client, sync_name)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: method(**kwargs))

    async def health(self) -> bool:
        try:
            await self._call("get_collections", "get_collections")
            return True
        except Exception as exc:
            mapped = _translate_qdrant_exception(
                exc, timeout_seconds=self._settings.timeout_seconds
            )
            if isinstance(mapped, QdrantTimeoutError):
                raise mapped from exc
            if isinstance(mapped, QdrantUnavailableError):
                return False
            raise mapped from exc

    async def get_collection_info(self, collection_name: str) -> CollectionInfo:
        try:
            info = await self._call(
                "get_collection",
                "get_collection",
                collection_name=collection_name,
            )
        except Exception as exc:
            mapped = _translate_qdrant_exception(
                exc, timeout_seconds=self._settings.timeout_seconds
            )
            if isinstance(mapped, MissingCollectionError):
                return CollectionInfo(name=collection_name, exists=False)
            raise mapped from exc

        size, distance, named, hybrid = _parse_qdrant_vectors(info)
        payload_fields = _parse_qdrant_payload_schema(info)
        meta_model, meta_fp = await self._read_index_meta(collection_name)
        return CollectionInfo(
            name=collection_name,
            exists=True,
            vector_size=size,
            distance=distance,
            embedding_model=meta_model,
            embedding_fingerprint=meta_fp,
            payload_schema_fields=payload_fields,
            hybrid_enabled=hybrid,
            sparse_vector_name=self._settings.sparse_vector_name if hybrid else None,
            dense_vector_name=self._settings.dense_vector_name if named else None,
        )

    async def _read_index_meta(self, collection_name: str) -> tuple[str | None, str | None]:
        point_id = self._settings.index_meta_point_id
        try:
            records = await self._call(
                "retrieve",
                "retrieve",
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
            )
        except Exception as exc:
            mapped = _translate_qdrant_exception(
                exc, timeout_seconds=self._settings.timeout_seconds
            )
            if isinstance(mapped, MissingCollectionError):
                return None, None
            raise mapped from exc
        if not records:
            return None, None
        record = records[0]
        payload = getattr(record, "payload", None) or {}
        if not isinstance(payload, Mapping):
            return None, None
        model = payload.get("embedding_model")
        fingerprint = payload.get("embedding_fingerprint")
        return (
            str(model) if model is not None else None,
            str(fingerprint) if fingerprint is not None else None,
        )

    async def search(
        self,
        *,
        collection_name: str,
        dense: Sequence[float],
        sparse: SparseEmbedding | None,
        top_k: int,
        filters: Mapping[str, Any] | None,
        score_threshold: float | None,
        timeout_seconds: float,
        hybrid: bool,
    ) -> list[SearchHit]:
        query_filter = _to_qdrant_filter(filters)
        dense_list = list(dense)
        try:
            points = await self._query_points(
                collection_name=collection_name,
                dense=dense_list,
                sparse=sparse if hybrid else None,
                top_k=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                timeout_seconds=timeout_seconds,
                hybrid=hybrid,
            )
        except Exception as exc:
            raise _translate_qdrant_exception(exc, timeout_seconds=timeout_seconds) from exc
        return _hits_from_qdrant_points(points)

    async def _query_points(
        self,
        *,
        collection_name: str,
        dense: list[float],
        sparse: SparseEmbedding | None,
        top_k: int,
        query_filter: Any,
        score_threshold: float | None,
        timeout_seconds: float,
        hybrid: bool,
    ) -> Any:
        named = self._settings.use_named_vectors or hybrid
        using = self._settings.dense_vector_name if named else None
        client = await self._ensure_async_client()
        timeout = int(max(1, math.ceil(timeout_seconds)))

        if hybrid and sparse is not None and qmodels is not None and client is not None:
            if hasattr(client, "query_points"):
                prefetch = [
                    qmodels.Prefetch(
                        query=dense,
                        using=self._settings.dense_vector_name,
                        limit=top_k,
                    ),
                    qmodels.Prefetch(
                        query=qmodels.SparseVector(indices=sparse.indices, values=sparse.values),
                        using=self._settings.sparse_vector_name,
                        limit=top_k,
                    ),
                ]
                result = await client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch,
                    query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    timeout=timeout,
                )
                return getattr(result, "points", result)

        kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "limit": top_k,
            "with_payload": True,
            "query_vector": (using, dense) if using else dense,
            "timeout": timeout,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold
        return await self._call("search", "search", **kwargs)


# ===========================================================================
# Startup validation — fail closed, never warn-only
# ===========================================================================


async def validate_index_consistency(
    embedding: EmbeddingComponent,
    vector_store: VectorStore,
    settings: Settings,
) -> CollectionInfo:
    if embedding.model_id != settings.embedding_model:
        raise IndexConsistencyError(
            f"embedding model_id {embedding.model_id!r} != settings {settings.embedding_model!r}"
        )
    if embedding.dimension != settings.embedding_dimension:
        raise IndexConsistencyError(
            f"embedding dimension {embedding.dimension} != settings {settings.embedding_dimension}"
        )
    if embedding.fingerprint != settings.embedding_fingerprint:
        raise IndexConsistencyError(
            "embedding fingerprint does not match settings.embedding_fingerprint"
        )
    if settings.hybrid_search_enabled and not _embedding_supports_sparse(embedding):
        raise IndexConsistencyError(
            "hybrid_search_enabled but embedding component does not support sparse vectors"
        )

    try:
        healthy = await vector_store.health()
    except RetrievalError:
        raise
    except Exception as exc:
        raise QdrantUnavailableError("vector store health check failed", cause=exc) from exc
    if not healthy:
        raise QdrantUnavailableError("vector store failed health check at startup")

    try:
        info = await vector_store.get_collection_info(settings.collection_name)
    except RetrievalError:
        raise
    except Exception as exc:
        raise SearchFailureError("failed to read collection info", cause=exc) from exc

    if not info.exists:
        raise MissingCollectionError(f"collection {settings.collection_name!r} does not exist")

    if info.vector_size is None or info.vector_size != settings.embedding_dimension:
        raise IndexConsistencyError(
            f"index vector size {info.vector_size} != settings {settings.embedding_dimension}"
        )
    if info.distance is None or normalize_distance(info.distance) != settings.distance_metric:
        raise IndexConsistencyError(
            f"index distance {info.distance!r} != settings {settings.distance_metric!r}"
        )
    if not info.embedding_model:
        raise IndexConsistencyError("collection missing embedding model identity")
    if info.embedding_model != settings.embedding_model:
        raise IndexConsistencyError(
            f"index embedding model {info.embedding_model!r} != {settings.embedding_model!r}"
        )
    if not info.embedding_fingerprint:
        raise IndexConsistencyError("collection missing embedding fingerprint metadata")
    if info.embedding_fingerprint != settings.embedding_fingerprint:
        raise IndexConsistencyError("index embedding fingerprint mismatch")
    expected = set(settings.expected_payload_fields)
    present = set(info.payload_schema_fields)
    missing = expected - present
    if missing:
        raise IndexConsistencyError(
            f"collection payload schema missing fields: {sorted(missing)}"
        )
    if settings.hybrid_search_enabled and not info.hybrid_enabled:
        raise IndexConsistencyError("hybrid search configured but collection has no sparse vectors")
    return info


# ===========================================================================
# Retriever
# ===========================================================================


class Retriever:
    """Retrieval-only orchestrator. Constructor-injected; no hidden retrieval globals."""

    def __init__(
        self,
        embedding: EmbeddingComponent,
        vector_store: VectorStore,
        settings: Settings | None = None,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        _validate_settings_object(self._settings)
        if embedding.dimension != self._settings.embedding_dimension:
            raise InvalidConfigurationError(
                f"embedder dimension {embedding.dimension} != settings {self._settings.embedding_dimension}"
            )
        if embedding.model_id != self._settings.embedding_model:
            raise InvalidConfigurationError(
                f"embedder model_id {embedding.model_id!r} != settings {self._settings.embedding_model!r}"
            )
        if embedding.fingerprint != self._settings.embedding_fingerprint:
            raise InvalidConfigurationError("embedder fingerprint does not match settings")
        self._embedding = embedding
        self._store = vector_store
        if executor is not None:
            self._executor = executor
            self._owns_executor = False
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=self._settings.executor_max_workers,
                thread_name_prefix="rag-embed",
            )
            self._owns_executor = True
        self._started = False
        self._start_lock = asyncio.Lock()
        self._collection_info: CollectionInfo | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def embedding(self) -> EmbeddingComponent:
        return self._embedding

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    async def startup(self) -> CollectionInfo:
        info = await validate_index_consistency(self._embedding, self._store, self._settings)
        self._collection_info = info
        self._started = True
        _log_event(
            "retriever_startup_ok",
            collection=self._settings.collection_name,
            embedding_model=self._embedding.model_id,
            dimension=self._embedding.dimension,
            fingerprint_prefix=self._embedding.fingerprint[:12],
        )
        return info

    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if not self._started:
                await self.startup()

    async def retrieve(
        self,
        query: Query | Mapping[str, Any],
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Fail-fast retrieval. No internal retry. Empty hits are valid, not a refusal."""

        total_t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex
        metrics = RetrievalMetrics()
        trace_token = _trace_id_var.set(trace_id)
        metrics_token = _metrics_var.set(metrics)
        try:
            parsed = _coerce_query(query)
            _validate_query(parsed)
            resolved_k = _resolve_top_k(parsed, self._settings, top_k)
            threshold = (
                parsed.metadata.score_threshold
                if parsed.metadata.score_threshold is not None
                else self._settings.score_threshold
            )
            filters = merge_filters(
                self._settings.default_filters,
                parsed.metadata.filters,
                parsed.metadata.ontology_codes,
                parsed.metadata.source_types,
            )
            polarity = parsed.metadata.collection_polarity or self._settings.collection_polarity

            _log_event(
                "retrieve_start",
                query_length=len(parsed.text),
                query_hash=query_hash(parsed.text),
                top_k=resolved_k,
                collection=self._settings.collection_name,
            )

            await self._ensure_started()

            dense, sparse, embed_t = await self._embed_query(parsed.text)
            metrics.embedding_time = embed_t
            hits, search_t = await self._search(
                dense=dense,
                sparse=sparse,
                top_k=resolved_k,
                filters=filters,
                score_threshold=threshold,
            )
            metrics.vector_search_time = search_t
            candidates, dropped_reasons = self._validate_candidates(hits, metrics)

            is_degraded = bool(dropped_reasons)
            data_quality_failure = bool(hits) and not candidates and is_degraded
            error_code: str | None = None
            if data_quality_failure:
                error_code = "data_quality_failure"
            elif is_degraded:
                error_code = "degraded_partial"

            semantics_cfg = self._settings.score_semantics
            fusion_owner: Literal["vector_store", "none"] = (
                semantics_cfg.fusion_owner if self._settings.hybrid_search_enabled else "none"
            )
            metrics.total_retrieval_time = time.perf_counter() - total_t0
            result = RetrievalResult(
                query=parsed,
                candidates=candidates,
                is_degraded=is_degraded,
                dropped_candidates_count=metrics.dropped_candidate_count,
                data_quality_failure=data_quality_failure,
                error_code=error_code,
                metrics=metrics,
                score_semantics=ScoreSemantics(
                    metric=semantics_cfg.metric,
                    higher_is_better=semantics_cfg.higher_is_better,
                    threshold_applied=threshold,
                    threshold_mode=semantics_cfg.threshold_mode,
                    fusion_owner=fusion_owner,
                    polarity=polarity,
                    description=semantics_cfg.description,
                ),
                trace_id=trace_id,
                dropped_reasons=dropped_reasons,
            )
            _log_event(
                "retrieve_done",
                query_length=len(parsed.text),
                query_hash=query_hash(parsed.text),
                candidate_count=metrics.candidate_count,
                valid_candidate_count=metrics.valid_candidate_count,
                dropped_candidate_count=metrics.dropped_candidate_count,
                is_degraded=is_degraded,
                data_quality_failure=data_quality_failure,
                embedding_time=round(metrics.embedding_time, 6),
                vector_search_time=round(metrics.vector_search_time, 6),
                candidate_validation_time=round(metrics.candidate_validation_time, 6),
                total_retrieval_time=round(metrics.total_retrieval_time, 6),
            )
            return result
        finally:
            _trace_id_var.reset(trace_token)
            _metrics_var.reset(metrics_token)

    async def _embed_query(self, text: str) -> tuple[list[float], SparseEmbedding | None, float]:
        t0 = time.perf_counter()
        try:
            dense_raw = await asyncio.wait_for(
                self._run_sync_embed(self._embedding.embed_dense, text),
                timeout=self._settings.embedding_timeout_seconds,
            )
        except TimeoutError as exc:
            raise EmbeddingFailureError("embedding timed out") from exc
        except (InvalidEmbeddingError, DimensionMismatchError, EmbeddingFailureError):
            raise
        except Exception as exc:
            raise EmbeddingFailureError("embedding component failed") from exc

        try:
            dense = validate_dense_vector(dense_raw, self._settings.embedding_dimension)
        except (InvalidEmbeddingError, DimensionMismatchError):
            raise
        except Exception as exc:
            raise InvalidEmbeddingError("embedding failed validation") from exc

        sparse: SparseEmbedding | None = None
        if self._settings.hybrid_search_enabled:
            embed_sparse = getattr(self._embedding, "embed_sparse", None)
            if not callable(embed_sparse):
                raise EmbeddingFailureError("hybrid search enabled but embed_sparse is missing")
            try:
                sparse_raw = await asyncio.wait_for(
                    self._run_sync_embed(embed_sparse, text),
                    timeout=self._settings.embedding_timeout_seconds,
                )
            except TimeoutError as exc:
                raise EmbeddingFailureError("sparse embedding timed out") from exc
            except (InvalidEmbeddingError, EmbeddingFailureError):
                raise
            except Exception as exc:
                raise EmbeddingFailureError("sparse embedding failed") from exc
            if sparse_raw is None:
                raise EmbeddingFailureError("hybrid search enabled but sparse embedding is None")
            if not isinstance(sparse_raw, SparseEmbedding):
                try:
                    sparse_raw = SparseEmbedding.model_validate(sparse_raw)
                except ValidationError as exc:
                    raise InvalidEmbeddingError("sparse embedding contract validation failed") from exc
            sparse = validate_sparse_vector(sparse_raw)

        return dense, sparse, time.perf_counter() - t0

    async def _run_sync_embed(self, fn: Any, text: str) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(text)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, text)

    async def _search(
        self,
        *,
        dense: Sequence[float],
        sparse: SparseEmbedding | None,
        top_k: int,
        filters: Mapping[str, Any],
        score_threshold: float | None,
    ) -> tuple[list[SearchHit], float]:
        t0 = time.perf_counter()
        timeout = self._settings.timeout_seconds
        try:
            hits = await asyncio.wait_for(
                self._store.search(
                    collection_name=self._settings.collection_name,
                    dense=dense,
                    sparse=sparse,
                    top_k=top_k,
                    filters=filters or None,
                    score_threshold=score_threshold,
                    timeout_seconds=timeout,
                    hybrid=self._settings.hybrid_search_enabled,
                ),
                timeout=timeout,
            )
        except RetrievalError:
            raise
        except TimeoutError as exc:
            raise QdrantTimeoutError(
                f"vector search exceeded timeout of {timeout}s",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise SearchFailureError("vector search failed", cause=exc) from exc

        elapsed = time.perf_counter() - t0
        if hits is None or not isinstance(hits, list):
            raise MalformedQdrantResultError(
                f"vector store returned {type(hits).__name__}, expected list"
            )
        return hits[:top_k], elapsed

    def _validate_candidates(
        self,
        hits: Sequence[Any],
        metrics: RetrievalMetrics,
    ) -> tuple[list[Candidate], list[str]]:
        t0 = time.perf_counter()
        candidates: list[Candidate] = []
        dropped: list[str] = []
        for raw in hits:
            candidate, reason = self._try_build_candidate(raw)
            if candidate is None:
                dropped.append(reason or "malformed_candidate")
                logger.warning(
                    "%s",
                    {
                        "event": "malformed_candidate_dropped",
                        "trace_id": current_trace_id(),
                        "reason": reason,
                        "point_id": _safe_point_id(raw),
                    },
                )
                continue
            candidates.append(candidate)
        metrics.candidate_validation_time = time.perf_counter() - t0
        metrics.candidate_count = len(hits)
        metrics.valid_candidate_count = len(candidates)
        metrics.dropped_candidate_count = len(dropped)
        return candidates, dropped

    def _try_build_candidate(self, raw: Any) -> tuple[Candidate | None, str | None]:
        try:
            if isinstance(raw, SearchHit):
                hit = raw
            elif isinstance(raw, Mapping):
                hit = SearchHit.model_validate(raw)
            else:
                return None, "malformed_hit_type"
        except ValidationError:
            return None, "malformed_hit_schema"
        if not _is_finite_number(hit.score):
            return None, "non_finite_score"
        payload = hit.payload
        if payload is None:
            return None, "missing_payload"
        if not isinstance(payload, Mapping):
            return None, "invalid_payload_type"
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return None, "missing_text"
        chunk_id = _payload_str(payload, "chunk_id")
        if chunk_id is None:
            if hit.id is not None and str(hit.id).strip() and str(hit.id) != "None":
                chunk_id = str(hit.id)
            else:
                return None, "missing_chunk_id"
        document_id = _payload_str(payload, "document_id", "source_id", "doc_id")
        if document_id is None:
            return None, "missing_document_id"
        source = _payload_str(payload, "source", "source_id")
        if source is None:
            return None, "missing_source"
        source_id = _payload_str(payload, "source_id") or source
        metadata = {k: v for k, v in payload.items() if k != "text"}
        try:
            candidate = Candidate(
                chunk_id=chunk_id,
                document_id=document_id,
                text=text,
                score=float(hit.score),
                metadata=metadata,
                source=source,
                source_id=source_id,
                loe=_payload_str(payload, "loe"),
                ontology_codes=_payload_str_list(payload.get("ontology_codes")),
                point_id=str(hit.id) if hit.id is not None else None,
            )
        except ValidationError:
            return None, "candidate_validation_error"
        return candidate, None

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False)

    async def aclose(self) -> None:
        self.close()

    async def __aenter__(self) -> Retriever:
        await self.startup()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


if __name__ == "__main__":
    print("Usage: await Retriever(embedding, QdrantVectorStore(settings), settings).retrieve(Query(text=...))")
