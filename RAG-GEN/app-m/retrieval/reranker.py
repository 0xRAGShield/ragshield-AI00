"""reranker.py - Strictly isolated, standalone Reranker for Medical RAG.

Pipeline:
    RetrievalResult → IRerankerModel (Cross-Encoder) → Candidate Truncation & Filtering → RerankedResult

Principles:
- Zero PII Logging: Queries are hashed, candidate texts are never logged.
- Graceful Degradation: If the model times out or fails, gracefully falls back to retrieval results.
- Async Execution: Model execution is offloaded to a thread pool to avoid blocking the event loop.
- Fail-Safe Sorting: None scores are robustly pushed to the bottom without triggering TypeErrors.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ============================================================================
# 1. DUMMY SETTINGS (Stub)
# ============================================================================
class RerankerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_n: int = 5
    max_top_n: int = 20
    score_threshold: float = 0.3
    higher_is_better: bool = True
    timeout_seconds: float = 5.0
    executor_max_workers: int = 4
    fallback_to_retrieval: bool = True  # If True, degrade gracefully on error. If False, fail fast.

# ============================================================================
# 2. EXCEPTIONS (Fail-Fast Policy)
# ============================================================================
class RerankerError(Exception):
    error_code: str = "reranker_error"
    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

class InvalidRerankerConfigurationError(RerankerError):
    error_code = "invalid_configuration"

class RerankerTimeoutError(RerankerError):
    error_code = "reranker_timeout"

class RerankerFailureError(RerankerError):
    error_code = "reranker_failure"

# ============================================================================
# 3. CONTRACTS & MODELS (Stub)
# ============================================================================
# Using simplified Retrieval stubs to allow standalone execution
class Query(BaseModel):
    text: str
    request_id: str | None = None

class Candidate(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str

class RetrievalResult(BaseModel):
    query: Query
    candidates: list[Candidate] = Field(default_factory=list)
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

class RerankedCandidate(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    original_score: float
    rerank_score: float | None
    was_reranked: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str
    original_rank: int

class RerankerMetrics(BaseModel):
    input_count: int = 0
    scored_count: int = 0
    dropped_by_threshold: int = 0
    final_count: int = 0
    scoring_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    used_fallback: bool = False

class RerankedResult(BaseModel):
    query: Query
    candidates: list[RerankedCandidate]
    is_fallback: bool
    metrics: RerankerMetrics
    trace_id: str
    error_code: str | None = None

# ============================================================================
# 4. OBSERVABILITY HELPERS
# ============================================================================
_trace_id_var: ContextVar[str] = ContextVar("rag_reranker_trace_id", default="")

def current_trace_id() -> str:
    return _trace_id_var.get()

def query_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

def _log_event(event: str, **fields: Any) -> None:
    logger.info("%s", {"event": event, "trace_id": current_trace_id(), **fields})

def _is_usable_text(text: str | None) -> bool:
    return isinstance(text, str) and bool(text.strip())

# ============================================================================
# 5. DEPENDENCY PROTOCOLS
# ============================================================================
class IRerankerModel(Protocol):
    """Injected model dependency. 
    Can be an async API client (e.g. Cohere) or a sync local model (e.g. BGE-M3).
    """
    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        ...

# ============================================================================
# 6. RERANKER IMPLEMENTATION
# ============================================================================
class Reranker:
    """Strictly bounded reranking orchestrator."""

    def __init__(
        self,
        model: IRerankerModel,
        settings: RerankerSettings | None = None,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._model = model
        self._settings = settings or RerankerSettings()
        
        if executor is not None:
            self._executor = executor
            self._owns_executor = False
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=self._settings.executor_max_workers,
                thread_name_prefix="rag-rerank"
            )
            self._owns_executor = True

    async def rerank(
        self,
        parsed: RetrievalResult,
        *,
        top_n: int | None = None,
    ) -> RerankedResult:
        """Main entry point. Scores, sorts, and filters candidates."""
        total_t0 = time.perf_counter()
        trace_id = parsed.trace_id or uuid.uuid4().hex
        token = _trace_id_var.set(trace_id)
        metrics = RerankerMetrics()

        try:
            resolved_n = self._resolve_top_n(top_n)
            
            # Fast Path: If Retrieval returned no candidates, do not invoke the model.
            if not parsed.candidates:
                return self._empty_result(parsed, metrics, trace_id, total_t0)

            indexed = [
                (rank, candidate) 
                for rank, candidate in enumerate(parsed.candidates)
                if _is_usable_text(candidate.text)
            ]
            metrics.input_count = len(indexed)

            if not indexed:
                return self._empty_result(parsed, metrics, trace_id, total_t0)

            try:
                scored = await self._score(parsed.query.text, indexed, metrics)
            except (RerankerTimeoutError, RerankerFailureError) as exc:
                if self._settings.fallback_to_retrieval:
                    fallback = self._fallback_result(
                        parsed, resolved_n, metrics, trace_id, error_code=exc.error_code, started_at=total_t0
                    )
                    _log_event(
                        "rerank_fallback",
                        query_hash=query_hash(parsed.query.text),
                        input_count=metrics.input_count,
                        final_count=fallback.metrics.final_count,
                        used_fallback=True,
                        error_code=exc.error_code,
                    )
                    return fallback
                raise

            ranked = self._filter_and_truncate(scored, resolved_n, metrics)
            metrics.final_count = len(ranked)
            metrics.execution_time_ms = (time.perf_counter() - total_t0) * 1000.0

            _log_event(
                "rerank_done",
                query_hash=query_hash(parsed.query.text),
                input_count=metrics.input_count,
                dropped_by_threshold=metrics.dropped_by_threshold,
                final_count=metrics.final_count,
                execution_time_ms=round(metrics.execution_time_ms, 3),
                used_fallback=False,
            )

            return RerankedResult(
                query=parsed.query,
                candidates=ranked,
                is_fallback=False,
                metrics=metrics,
                trace_id=trace_id,
            )
        finally:
            _trace_id_var.reset(token)

    def _resolve_top_n(self, override: int | None) -> int:
        value = self._settings.top_n if override is None else override
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise InvalidRerankerConfigurationError(f"invalid top_n: {value!r}")
        if value > self._settings.max_top_n:
            raise InvalidRerankerConfigurationError(f"top_n exceeds max_top_n")
        return value

    def _empty_result(
        self, parsed: RetrievalResult, metrics: RerankerMetrics, trace_id: str, started_at: float
    ) -> RerankedResult:
        """Fast-path exit when no valid candidates were provided."""
        metrics.execution_time_ms = (time.perf_counter() - started_at) * 1000.0
        return RerankedResult(
            query=parsed.query,
            candidates=[],
            is_fallback=False,
            metrics=metrics,
            trace_id=trace_id,
        )

    async def _score(
        self,
        query_text: str,
        indexed: Sequence[tuple[int, Candidate]],
        metrics: RerankerMetrics,
    ) -> list[RerankedCandidate]:
        texts = [candidate.text for _, candidate in indexed]
        t0 = time.perf_counter()
        
        raw_scores = await self._call_model(query_text, texts)
        
        metrics.scoring_time_ms = (time.perf_counter() - t0) * 1000.0
        scores = self._validate_scores(raw_scores, expected_length=len(texts))
        metrics.scored_count = len([s for s in scores if s is not None])

        reranked: list[RerankedCandidate] = []
        try:
            for (original_rank, candidate), score in zip(indexed, scores, strict=True):
                reranked.append(
                    RerankedCandidate(
                        chunk_id=candidate.chunk_id,
                        document_id=candidate.document_id,
                        text=candidate.text,
                        original_score=candidate.score,
                        rerank_score=score,
                        was_reranked=True,
                        metadata=candidate.metadata,
                        source=candidate.source,
                        original_rank=original_rank,
                    )
                )
        except ValueError as exc:
            # FIX: Caught strict=True ValueError to prevent raw exception leakage.
            raise RerankerFailureError("Model returned incorrect number of scores", cause=exc)

        # FIX: Fail-safe sorting logic. Valid items are sorted properly, None items are appended at the bottom.
        valid_items = [item for item in reranked if item.rerank_score is not None]
        none_items = [item for item in reranked if item.rerank_score is None]
        
        valid_items.sort(key=lambda item: item.rerank_score, reverse=self._settings.higher_is_better)
        return valid_items + none_items

    async def _call_model(self, query_text: str, texts: list[str]) -> Any:
        timeout = self._settings.timeout_seconds
        try:
            return await asyncio.wait_for(self._invoke_score_pairs(query_text, texts), timeout=timeout)
        except TimeoutError as exc:
            raise RerankerTimeoutError(f"Reranker exceeded timeout of {timeout}s", cause=exc)
        except RerankerError:
            raise
        except Exception as exc:
            raise RerankerFailureError("Reranker model failed internally", cause=exc)

    async def _invoke_score_pairs(self, query_text: str, texts: list[str]) -> Any:
        fn = self._model.score_pairs
        unbound = getattr(fn, "__func__", fn)
        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(unbound):
            return await fn(query_text, texts)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, query_text, texts)

    def _validate_scores(self, raw_scores: Any, expected_length: int) -> list[float | None]:
        """Ensures the output is strictly a list of floats or Nones, avoiding NaNs/Infs."""
        if not isinstance(raw_scores, list):
            raise RerankerFailureError(f"Expected model to return a list, got {type(raw_scores).__name__}")
        
        if len(raw_scores) != expected_length:
            raise RerankerFailureError(f"Score count mismatch. Expected {expected_length}, got {len(raw_scores)}")

        cleaned: list[float | None] = []
        for s in raw_scores:
            if s is None:
                cleaned.append(None)
                continue
            try:
                val = float(s)
                if not math.isfinite(val):
                    cleaned.append(None)
                else:
                    cleaned.append(val)
            except (TypeError, ValueError):
                cleaned.append(None)
        return cleaned

    def _filter_and_truncate(
        self,
        scored: Sequence[RerankedCandidate],
        top_n: int,
        metrics: RerankerMetrics,
    ) -> list[RerankedCandidate]:
        threshold = self._settings.score_threshold
        higher_is_better = self._settings.higher_is_better
        kept: list[RerankedCandidate] = []
        
        for item in scored:
            score = item.rerank_score
            if score is None:
                metrics.dropped_by_threshold += 1
                continue
            
            passes = (score >= threshold) if higher_is_better else (score <= threshold)
            if not passes:
                metrics.dropped_by_threshold += 1
                continue
                
            kept.append(item)
            
        return kept[:top_n]

    def _fallback_result(
        self,
        parsed: RetrievalResult,
        top_n: int,
        metrics: RerankerMetrics,
        trace_id: str,
        *,
        error_code: str,
        started_at: float,
    ) -> RerankedResult:
        """Original retrieval order, truncated to top_n, without rerank scores."""
        fallback_items = []
        for rank, candidate in enumerate(parsed.candidates):
            if not _is_usable_text(candidate.text):
                continue
            fallback_items.append(
                RerankedCandidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    text=candidate.text,
                    original_score=candidate.score,
                    rerank_score=None,
                    was_reranked=False,
                    metadata=candidate.metadata,
                    source=candidate.source,
                    original_rank=rank,
                )
            )
            
        fallback_items = fallback_items[:top_n]
        metrics.dropped_by_threshold = 0
        metrics.final_count = len(fallback_items)
        metrics.used_fallback = True
        metrics.execution_time_ms = (time.perf_counter() - started_at) * 1000.0
        
        return RerankedResult(
            query=parsed.query,
            candidates=fallback_items,
            is_fallback=True,
            metrics=metrics,
            trace_id=trace_id,
            error_code=error_code,
        )

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False)

    async def aclose(self) -> None:
        self.close()