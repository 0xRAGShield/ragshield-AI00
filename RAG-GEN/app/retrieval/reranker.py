from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any, Protocol, Sequence

from app.core.settings import RerankerSettings
from app.models.rag_models import (
    RerankedCandidate,
    RerankedResult,
    RetrievalCandidate,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

_trace_id_var: ContextVar[str] = ContextVar(
    "ragshield_reranker_trace_id",
    default="",
)


# ============================================================================
# Exceptions
# ============================================================================

class RerankerError(RuntimeError):
    """Base exception for reranker failures."""


class InvalidRerankerConfigurationError(RerankerError):
    """Raised when reranker configuration is invalid."""


class InvalidRerankerInputError(RerankerError):
    """Raised when the retrieval result violates the reranker contract."""


class RerankerTimeoutError(RerankerError):
    """Raised when reranker execution exceeds its timeout."""


class RerankerFailureError(RerankerError):
    """Raised when the reranker model fails."""


class InvalidRerankerOutputError(RerankerError):
    """Raised when the reranker model returns invalid output."""


# ============================================================================
# Reranker Model Contract
# ============================================================================

class IRerankerModel(Protocol):
    """
    Replaceable reranking-model contract.

    The implementation may be:
    - a local cross-encoder
    - another local reranker
    - a remote reranking provider
    """

    def score_pairs(
        self,
        query: str,
        texts: list[str],
    ) -> list[float | None]:
        ...


# ============================================================================
# Observability
# ============================================================================

def current_trace_id() -> str:
    return _trace_id_var.get()


def _log_event(event: str, **fields: Any) -> None:
    """
    Structured logging without candidate text or query contents.
    """

    logger.info(
        "%s",
        {
            "event": event,
            "trace_id": current_trace_id(),
            **fields,
        },
    )


# ============================================================================
# Reranker
# ============================================================================

class Reranker:
    """
    Production reranking orchestrator.

    Runtime flow:

        RetrievalResult
            ↓
        validate candidates
            ↓
        Cross-Encoder
            ↓
        validate scores
            ↓
        sort
            ↓
        optional threshold
            ↓
        top_n
            ↓
        RerankedResult

    Responsibilities:
    - Consume canonical RetrievalResult.
    - Execute an injected reranking model.
    - Validate model output.
    - Rank candidates deterministically.
    - Apply optional score threshold.
    - Provide bounded fallback to retrieval results.
    - Keep synchronous model execution off the event loop.
    - Preserve traceability and execution metrics.

    Forbidden:
    - Retrieval.
    - Embeddings.
    - Vector-store access.
    - Chunking.
    - Context assembly.
    - Prompt construction.
    - Generation.
    - Hybrid retrieval.
    - Sparse retrieval.
    """

    def __init__(
        self,
        model: IRerankerModel,
        settings: RerankerSettings | None = None,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        if model is None:
            raise InvalidRerankerConfigurationError(
                "model cannot be None."
            )

        self._model = model
        self._settings = settings or RerankerSettings()

        self._owns_executor = executor is None

        self._executor = executor or ThreadPoolExecutor(
            max_workers=self._settings.executor_max_workers,
            thread_name_prefix="ragshield-reranker",
        )

        self._closed = False

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    async def rerank(
        self,
        retrieval_result: RetrievalResult,
        *,
        top_n: int | None = None,
    ) -> RerankedResult:
        """
        Rerank candidates returned by the canonical Retriever.
        """

        self._ensure_open()

        if not isinstance(retrieval_result, RetrievalResult):
            raise InvalidRerankerInputError(
                "retrieval_result must be a RetrievalResult instance."
            )

        started_at = time.perf_counter()

        trace_id = (
            retrieval_result.metrics.trace_id
            if retrieval_result.metrics.trace_id
            else uuid.uuid4().hex
        )

        token = _trace_id_var.set(trace_id)

        metrics = self._create_metrics()

        try:
            resolved_top_n = self._resolve_top_n(top_n)

            candidates = self._validate_input_candidates(
                retrieval_result.candidates
            )

            metrics.input_count = len(candidates)

            if not candidates:
                return self._build_empty_result(
                    retrieval_result=retrieval_result,
                    trace_id=trace_id,
                    metrics=metrics,
                    started_at=started_at,
                )

            try:
                reranked_candidates = await self._score_candidates(
                    query=retrieval_result.query.text,
                    candidates=candidates,
                    metrics=metrics,
                )

            except (
                RerankerTimeoutError,
                RerankerFailureError,
                InvalidRerankerOutputError,
            ) as error:

                if not self._settings.fallback_to_retrieval:
                    raise

                result = self._build_fallback_result(
                    retrieval_result=retrieval_result,
                    top_n=resolved_top_n,
                    trace_id=trace_id,
                    metrics=metrics,
                    error_code=self._error_code(error),
                    started_at=started_at,
                )

                _log_event(
                    "rerank_fallback",
                    input_count=metrics.input_count,
                    final_count=metrics.final_count,
                    used_fallback=True,
                    error_code=result.error_code,
                )

                return result

            ranked = self._apply_threshold_and_top_n(
                reranked_candidates,
                top_n=resolved_top_n,
                metrics=metrics,
            )

            metrics.final_count = len(ranked)
            metrics.execution_time_ms = (
                time.perf_counter() - started_at
            ) * 1000.0

            _log_event(
                "rerank_completed",
                input_count=metrics.input_count,
                scored_count=metrics.scored_count,
                dropped_by_threshold=metrics.dropped_by_threshold,
                final_count=metrics.final_count,
                scoring_time_ms=round(
                    metrics.scoring_time_ms,
                    3,
                ),
                execution_time_ms=round(
                    metrics.execution_time_ms,
                    3,
                ),
                used_fallback=False,
            )

            return RerankedResult(
                query=retrieval_result.query.text,
                candidates=ranked,
                is_fallback=False,
                metrics=metrics,
                trace_id=trace_id,
            )

        finally:
            _trace_id_var.reset(token)

    # ------------------------------------------------------------------------
    # Metrics construction
    # ------------------------------------------------------------------------

    @staticmethod
    def _create_metrics() -> Any:
        """
        Construct the canonical RerankerMetrics contract.

        Kept isolated so the orchestration logic does not depend on
        model construction details.
        """

        from app.models.rag_models import RerankerMetrics

        return RerankerMetrics()

    # ------------------------------------------------------------------------
    # Candidate validation
    # ------------------------------------------------------------------------

    def _validate_input_candidates(
        self,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[tuple[int, RetrievalCandidate]]:
        if (
            isinstance(candidates, (str, bytes))
            or not isinstance(candidates, Sequence)
        ):
            raise InvalidRerankerInputError(
                "Retrieval candidates must be a sequence."
            )

        validated: list[tuple[int, RetrievalCandidate]] = []
        seen_chunk_ids: set[str] = set()

        for rank, candidate in enumerate(candidates):
            if not isinstance(candidate, RetrievalCandidate):
                raise InvalidRerankerInputError(
                    f"Candidate at rank {rank} is not a "
                    "RetrievalCandidate instance."
                )

            if not candidate.text.strip():
                raise InvalidRerankerInputError(
                    f"Candidate {candidate.chunk_id} has empty text."
                )

            if candidate.chunk_id in seen_chunk_ids:
                raise InvalidRerankerInputError(
                    f"Duplicate chunk_id detected: "
                    f"{candidate.chunk_id}"
                )

            seen_chunk_ids.add(candidate.chunk_id)
            validated.append((rank, candidate))

        return validated

    # ------------------------------------------------------------------------
    # Model execution
    # ------------------------------------------------------------------------

    async def _score_candidates(
        self,
        query: str,
        candidates: Sequence[tuple[int, RetrievalCandidate]],
        metrics: Any,
    ) -> list[RerankedCandidate]:

        texts = [
            candidate.text
            for _, candidate in candidates
        ]

        started_at = time.perf_counter()

        raw_scores = await self._call_model(
            query=query,
            texts=texts,
        )

        metrics.scoring_time_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        scores = self._validate_scores(
            raw_scores,
            expected_count=len(texts),
        )

        metrics.scored_count = sum(
            score is not None
            for score in scores
        )

        reranked: list[RerankedCandidate] = []

        for (original_rank, candidate), score in zip(
            candidates,
            scores,
            strict=True,
        ):
            reranked.append(
                RerankedCandidate(
                    chunk_id=candidate.chunk_id,
                    source_id=candidate.source_id,
                    text=candidate.text,
                    original_score=float(candidate.score),
                    rerank_score=score,
                    was_reranked=score is not None,
                    original_rank=original_rank,
                    metadata=dict(candidate.metadata),
                )
            )

        valid = [
            candidate
            for candidate in reranked
            if candidate.rerank_score is not None
        ]

        invalid = [
            candidate
            for candidate in reranked
            if candidate.rerank_score is None
        ]

        valid.sort(
            key=lambda candidate: float(
                candidate.rerank_score  # type: ignore[arg-type]
            ),
            reverse=self._settings.higher_is_better,
        )

        return valid + invalid

    async def _call_model(
        self,
        *,
        query: str,
        texts: list[str],
    ) -> Any:

        try:
            return await asyncio.wait_for(
                self._invoke_model(
                    query=query,
                    texts=texts,
                ),
                timeout=self._settings.timeout_seconds,
            )

        except asyncio.TimeoutError as error:
            raise RerankerTimeoutError(
                "Reranker model execution timed out."
            ) from error

        except RerankerError:
            raise

        except Exception as error:
            raise RerankerFailureError(
                "Reranker model execution failed."
            ) from error

    async def _invoke_model(
        self,
        *,
        query: str,
        texts: list[str],
    ) -> Any:

        function = self._model.score_pairs

        underlying = getattr(
            function,
            "__func__",
            function,
        )

        if (
            inspect.iscoroutinefunction(function)
            or inspect.iscoroutinefunction(underlying)
        ):
            return await function(
                query,
                texts,
            )

        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            self._executor,
            function,
            query,
            texts,
        )

    # ------------------------------------------------------------------------
    # Score validation
    # ------------------------------------------------------------------------

    def _validate_scores(
        self,
        raw_scores: Any,
        *,
        expected_count: int,
    ) -> list[float | None]:

        if not isinstance(raw_scores, Sequence) or isinstance(
            raw_scores,
            (str, bytes),
        ):
            raise InvalidRerankerOutputError(
                "Reranker model must return a sequence of scores."
            )

        if len(raw_scores) != expected_count:
            raise InvalidRerankerOutputError(
                "Reranker score count does not match "
                "candidate count."
            )

        validated: list[float | None] = []

        for score in raw_scores:
            if score is None:
                validated.append(None)
                continue

            if isinstance(score, bool):
                validated.append(None)
                continue

            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                validated.append(None)
                continue

            if not math.isfinite(numeric_score):
                validated.append(None)
                continue

            validated.append(numeric_score)

        return validated

    # ------------------------------------------------------------------------
    # Filtering / truncation
    # ------------------------------------------------------------------------

    def _apply_threshold_and_top_n(
        self,
        candidates: Sequence[RerankedCandidate],
        *,
        top_n: int,
        metrics: Any,
    ) -> list[RerankedCandidate]:

        threshold = self._settings.score_threshold

        if threshold is None:
            return list(candidates[:top_n])

        kept: list[RerankedCandidate] = []

        for candidate in candidates:
            score = candidate.rerank_score

            if score is None:
                metrics.dropped_by_threshold += 1
                continue

            if self._settings.higher_is_better:
                passes = score >= threshold
            else:
                passes = score <= threshold

            if not passes:
                metrics.dropped_by_threshold += 1
                continue

            kept.append(candidate)

            if len(kept) >= top_n:
                break

        return kept

    # ------------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------------

    def _build_fallback_result(
        self,
        *,
        retrieval_result: RetrievalResult,
        top_n: int,
        trace_id: str,
        metrics: Any,
        error_code: str,
        started_at: float,
    ) -> RerankedResult:

        fallback_candidates: list[RerankedCandidate] = []

        for rank, candidate in enumerate(
            retrieval_result.candidates
        ):
            if not candidate.text.strip():
                continue

            fallback_candidates.append(
                RerankedCandidate(
                    chunk_id=candidate.chunk_id,
                    source_id=candidate.source_id,
                    text=candidate.text,
                    original_score=float(candidate.score),
                    rerank_score=None,
                    was_reranked=False,
                    original_rank=rank,
                    metadata=dict(candidate.metadata),
                )
            )

            if len(fallback_candidates) >= top_n:
                break

        metrics.final_count = len(fallback_candidates)
        metrics.used_fallback = True
        metrics.execution_time_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        return RerankedResult(
            query=retrieval_result.query.text,
            candidates=fallback_candidates,
            is_fallback=True,
            metrics=metrics,
            trace_id=trace_id,
            error_code=error_code,
        )

    # ------------------------------------------------------------------------
    # Empty result
    # ------------------------------------------------------------------------

    def _build_empty_result(
        self,
        *,
        retrieval_result: RetrievalResult,
        trace_id: str,
        metrics: Any,
        started_at: float,
    ) -> RerankedResult:

        metrics.final_count = 0
        metrics.execution_time_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        return RerankedResult(
            query=retrieval_result.query.text,
            candidates=[],
            is_fallback=False,
            metrics=metrics,
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------------
    # Configuration / lifecycle
    # ------------------------------------------------------------------------

    def _resolve_top_n(
        self,
        override: int | None,
    ) -> int:

        value = (
            self._settings.top_n
            if override is None
            else override
        )

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise InvalidRerankerConfigurationError(
                "top_n must be a positive integer."
            )

        if value > self._settings.max_top_n:
            raise InvalidRerankerConfigurationError(
                f"top_n cannot exceed "
                f"{self._settings.max_top_n}."
            )

        return value

    @staticmethod
    def _error_code(
        error: RerankerError,
    ) -> str:

        if isinstance(error, RerankerTimeoutError):
            return "reranker_timeout"

        if isinstance(error, InvalidRerankerOutputError):
            return "invalid_reranker_output"

        return "reranker_failure"

    def _ensure_open(self) -> None:
        if self._closed:
            raise RerankerError(
                "Reranker is closed."
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

    def __enter__(self) -> "Reranker":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    async def __aenter__(self) -> "Reranker":
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        await self.aclose()
