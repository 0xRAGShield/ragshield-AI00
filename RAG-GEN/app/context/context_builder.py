"""
Production context assembly for RAGShield.

Pipeline:

    RerankedResult
        ↓
    Validate candidates
        ↓
    Deduplicate (source_id, chunk_id)
        ↓
    Preserve reranker order
        ↓
    Apply evidence-count limit
        ↓
    Apply whole-chunk token budget
        ↓
    AssembledContext
        ↓
    Prompt Builder

This component does NOT own:
- retrieval
- reranking
- embeddings
- vector-store access
- chunking
- prompt construction
- LLM/VLM execution
- refusal decisions
- medical reasoning
- evidence rewriting or summarization
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from contextvars import ContextVar
from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict

from app.core.settings import ContextSettings
from app.models.rag_models import (
    AssembledContext,
    ContextBudgetStats,
    ContextBuilderMetrics,
    EvidenceBlock,
    EvidenceIdentity,
    EvidenceProvenance,
)
from app.retrieval.reranker import (
    RerankedCandidate,
    RerankedResult,
)


logger = logging.getLogger("ragshield.context_builder")

_trace_id_var: ContextVar[str] = ContextVar(
    "ragshield_context_trace_id",
    default="",
)

_QUERY_HASH_BYTES = 12


# ============================================================================
# Exceptions
# ============================================================================

class ContextBuilderError(RuntimeError):
    """Base exception for context-building failures."""

    error_code: str = "context_builder_error"


class InvalidContextConfigurationError(ContextBuilderError):
    """Raised when context-builder configuration is invalid."""

    error_code = "invalid_context_configuration"


class InvalidContextInputError(ContextBuilderError):
    """Raised when reranker output violates the context contract."""

    error_code = "invalid_context_input"


class TokenizerFailureError(ContextBuilderError):
    """Raised when token counting infrastructure fails."""

    error_code = "tokenizer_failure"


# ============================================================================
# Token Counter Contract
# ============================================================================

@runtime_checkable
class ITokenCounter(Protocol):
    """
    Token counting dependency.

    encoding_name identifies the tokenizer used for the count.
    """

    encoding_name: str | None

    def count(self, text: str) -> int:
        ...


class CharRatioFallbackTokenCounter:
    """
    Approximate token counter for tests/non-production operation.

    It intentionally exposes encoding_name=None so it cannot pretend
    to be an exact generation tokenizer.
    """

    encoding_name: str | None = None

    def __init__(self, chars_per_token: float) -> None:
        if (
            not math.isfinite(float(chars_per_token))
            or chars_per_token <= 0
        ):
            raise InvalidContextConfigurationError(
                "chars_per_token must be a finite positive number."
            )

        self._chars_per_token = float(chars_per_token)

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TokenizerFailureError(
                "token counter received non-string text."
            )

        return max(
            1,
            math.ceil(
                len(text) / self._chars_per_token
            ),
        )


# ============================================================================
# Observability
# ============================================================================

def current_trace_id() -> str:
    return _trace_id_var.get()


def query_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:_QUERY_HASH_BYTES]


def _log_event(
    event: str,
    **fields: Any,
) -> None:
    """
    Structured PII-safe logging.

    Never logs query text, candidate text, or clinical evidence.
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
# Input Validation
# ============================================================================

def _validate_reranked_result(
    result: RerankedResult,
) -> None:
    if not isinstance(result, RerankedResult):
        raise InvalidContextInputError(
            "result must be a RerankedResult instance."
        )

    if not isinstance(result.query, str):
        raise InvalidContextInputError(
            "RerankedResult.query must be a string."
        )

    if not result.query.strip():
        raise InvalidContextInputError(
            "RerankedResult.query cannot be empty."
        )

    if not isinstance(result.candidates, Sequence):
        raise InvalidContextInputError(
            "RerankedResult.candidates must be a sequence."
        )


def _validate_candidate(
    candidate: RerankedCandidate,
) -> str | None:
    """Return a drop reason or None when valid."""

    if (
        not isinstance(candidate.chunk_id, str)
        or not candidate.chunk_id.strip()
    ):
        return "missing_chunk_id"

    if (
        not isinstance(candidate.source_id, str)
        or not candidate.source_id.strip()
    ):
        return "missing_source_id"

    if (
        not isinstance(candidate.text, str)
        or not candidate.text.strip()
    ):
        return "empty_text"

    if (
        isinstance(candidate.rerank_score, bool)
        or (
            candidate.rerank_score is not None
            and (
                not isinstance(
                    candidate.rerank_score,
                    (int, float),
                )
                or not math.isfinite(
                    float(candidate.rerank_score)
                )
            )
        )
    ):
        return "invalid_rerank_score"

    if not isinstance(candidate.metadata, dict):
        return "invalid_metadata"

    return None


def _dedupe_key(
    candidate: RerankedCandidate,
) -> tuple[str, str]:
    """
    Evidence identity is source + chunk.

    We do not dedupe by score, text, or title.
    """

    return (
        candidate.source_id.strip(),
        candidate.chunk_id.strip(),
    )


# ============================================================================
# Provenance
# ============================================================================

def _resolve_section(
    candidate: RerankedCandidate,
) -> str | None:

    value = candidate.metadata.get("section")

    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def _resolve_page(
    candidate: RerankedCandidate,
) -> int | None:

    value = candidate.metadata.get("page")

    if value is None or value == "":
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value >= 1 else None

    if isinstance(value, str):
        value = value.strip()

        if value.isdigit():
            page = int(value)
            return page if page >= 1 else None

    return None


def _allowlisted_metadata(
    metadata: dict[str, Any],
    allowlist: Sequence[str],
) -> dict[str, Any]:

    allowed = set(allowlist)

    return {
        key: value
        for key, value in metadata.items()
        if key in allowed
    }


def _build_evidence_block(
    candidate: RerankedCandidate,
    *,
    rank: int,
    token_count: int,
    metadata_allowlist: Sequence[str],
) -> EvidenceBlock:

    source_id = candidate.source_id.strip()

    metadata = _allowlisted_metadata(
        candidate.metadata,
        metadata_allowlist,
    )

    return EvidenceBlock(
        identity=EvidenceIdentity(
            chunk_id=candidate.chunk_id.strip(),
            source_id=source_id,
        ),
        provenance=EvidenceProvenance(
            source=source_id,
            section=_resolve_section(candidate),
            page=_resolve_page(candidate),
            source_id=source_id,
            metadata=metadata,
        ),
        rank=rank,
        rerank_score=(
            None
            if candidate.rerank_score is None
            else float(candidate.rerank_score)
        ),
        content=candidate.text,
        content_kind="untrusted_evidence",
        token_count=token_count,
    )


# ============================================================================
# Token Counter
# ============================================================================

def _resolve_token_counter(
    settings: ContextSettings,
    token_counter: ITokenCounter | None,
) -> ITokenCounter:

    if token_counter is None:

        if settings.require_exact_tokenizer:
            raise InvalidContextConfigurationError(
                "Production context building requires an injected "
                "token counter with a matching tokenizer encoding."
            )

        return CharRatioFallbackTokenCounter(
            settings.estimated_chars_per_token
        )

    encoding_name = getattr(
        token_counter,
        "encoding_name",
        None,
    )

    if settings.require_exact_tokenizer:

        if (
            not isinstance(encoding_name, str)
            or not encoding_name.strip()
        ):
            raise InvalidContextConfigurationError(
                "Injected token counter does not expose encoding_name."
            )

        if (
            encoding_name.strip()
            != settings.tokenizer_encoding.strip()
        ):
            raise InvalidContextConfigurationError(
                "Token counter encoding does not match "
                "ContextSettings.tokenizer_encoding."
            )

    return token_counter


def _count_tokens(
    counter: ITokenCounter,
    text: str,
) -> int:

    try:
        value = counter.count(text)

    except TokenizerFailureError:
        raise

    except Exception as error:
        raise TokenizerFailureError(
            "Token counter failed."
        ) from error

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TokenizerFailureError(
            "Token counter returned an invalid value."
        )

    if not math.isfinite(float(value)):
        raise TokenizerFailureError(
            "Token counter returned a non-finite value."
        )

    count = int(value)

    if count <= 0:
        raise TokenizerFailureError(
            "Token counter returned a non-positive token count."
        )

    return count


# ============================================================================
# Context Builder
# ============================================================================

class ContextBuilder:
    """
    Production context assembler.

    Ranking ownership remains entirely with the Reranker.
    """

    def __init__(
        self,
        settings: ContextSettings,
        token_counter: ITokenCounter | None = None,
    ) -> None:

        if not isinstance(settings, ContextSettings):
            raise InvalidContextConfigurationError(
                "settings must be a ContextSettings instance."
            )

        self._settings = settings

        self._token_counter = _resolve_token_counter(
            settings,
            token_counter,
        )

        self._closed = False

    @property
    def settings(self) -> ContextSettings:
        return self._settings

    @property
    def token_counter(self) -> ITokenCounter:
        return self._token_counter

    def build(
        self,
        result: RerankedResult,
    ) -> AssembledContext:
        """
        Assemble structured evidence.

        Does not:
        - rerank
        - summarize
        - rewrite
        - make refusal decisions
        """

        self._ensure_open()

        _validate_reranked_result(result)

        started_at = time.perf_counter()

        trace_id = (
            result.trace_id
            if result.trace_id
            else uuid.uuid4().hex
        )

        trace_token = _trace_id_var.set(trace_id)

        metrics = ContextBuilderMetrics(
            input_candidate_count=len(result.candidates),
            context_token_budget=(
                self._settings.max_context_tokens
            ),
        )

        try:

            _log_event(
                "context_build_started",
                query_hash=query_hash(result.query),
                input_candidate_count=(
                    metrics.input_candidate_count
                ),
                context_token_budget=(
                    self._settings.max_context_tokens
                ),
                max_evidence_items=(
                    self._settings.max_evidence_items
                ),
            )

            evidence = self._assemble(
                result.candidates,
                metrics,
            )

            used_tokens = sum(
                block.token_count
                for block in evidence
            )

            remaining_tokens = (
                self._settings.max_context_tokens
                - used_tokens
            )

            metrics.selected_candidate_count = len(evidence)

            metrics.dropped_evidence_count = (
                metrics.dropped_invalid
                + metrics.dropped_duplicate
                + metrics.dropped_token_budget
                + metrics.dropped_count_limit
            )

            metrics.estimated_context_tokens = used_tokens

            metrics.context_build_time_ms = (
                time.perf_counter() - started_at
            ) * 1000.0

            is_degraded = (
                result.is_degraded
                or metrics.dropped_invalid > 0
                or metrics.dropped_duplicate > 0
                or metrics.dropped_token_budget > 0
                or metrics.dropped_count_limit > 0
            )

            assembled = AssembledContext(
                query=result.query,
                evidence=evidence,
                is_degraded=is_degraded,
                dropped_evidence_count=(
                    metrics.dropped_evidence_count
                ),
                budget=ContextBudgetStats(
                    max_context_tokens=(
                        self._settings.max_context_tokens
                    ),
                    estimated_context_tokens=used_tokens,
                    remaining_tokens=remaining_tokens,
                    prompt_reserve_tokens=(
                        self._settings.prompt_reserve_tokens
                    ),
                    query_reserve_tokens=(
                        self._settings.query_reserve_tokens
                    ),
                    output_reserve_tokens=(
                        self._settings.output_reserve_tokens
                    ),
                ),
                metrics=metrics,
                trace_id=trace_id,
                tokenizer_encoding=(
                    getattr(
                        self._token_counter,
                        "encoding_name",
                        None,
                    )
                ),
            )

            _log_event(
                "context_build_completed",
                query_hash=query_hash(result.query),
                input_candidate_count=(
                    metrics.input_candidate_count
                ),
                selected_candidate_count=(
                    metrics.selected_candidate_count
                ),
                dropped_invalid=(
                    metrics.dropped_invalid
                ),
                dropped_duplicate=(
                    metrics.dropped_duplicate
                ),
                dropped_token_budget=(
                    metrics.dropped_token_budget
                ),
                dropped_count_limit=(
                    metrics.dropped_count_limit
                ),
                dropped_evidence_count=(
                    metrics.dropped_evidence_count
                ),
                estimated_context_tokens=(
                    metrics.estimated_context_tokens
                ),
                context_build_time_ms=round(
                    metrics.context_build_time_ms,
                    3,
                ),
                is_degraded=is_degraded,
            )

            return assembled

        finally:
            _trace_id_var.reset(trace_token)

    def _assemble(
        self,
        candidates: Sequence[RerankedCandidate],
        metrics: ContextBuilderMetrics,
    ) -> list[EvidenceBlock]:

        selected: list[EvidenceBlock] = []
        seen: set[tuple[str, str]] = set()

        remaining_budget = (
            self._settings.max_context_tokens
        )

        max_items = (
            self._settings.max_evidence_items
        )

        for index, candidate in enumerate(candidates):

            # Preserve reranker order.
            if len(selected) >= max_items:

                metrics.dropped_count_limit += (
                    len(candidates) - index
                )

                break

            reason = _validate_candidate(candidate)

            if reason is not None:

                metrics.dropped_invalid += 1

                logger.warning(
                    "%s",
                    {
                        "event": "invalid_candidate_dropped",
                        "trace_id": current_trace_id(),
                        "reason": reason,
                    },
                )

                continue

            key = _dedupe_key(candidate)

            if key in seen:

                metrics.dropped_duplicate += 1
                continue

            seen.add(key)

            token_count = _count_tokens(
                self._token_counter,
                candidate.text,
            )

            if token_count > remaining_budget:

                metrics.dropped_token_budget += 1
                continue

            evidence_rank = len(selected) + 1

            selected.append(
                _build_evidence_block(
                    candidate,
                    rank=evidence_rank,
                    token_count=token_count,
                    metadata_allowlist=(
                        self._settings.metadata_allowlist
                    ),
                )
            )

            remaining_budget -= token_count

        return selected

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContextBuilderError(
                "ContextBuilder is closed."
            )

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "ContextBuilder":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()
