"""Context assembly component for a Medical RAG pipeline.

Pipeline:

    RerankedResult
        → validate
        → dedupe (document_id, chunk_id)
        → preserve reranker order
        → max_evidence_items
        → whole-chunk token budget
        → AssembledContext

This module assembles structured evidence only. It does not retrieve, rerank,
build prompts, call an LLM, or decide refusal. Ranking ownership stays with
the reranker; this file never re-sorts by score, document, or source.

Until teammate ``settings.py`` / ``rag_models.py`` land, typed exceptions,
``ContextSettings``, and Pydantic contracts live at the top of this file.
This file does not import the reranker implementation.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from contextvars import ContextVar
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Exceptions — typed; never swallowed into an empty context
# ---------------------------------------------------------------------------


class ContextBuilderError(Exception):
    """Base class for context-assembly failures."""

    error_code: str = "context_builder_error"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause


class InvalidContextConfigurationError(ContextBuilderError):
    error_code = "invalid_context_configuration"


class InvalidContextInputError(ContextBuilderError):
    error_code = "invalid_context_input"


class TokenizerFailureError(ContextBuilderError):
    """Tokenizer / infrastructure failure. Aborts the entire build()."""

    error_code = "tokenizer_failure"


# ---------------------------------------------------------------------------
# Dummy settings — stand-in for centralized production config
# ---------------------------------------------------------------------------

_DEFAULT_METADATA_ALLOWLIST: tuple[str, ...] = (
    "title",
    "section",
    "source",
    "page",
    "publication",
    "document_type",
)


class ContextSettings(BaseModel):
    """Context knobs. Packing values are not hardcoded in the assembler loop.

    ``max_context_tokens`` is the evidence allocation only — not the full LLM
    window. Reserve fields are informational pass-through; they are never
    subtracted here.
    """

    model_config = ConfigDict(extra="forbid")

    max_context_tokens: int = 2048
    max_evidence_items: int = 10
    prompt_reserve_tokens: int = 0
    query_reserve_tokens: int = 0
    output_reserve_tokens: int = 0
    estimated_chars_per_token: float = 4.0
    tokenizer_encoding: str = "cl100k_base"
    require_exact_tokenizer: bool = True
    metadata_allowlist: tuple[str, ...] = _DEFAULT_METADATA_ALLOWLIST

    @field_validator(
        "max_context_tokens",
        "max_evidence_items",
        "prompt_reserve_tokens",
        "query_reserve_tokens",
        "output_reserve_tokens",
    )
    @classmethod
    def _non_negative_int(cls, value: int, info: Any) -> int:
        name = getattr(info, "field_name", "")
        if name in {"max_context_tokens", "max_evidence_items"}:
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        elif value < 0:
            raise ValueError(f"{name} must be >= 0")
        return value

    @field_validator("estimated_chars_per_token")
    @classmethod
    def _positive_ratio(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("estimated_chars_per_token must be a finite value > 0")
        return value

    @field_validator("tokenizer_encoding")
    @classmethod
    def _encoding_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tokenizer_encoding must be non-empty")
        return value.strip()

    @field_validator("metadata_allowlist")
    @classmethod
    def _allowlist_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item and str(item).strip())
        if not cleaned:
            raise ValueError("metadata_allowlist must contain at least one field")
        return cleaned


def get_context_settings() -> ContextSettings:
    """Fresh dummy settings. The builder copies config at construction."""

    return ContextSettings()


# ---------------------------------------------------------------------------
# Input stubs — duck-type compatible with reranker output; not a parallel RAG model
# ---------------------------------------------------------------------------


class Query(BaseModel):
    """Processed query stub. Consumed as-is; never re-normalized here."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=False)

    text: str = ""
    request_id: str | None = None

    def __repr__(self) -> str:
        return f"Query(length={len(self.text)}, request_id={self.request_id!r})"

    def __str__(self) -> str:
        return self.__repr__()


class RerankedCandidate(BaseModel):
    """Input candidate stub. Extra fields from the reranker are ignored."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    document_id: str
    text: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_id: str | None = None
    section: str | None = None
    page: str | int | None = None
    original_rank: int | None = None
    rerank_score: float | None = None
    loe: str | None = None
    ontology_codes: list[str] = Field(default_factory=list)
    point_id: str | None = None
    was_reranked: bool | None = None


class RerankedResult(BaseModel):
    """Typed reranker output stub. Primary input contract; not a raw dict."""

    model_config = ConfigDict(extra="allow")

    query: Query
    candidates: list[RerankedCandidate] = Field(default_factory=list)
    is_fallback: bool = False
    trace_id: str | None = None
    error_code: str | None = None


# ---------------------------------------------------------------------------
# Output contracts — structured evidence, never a final prompt string
# ---------------------------------------------------------------------------

ContentKind = Literal["untrusted_evidence"]


class EvidenceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str


class EvidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    section: str | None = None
    page: str | int | None = None
    source_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBlock(BaseModel):
    """One provenance-preserving evidence unit for prompt_builder.py."""

    model_config = ConfigDict(extra="forbid")

    identity: EvidenceIdentity
    provenance: EvidenceProvenance
    rank: int
    rerank_score: float | None = None
    content: str
    content_kind: ContentKind = "untrusted_evidence"
    token_count: int


class ContextBudgetStats(BaseModel):
    """Evidence packing budget. Reserve fields are pass-through documentation."""

    model_config = ConfigDict(extra="forbid")

    max_context_tokens: int
    estimated_context_tokens: int
    remaining_tokens: int
    prompt_reserve_tokens: int
    query_reserve_tokens: int
    output_reserve_tokens: int


class ContextBuilderMetrics(BaseModel):
    """PII-safe counters. Never includes query or chunk text."""

    model_config = ConfigDict(extra="forbid")

    context_build_time: float = 0.0
    context_build_time_ms: float = 0.0
    input_candidate_count: int = 0
    selected_candidate_count: int = 0
    dropped_invalid: int = 0
    dropped_duplicate: int = 0
    dropped_token_budget: int = 0
    dropped_count_limit: int = 0
    dropped_evidence_count: int = 0
    estimated_context_tokens: int = 0
    context_token_budget: int = 0


class AssembledContext(BaseModel):
    """Typed context output. Empty evidence is a valid non-refusal result."""

    model_config = ConfigDict(extra="forbid")

    query: Query
    evidence: list[EvidenceBlock] = Field(default_factory=list)
    is_degraded: bool = False
    dropped_evidence_count: int = 0
    budget: ContextBudgetStats
    metrics: ContextBuilderMetrics
    trace_id: str
    tokenizer_encoding: str | None = None


# ---------------------------------------------------------------------------
# Tokenizer protocol — identity-checked; no vendor model loaded here
# ---------------------------------------------------------------------------


@runtime_checkable
class ITokenCounter(Protocol):
    """Counts tokens with a declared encoding identity.

    ``encoding_name`` is the generation-tokenizer identity (e.g. ``cl100k_base``).
    It is ``None`` only on the explicit char-ratio fallback, which must never
    pretend to be a real encoding.
    """

    encoding_name: str | None

    def count(self, text: str) -> int: ...


class CharRatioFallbackTokenCounter:
    """Approximate char/token fallback. Test / absent-tokenizer use only.

    A single Latin-centric ratio is systematically biased for Arabic/Egyptian
    medical text (BPE typically spends more tokens per character). This counter
    is not trusted to tune production evidence budgets. Production must inject
    the real generation tokenizer with a matching ``encoding_name``.

    ``encoding_name`` is always ``None`` so this adapter cannot impersonate
    ``cl100k_base`` or any other real encoding.
    """

    encoding_name: str | None = None

    def __init__(self, chars_per_token: float) -> None:
        if not math.isfinite(chars_per_token) or chars_per_token <= 0:
            raise InvalidContextConfigurationError(
                "estimated_chars_per_token must be a finite value > 0"
            )
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TokenizerFailureError("fallback tokenizer received non-string text")
        return max(1, math.ceil(len(text) / self._chars_per_token))


# ---------------------------------------------------------------------------
# Observability — request-scoped, never logs PHI / clinical text
# ---------------------------------------------------------------------------


logger = logging.getLogger("rag.context_builder")
_trace_id_var: ContextVar[str] = ContextVar("rag_context_builder_trace_id", default="")
_QUERY_HASH_BYTES = 12


def current_trace_id() -> str:
    return _trace_id_var.get()


def query_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_QUERY_HASH_BYTES]


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "trace_id": current_trace_id(), **fields}
    logger.info("%s", payload)


# ---------------------------------------------------------------------------
# Validation / coercion helpers
# ---------------------------------------------------------------------------


def _validate_settings(settings: ContextSettings) -> ContextSettings:
    try:
        return ContextSettings.model_validate(settings.model_dump())
    except Exception as exc:
        raise InvalidContextConfigurationError(str(exc), cause=exc) from exc


def _coerce_query(raw: Any) -> Query:
    if isinstance(raw, Query):
        return raw
    if isinstance(raw, Mapping):
        return Query.model_validate(raw)
    if raw is None:
        return Query(text="")
    text = getattr(raw, "text", "")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise InvalidContextInputError("query.text must be a string")
    request_id = getattr(raw, "request_id", None)
    return Query(text=text, request_id=request_id if isinstance(request_id, str) else None)


def _coerce_candidate(raw: Any) -> RerankedCandidate | None:
    """Return a typed candidate, or None if the object cannot even be parsed."""

    if isinstance(raw, RerankedCandidate):
        return raw
    if isinstance(raw, Mapping):
        try:
            return RerankedCandidate.model_validate(raw)
        except Exception:
            return None
    try:
        return RerankedCandidate.model_validate(
            {
                "chunk_id": getattr(raw, "chunk_id", None),
                "document_id": getattr(raw, "document_id", None),
                "text": getattr(raw, "text", None),
                "source": getattr(raw, "source", None),
                "score": getattr(raw, "score", None),
                "metadata": getattr(raw, "metadata", None) or {},
                "source_id": getattr(raw, "source_id", None),
                "section": getattr(raw, "section", None),
                "page": getattr(raw, "page", None),
                "original_rank": getattr(raw, "original_rank", None),
                "rerank_score": getattr(raw, "rerank_score", None),
                "loe": getattr(raw, "loe", None),
                "ontology_codes": getattr(raw, "ontology_codes", None) or [],
                "point_id": getattr(raw, "point_id", None),
                "was_reranked": getattr(raw, "was_reranked", None),
            }
        )
    except Exception:
        return None


def _coerce_reranked_result(result: RerankedResult | Mapping[str, Any] | Any) -> RerankedResult:
    if isinstance(result, RerankedResult):
        return result
    if isinstance(result, Mapping):
        try:
            return RerankedResult.model_validate(result)
        except Exception as exc:
            raise InvalidContextInputError(
                "input does not match the RerankedResult contract",
                cause=exc,
            ) from exc
    query = _coerce_query(getattr(result, "query", None))
    raw_candidates = getattr(result, "candidates", None)
    if raw_candidates is None:
        raise InvalidContextInputError("reranked result is missing candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise InvalidContextInputError("candidates must be a sequence")
    candidates: list[RerankedCandidate] = []
    for item in raw_candidates:
        coerced = _coerce_candidate(item)
        if coerced is not None:
            candidates.append(coerced)
        else:
            candidates.append(
                RerankedCandidate(
                    chunk_id="",
                    document_id="",
                    text="",
                    source="",
                )
            )
    return RerankedResult(
        query=query,
        candidates=candidates,
        is_fallback=bool(getattr(result, "is_fallback", False)),
        trace_id=getattr(result, "trace_id", None),
        error_code=getattr(result, "error_code", None),
    )


def _non_empty_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _rerank_score_ok(score: Any) -> bool:
    if score is None:
        return True
    try:
        return math.isfinite(float(score))
    except (TypeError, ValueError):
        return False


def _validate_candidate(candidate: RerankedCandidate) -> str | None:
    """Return a drop reason, or None if the candidate is usable evidence."""

    if _non_empty_id(candidate.chunk_id) is None:
        return "missing_chunk_id"
    if _non_empty_id(candidate.document_id) is None:
        return "missing_document_id"
    if not isinstance(candidate.text, str) or not candidate.text.strip():
        return "empty_text"
    if _non_empty_id(candidate.source) is None:
        return "missing_source"
    if not _rerank_score_ok(candidate.rerank_score):
        return "invalid_rerank_score"
    if candidate.metadata is None or not isinstance(candidate.metadata, Mapping):
        return "invalid_metadata"
    return None


def _dedupe_key(candidate: RerankedCandidate) -> tuple[str, str]:
    chunk_id = _non_empty_id(candidate.chunk_id) or ""
    document_id = _non_empty_id(candidate.document_id) or ""
    return (document_id, chunk_id)


def _allowlisted_metadata(
    metadata: Mapping[str, Any],
    allowlist: Sequence[str],
) -> dict[str, Any]:
    allowed = set(allowlist)
    return {key: value for key, value in metadata.items() if key in allowed}


def _provenance_section(candidate: RerankedCandidate) -> str | None:
    if isinstance(candidate.section, str) and candidate.section.strip():
        return candidate.section
    meta = candidate.metadata or {}
    section = meta.get("section") if isinstance(meta, Mapping) else None
    if isinstance(section, str) and section.strip():
        return section
    return None


def _provenance_page(candidate: RerankedCandidate) -> str | int | None:
    if candidate.page is not None and candidate.page != "":
        return candidate.page
    meta = candidate.metadata or {}
    page = meta.get("page") if isinstance(meta, Mapping) else None
    if page is None or page == "":
        return None
    return page


def _to_evidence_block(
    candidate: RerankedCandidate,
    *,
    rank: int,
    token_count: int,
    allowlist: Sequence[str],
) -> EvidenceBlock:
    metadata = _allowlisted_metadata(candidate.metadata, allowlist)
    return EvidenceBlock(
        identity=EvidenceIdentity(
            chunk_id=candidate.chunk_id.strip(),
            document_id=candidate.document_id.strip(),
        ),
        provenance=EvidenceProvenance(
            source=candidate.source.strip(),
            section=_provenance_section(candidate),
            page=_provenance_page(candidate),
            source_id=_non_empty_id(candidate.source_id),
            metadata=metadata,
        ),
        rank=rank,
        rerank_score=None if candidate.rerank_score is None else float(candidate.rerank_score),
        content=candidate.text,
        content_kind="untrusted_evidence",
        token_count=token_count,
    )


def _resolve_counter(
    settings: ContextSettings,
    token_counter: ITokenCounter | None,
) -> ITokenCounter:
    if token_counter is None:
        if settings.require_exact_tokenizer:
            raise InvalidContextConfigurationError(
                "require_exact_tokenizer=True but no ITokenCounter was injected; "
                "char-ratio fallback is not allowed on the production path"
            )
        return CharRatioFallbackTokenCounter(settings.estimated_chars_per_token)

    encoding_name = getattr(token_counter, "encoding_name", None)
    if settings.require_exact_tokenizer:
        if not isinstance(encoding_name, str) or not encoding_name.strip():
            raise InvalidContextConfigurationError(
                "injected token counter is missing encoding_name; "
                "char-ratio fallback cannot satisfy require_exact_tokenizer=True"
            )
        if encoding_name.strip() != settings.tokenizer_encoding:
            raise InvalidContextConfigurationError(
                f"token counter encoding {encoding_name!r} != "
                f"settings.tokenizer_encoding {settings.tokenizer_encoding!r}"
            )
        return token_counter

    if isinstance(encoding_name, str) and encoding_name.strip():
        if encoding_name.strip() != settings.tokenizer_encoding:
            raise InvalidContextConfigurationError(
                f"token counter encoding {encoding_name!r} != "
                f"settings.tokenizer_encoding {settings.tokenizer_encoding!r}"
            )
    return token_counter


def _count_tokens(counter: ITokenCounter, text: str) -> int:
    try:
        raw = counter.count(text)
    except TokenizerFailureError:
        raise
    except Exception as exc:
        raise TokenizerFailureError("token counter failed", cause=exc) from exc
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TokenizerFailureError(
            f"token counter returned {type(raw).__name__}, expected int"
        )
    if not math.isfinite(float(raw)):
        raise TokenizerFailureError("token counter returned a non-finite count")
    count = int(raw)
    if count < 1:
        raise TokenizerFailureError("token counter returned a non-positive count")
    return count


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Sync context assembler. Constructor-injected; no hidden global state."""

    def __init__(
        self,
        settings: ContextSettings | None = None,
        token_counter: ITokenCounter | None = None,
    ) -> None:
        self._settings = _validate_settings(settings or get_context_settings())
        self._counter = _resolve_counter(self._settings, token_counter)

    @property
    def settings(self) -> ContextSettings:
        return self._settings

    @property
    def token_counter(self) -> ITokenCounter:
        return self._counter

    def build(self, result: RerankedResult | Mapping[str, Any] | Any) -> AssembledContext:
        """Assemble structured evidence from a typed rerank result.

        Empty rerank candidates yield a valid empty context (not a refusal).
        Tokenizer infrastructure failure aborts the entire call.
        """

        total_t0 = time.perf_counter()
        parsed = _coerce_reranked_result(result)
        trace_id = parsed.trace_id or uuid.uuid4().hex
        token = _trace_id_var.set(trace_id)
        metrics = ContextBuilderMetrics(
            input_candidate_count=len(parsed.candidates),
            context_token_budget=self._settings.max_context_tokens,
        )
        try:
            _log_event(
                "context_build_start",
                query_length=len(parsed.query.text),
                query_hash=query_hash(parsed.query.text),
                input_candidate_count=metrics.input_candidate_count,
                context_token_budget=self._settings.max_context_tokens,
                max_evidence_items=self._settings.max_evidence_items,
            )
            evidence = self._assemble(parsed.candidates, metrics)
            used = sum(item.token_count for item in evidence)
            remaining = self._settings.max_context_tokens - used
            metrics.selected_candidate_count = len(evidence)
            metrics.dropped_evidence_count = (
                metrics.dropped_invalid
                + metrics.dropped_duplicate
                + metrics.dropped_token_budget
                + metrics.dropped_count_limit
            )
            metrics.estimated_context_tokens = used
            elapsed = time.perf_counter() - total_t0
            metrics.context_build_time = elapsed
            metrics.context_build_time_ms = elapsed * 1000.0
            is_degraded = metrics.dropped_invalid > 0
            assembled = AssembledContext(
                query=parsed.query,
                evidence=evidence,
                is_degraded=is_degraded,
                dropped_evidence_count=metrics.dropped_evidence_count,
                budget=ContextBudgetStats(
                    max_context_tokens=self._settings.max_context_tokens,
                    estimated_context_tokens=used,
                    remaining_tokens=remaining,
                    prompt_reserve_tokens=self._settings.prompt_reserve_tokens,
                    query_reserve_tokens=self._settings.query_reserve_tokens,
                    output_reserve_tokens=self._settings.output_reserve_tokens,
                ),
                metrics=metrics,
                trace_id=trace_id,
                tokenizer_encoding=getattr(self._counter, "encoding_name", None),
            )
            _log_event(
                "context_build_done",
                query_length=len(parsed.query.text),
                query_hash=query_hash(parsed.query.text),
                input_candidate_count=metrics.input_candidate_count,
                selected_candidate_count=metrics.selected_candidate_count,
                dropped_invalid=metrics.dropped_invalid,
                dropped_duplicate=metrics.dropped_duplicate,
                dropped_token_budget=metrics.dropped_token_budget,
                dropped_count_limit=metrics.dropped_count_limit,
                dropped_evidence_count=metrics.dropped_evidence_count,
                estimated_context_tokens=metrics.estimated_context_tokens,
                context_token_budget=metrics.context_token_budget,
                context_build_time_ms=round(metrics.context_build_time_ms, 3),
                is_degraded=is_degraded,
            )
            return assembled
        finally:
            _trace_id_var.reset(token)

    def _assemble(
        self,
        candidates: Sequence[RerankedCandidate],
        metrics: ContextBuilderMetrics,
    ) -> list[EvidenceBlock]:
        selected: list[EvidenceBlock] = []
        seen: set[tuple[str, str]] = set()
        remaining_budget = self._settings.max_context_tokens
        max_items = self._settings.max_evidence_items
        allowlist = self._settings.metadata_allowlist

        for index, candidate in enumerate(candidates):
            if len(selected) >= max_items:
                metrics.dropped_count_limit += len(candidates) - index
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

            token_count = _count_tokens(self._counter, candidate.text)
            if token_count > remaining_budget:
                metrics.dropped_token_budget += 1
                continue

            rank = index + 1
            selected.append(
                _to_evidence_block(
                    candidate,
                    rank=rank,
                    token_count=token_count,
                    allowlist=allowlist,
                )
            )
            remaining_budget -= token_count

        return selected


def build_context_builder(
    settings: ContextSettings | None = None,
    token_counter: ITokenCounter | None = None,
) -> ContextBuilder:
    """Factory used by the orchestration layer above the reranker."""

    return ContextBuilder(settings, token_counter)
