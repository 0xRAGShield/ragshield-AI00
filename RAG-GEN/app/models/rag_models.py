from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

# Shared Chat Contract

MessageRole = Literal["system", "user"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    role: MessageRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "message content must not be empty."
            )

        return value

# Retrieval

class RetrievalQuery(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    text: str = Field(
        min_length=1,
        max_length=100_000,
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Query text must not be empty."
            )

        return value


class RetrievalCandidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    score: float

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError(
                "Retrieval score must be finite."
            )

        return float(value)


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    trace_id: str = ""

    started_at: float = 0.0

    requested_top_k: int = 0

    returned_candidates: int = 0

    dropped_candidates: int = 0

    embedding_latency_ms: float = 0.0

    search_latency_ms: float = 0.0

    total_latency_ms: float = 0.0

    is_degraded: bool = False


class RetrievalResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query: RetrievalQuery

    candidates: list[RetrievalCandidate] = Field(
        default_factory=list
    )

    metrics: RetrievalMetrics

    is_degraded: bool = False

    dropped_candidates_count: int = 0

# Reranking

class RerankedCandidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    original_score: float

    rerank_score: float | None = None

    original_rank: int = Field(
        default=0,
        ge=0,
    )

    rank: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    was_reranked: bool = True

    @field_validator(
        "original_score",
        "rerank_score",
    )
    @classmethod
    def validate_scores(
        cls,
        value: float | None,
    ) -> float | None:

        if value is not None and not math.isfinite(
            float(value)
        ):
            raise ValueError(
                "Scores must be finite."
            )

        return (
            None
            if value is None
            else float(value)
        )


class RerankerMetrics(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    trace_id: str = ""

    input_count: int = 0

    scored_count: int = 0

    dropped_by_threshold: int = 0

    final_count: int = 0

    scoring_time_ms: float = 0.0

    execution_time_ms: float = 0.0

    used_fallback: bool = False


class RerankedResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query: str = Field(min_length=1)

    candidates: list[RerankedCandidate] = Field(
        default_factory=list
    )

    is_degraded: bool = False

    is_fallback: bool = False

    metrics: RerankerMetrics = Field(
        default_factory=RerankerMetrics
    )

    trace_id: str | None = None

    error_code: str | None = None

    dropped_candidates_count: int = 0

    was_reranked: bool = True

# Evidence / Context

class EvidenceIdentity(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: str = Field(min_length=1)

    source_id: str = Field(min_length=1)


class EvidenceProvenance(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source: str | None = None

    section: str | None = None

    page: int | None = Field(
        default=None,
        ge=1,
    )

    source_id: str = Field(min_length=1)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class EvidenceBlock(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    identity: EvidenceIdentity

    provenance: EvidenceProvenance

    rank: int = Field(
        default=0,
        ge=0,
    )

    rerank_score: float | None = None

    content: str = Field(min_length=1)

    content_kind: Literal[
        "untrusted_evidence"
    ] = "untrusted_evidence"

    token_count: int = Field(
        default=0,
        ge=0,
    )


class ContextBudgetStats(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    max_context_tokens: int = Field(
        ge=1
    )

    estimated_context_tokens: int = Field(
        default=0,
        ge=0,
    )

    remaining_tokens: int = Field(
        default=0,
        ge=0,
    )

    prompt_reserve_tokens: int = Field(
        default=0,
        ge=0,
    )

    query_reserve_tokens: int = Field(
        default=0,
        ge=0,
    )

    output_reserve_tokens: int = Field(
        default=0,
        ge=0,
    )


class ContextBuilderMetrics(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    input_candidate_count: int = 0

    selected_candidate_count: int = 0

    dropped_invalid: int = 0

    dropped_duplicate: int = 0

    dropped_token_budget: int = 0

    dropped_count_limit: int = 0

    dropped_evidence_count: int = 0

    estimated_context_tokens: int = 0

    context_token_budget: int = 0

    context_build_time_ms: float = 0.0


class AssembledContext(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query: str = Field(min_length=1)

    evidence: list[EvidenceBlock] = Field(
        default_factory=list
    )

    is_degraded: bool = False

    dropped_evidence_count: int = 0

    budget: ContextBudgetStats

    metrics: ContextBuilderMetrics

    trace_id: str | None = None

    tokenizer_encoding: str | None = None

# Prompt

class LLMPrompt(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    messages: list[ChatMessage] = Field(
        min_length=1
    )

    trace_id: str | None = None

    safety_mode: Literal["default", "strict"] = "default"

    input_token_count: int = 0

    allowed_input_tokens: int = 0

    reserved_output_tokens: int = 0

    output_reserve_tokens: int = 0

    max_prompt_tokens: int = 0

    evidence_count: int = 0

    is_empty_evidence: bool = False

    is_degraded: bool = False

    tokenizer_encoding: str | None = None