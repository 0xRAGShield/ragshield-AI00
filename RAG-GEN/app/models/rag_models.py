"""
Canonical runtime contracts for the RAG pipeline.

This module contains shared data models exchanged between RAG stages.

Responsibilities
----------------
- Define stable, typed contracts between pipeline components.
- Represent retrieval and reranking outputs.
- Represent assembled evidence context.
- Represent the final LLM prompt contract.

Non-responsibilities
--------------------
- Retrieval logic
- Reranking logic
- Context assembly
- Prompt construction
- LLM execution
- Vector database access
- Business logic
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# Retrieval
# ============================================================

class RetrievalQuery(BaseModel):
    """Canonical query entering the retrieval stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=100_000)
    top_k: int | None = Field(default=None, ge=1, le=100)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Query text must not be empty.")

        return value


class RetrievalCandidate(BaseModel):
    """Single candidate returned by the retrieval layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    score: float

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not (-float("inf") < value < float("inf")):
            raise ValueError("Retrieval score must be finite.")

        return value


class RetrievalMetrics(BaseModel):
    """Operational metrics produced by retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_top_k: int
    returned_candidates: int
    dropped_candidates: int = Field(default=0, ge=0)


class RetrievalResult(BaseModel):
    """Canonical result of the retrieval stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: RetrievalQuery

    candidates: list[RetrievalCandidate] = Field(
        default_factory=list
    )

    metrics: RetrievalMetrics

    is_degraded: bool = False
    dropped_candidates_count: int = Field(
        default=0,
        ge=0,
    )


# ============================================================
# Reranking
# ============================================================

class RerankedCandidate(BaseModel):
    """Candidate after reranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    retrieval_score: float
    rerank_score: float | None = None

    original_rank: int = Field(ge=0)
    rank: int = Field(ge=0)

    metadata: dict[str, Any] = Field(default_factory=dict)

    was_reranked: bool = True

    @field_validator("retrieval_score", "rerank_score")
    @classmethod
    def validate_scores(
        cls,
        value: float | None,
    ) -> float | None:
        if value is not None and not (
            -float("inf") < value < float("inf")
        ):
            raise ValueError("Scores must be finite.")

        return value


class RerankedResult(BaseModel):
    """Canonical result of the reranking stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)

    candidates: list[RerankedCandidate] = Field(
        default_factory=list
    )

    is_degraded: bool = False

    dropped_candidates_count: int = Field(
        default=0,
        ge=0,
    )

    was_reranked: bool = True

    trace_id: str | None = None


# ============================================================
# Evidence / Context
# ============================================================

class EvidenceIdentity(BaseModel):
    """Stable identity of a retrieved evidence item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class EvidenceProvenance(BaseModel):
    """Traceable provenance metadata for evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str | None = None
    section: str | None = None
    page: int | None = Field(default=None, ge=1)

    source_id: str = Field(min_length=1)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class EvidenceBlock(BaseModel):
    """Immutable evidence unit passed to prompt construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: EvidenceIdentity

    provenance: EvidenceProvenance

    rank: int = Field(ge=0)

    rerank_score: float | None = None

    content: str = Field(min_length=1)

    content_kind: Literal["untrusted_evidence"] = (
        "untrusted_evidence"
    )

    token_count: int = Field(
        default=0,
        ge=0,
    )

    @field_validator("rerank_score")
    @classmethod
    def validate_rerank_score(
        cls,
        value: float | None,
    ) -> float | None:
        if value is not None and not (
            -float("inf") < value < float("inf")
        ):
            raise ValueError(
                "rerank_score must be finite."
            )

        return value


class ContextBudgetStats(BaseModel):
    """Token-budget accounting for assembled evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_context_tokens: int = Field(ge=1)

    used_context_tokens: int = Field(
        default=0,
        ge=0,
    )

    dropped_evidence_items: int = Field(
        default=0,
        ge=0,
    )


class ContextBuilderMetrics(BaseModel):
    """Operational metrics for context assembly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_candidates: int = Field(
        default=0,
        ge=0,
    )

    accepted_evidence: int = Field(
        default=0,
        ge=0,
    )

    deduplicated_candidates: int = Field(
        default=0,
        ge=0,
    )

    dropped_for_budget: int = Field(
        default=0,
        ge=0,
    )


class AssembledContext(BaseModel):
    """Canonical output of the context-building stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)

    evidence: list[EvidenceBlock] = Field(
        default_factory=list
    )

    is_degraded: bool = False

    dropped_evidence_count: int = Field(
        default=0,
        ge=0,
    )

    budget: ContextBudgetStats

    metrics: ContextBuilderMetrics

    trace_id: str | None = None

    tokenizer_encoding: str


# ============================================================
# Prompt
# ============================================================

class PromptMessage(BaseModel):
    """Single message in the final LLM prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class LLMPrompt(BaseModel):
    """Canonical prompt contract passed to the LLM runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: list[PromptMessage] = Field(
        min_length=1
    )

    trace_id: str | None = None

    input_token_count: int = Field(
        default=0,
        ge=0,
    )

    reserved_output_tokens: int = Field(
        default=0,
        ge=0,
    )

    is_degraded: bool = False