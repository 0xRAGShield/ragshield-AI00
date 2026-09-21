"""
Central application configuration.

This module contains configuration only.
No retrieval, ranking, prompting, or generation logic belongs here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


Quantization = Literal["4bit", "8bit", "none"]
DeviceChoice = Literal["auto", "cuda", "cpu"]
BackendId = Literal["qwen3_vl"]


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_name: str = "ragshield_test"
    qdrant_mode: Literal["server", "local"] = "server"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    default_top_k: int = Field(default=5, ge=1, le=100)
    max_top_k: int = Field(default=20, ge=1, le=100)

    embedding_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
    )

    search_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
    )

    executor_workers: int = Field(
        default=2,
        ge=1,
    )

    allow_degraded_results: bool = True

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "collection_name must not be empty."
            )

        return value


class RerankerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_n: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    max_top_n: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    timeout_seconds: float = Field(
        default=10.0,
        gt=0,
    )

    executor_max_workers: int = Field(
        default=2,
        ge=1,
    )

    score_threshold: float | None = None

    higher_is_better: bool = True

    fallback_to_retrieval: bool = True


class ContextSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_context_tokens: int = Field(
        default=3000,
        ge=1,
    )

    max_evidence_items: int = Field(
        default=8,
        ge=1,
    )

    prompt_reserve_tokens: int = Field(
        default=1000,
        ge=0,
    )

    query_reserve_tokens: int = Field(
        default=256,
        ge=0,
    )

    output_reserve_tokens: int = Field(
        default=1024,
        ge=0,
    )

    metadata_allowlist: Sequence[str] = (
        "file_name",
        "file_extension",
        "page",
        "section",
        "title",
        "source",
        "source_id",
        "chunk_id",
        "media_type",
    )

    require_exact_tokenizer: bool = False

    tokenizer_encoding: str = "cl100k_base"

    estimated_chars_per_token: float = Field(
        default=4.0,
        gt=0,
    )


class PromptSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_prompt_tokens: int = Field(
        default=4000,
        ge=1,
    )

    output_reserve_tokens: int = Field(
        default=1024,
        ge=0,
    )

    safety_mode: Literal["default", "strict"] = "strict"

    require_exact_tokenizer: bool = False

    tokenizer_encoding: str = "cl100k_base"

    estimated_chars_per_token: float = Field(
        default=4.0,
        gt=0,
    )


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_id: BackendId = "qwen3_vl"

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"

    local_weights_dir: str = "models/qwen3-vl-8b"

    quantization: Quantization = "4bit"

    device: DeviceChoice = "auto"

    max_new_tokens: int = Field(
        default=1024,
        ge=1,
    )

    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
    )

    max_concurrent: int = Field(
        default=1,
        ge=1,
    )

    require_model_at_startup: bool = True

    max_image_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1,
    )

    max_images: int = Field(
        default=1,
        ge=1,
    )

    executor_max_workers: int = Field(
        default=2,
        ge=1,
    )

    @field_validator(
        "model_id",
        "local_weights_dir",
    )
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "value must not be empty"
            )

        return value

    @property
    def weights_path(self) -> Path:
        return Path(self.local_weights_dir)


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalSettings = Field(
        default_factory=RetrievalSettings
    )

    reranker: RerankerSettings = Field(
        default_factory=RerankerSettings
    )

    context: ContextSettings = Field(
        default_factory=ContextSettings
    )

    prompt: PromptSettings = Field(
        default_factory=PromptSettings
    )

    llm: LLMSettings = Field(
        default_factory=LLMSettings
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()