"""Application configuration.

Central source of truth for runtime configuration.
No inference or business logic belongs here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Quantization = Literal["4bit", "8bit", "none"]
DeviceChoice = Literal["auto", "cuda", "cpu"]
BackendId = Literal["qwen3_vl"]


class LLMSettings(BaseModel):
    """Configuration for the local VLM execution layer."""

    model_config = ConfigDict(extra="forbid")

    backend_id: BackendId = "qwen3_vl"

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"

    local_weights_dir: str = "models/qwen3-vl-8b"

    quantization: Quantization = "4bit"

    device: DeviceChoice = "auto"

    max_new_tokens: int = Field(default=1024, ge=1)

    timeout_seconds: float = Field(default=60.0, gt=0)

    max_concurrent: int = Field(default=1, ge=1)

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

    @field_validator("model_id", "local_weights_dir")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("value must not be empty")

        return value

    @property
    def weights_path(self) -> Path:
        """Resolve the local model weights directory."""

        return Path(self.local_weights_dir)


class AppSettings(BaseModel):
    """Root application configuration."""

    model_config = ConfigDict(extra="forbid")

    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the process-wide immutable-by-convention application settings."""

    return AppSettings()