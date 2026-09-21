"""LLM / VLM execution layer.

Responsibilities
----------------

- Validate LLM generation requests.
- Own model lifecycle.
- Execute local VLM inference.
- Enforce concurrency and execution timeout semantics.
- Validate images structurally.
- Normalize backend failures.
- Produce vendor-neutral generation contracts.
- Emit PII/PHI-safe operational logs.

This module does NOT:

- retrieve evidence
- rerank evidence
- build context
- build prompts
- perform OCR
- make clinical decisions
- decide refusal/safety policy
- map API responses
- download model weights

The public pipeline contracts do not expose Qwen-specific types.

Current backend:

    Qwen3-VL-8B-Instruct

Expected local weights:

    models/qwen3-vl-8b/

Model downloads are intentionally disabled.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.core.settings import (
    DeviceChoice,
    LLMSettings,
    Quantization,
    get_settings,
)


# ============================================================================
# Public Types
# ============================================================================

MessageRole = Literal["system", "user"]

Modality = Literal[
    "text",
    "image",
    "image_text",
]

ResolvedDevice = Literal[
    "cuda",
    "cpu",
]

FinishReason = Literal[
    "stop",
    "length",
]

MimeType = Literal[
    "image/jpeg",
    "image/png",
    "image/webp",
]


_SAFE_DIAGNOSTIC_KEYS = (
    "trace_id",
    "modality",
    "model_id",
    "backend_id",
    "timeout_seconds",
    "max_new_tokens",
    "image_count",
    "message_count",
)


# ============================================================================
# Exceptions
# ============================================================================


class LLMError(Exception):
    """Base class for all LLM execution failures.

    Diagnostics are intentionally restricted to PHI-safe operational fields.
    """

    error_code: str = "llm_error"

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        **diagnostics: Any,
    ) -> None:
        super().__init__(message)

        self.cause = cause

        if cause is not None:
            self.__cause__ = cause

        for key in _SAFE_DIAGNOSTIC_KEYS:
            setattr(self, key, diagnostics.get(key))


class InvalidLLMConfigurationError(LLMError):
    error_code = "invalid_llm_configuration"


class InvalidLLMInputError(LLMError):
    error_code = "invalid_llm_input"


class ModelNotReadyError(LLMError):
    error_code = "model_not_ready"


class LLMTimeoutError(LLMError):
    error_code = "llm_timeout"


class LLMBackendError(LLMError):
    error_code = "llm_backend_error"

    def __init__(
        self,
        message: str = "vlm backend failed",
        *,
        cause: BaseException | None = None,
        **diagnostics: Any,
    ) -> None:
        super().__init__(
            message,
            cause=cause,
            **diagnostics,
        )


class LLMConcurrencyError(LLMError):
    error_code = "llm_concurrency_error"


# ============================================================================
# Settings Integration
# ============================================================================


def get_llm_settings() -> LLMSettings:
    """Return the centrally managed LLM settings.

    No LLM-specific configuration is created locally.
    """
    return get_settings().llm


# ============================================================================
# Public Contracts
# ============================================================================


class ChatMessage(BaseModel):
    """Vendor-neutral text message."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(
        cls,
        value: str,
    ) -> MessageRole:
        if value not in {"system", "user"}:
            raise ValueError(
                "only system and user roles are allowed"
            )

        return value  # type: ignore[return-value]

    @field_validator("content")
    @classmethod
    def validate_content(
        cls,
        value: str,
    ) -> str:
        if not isinstance(value, str):
            raise ValueError("content must be str")

        return value

    @model_validator(mode="after")
    def validate_system_message(self) -> ChatMessage:
        if self.role == "system" and not self.content.strip():
            raise ValueError(
                "system message content cannot be empty"
            )

        return self


# ============================================================================
# Image Validation
# ============================================================================


_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_WEBP_RIFF = b"RIFF"
_WEBP_TAG = b"WEBP"


def sniff_image_mime(data: bytes) -> MimeType | None:
    """Detect image type using file signature rather than trusted labels."""

    if not data:
        return None

    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"

    if data.startswith(_PNG_MAGIC):
        return "image/png"

    if (
        len(data) >= 12
        and data.startswith(_WEBP_RIFF)
        and data[8:12] == _WEBP_TAG
    ):
        return "image/webp"

    return None


class ImagePart(BaseModel):
    """Validated image payload."""

    model_config = ConfigDict(extra="forbid")

    data: bytes
    mime_type: MimeType

    @field_validator("data")
    @classmethod
    def validate_data(
        cls,
        value: bytes,
    ) -> bytes:
        if isinstance(value, bytearray):
            value = bytes(value)

        if not isinstance(value, bytes) or not value:
            raise ValueError(
                "image data must be non-empty bytes"
            )

        return value

    @model_validator(mode="after")
    def validate_signature(self) -> ImagePart:
        detected = sniff_image_mime(self.data)

        if detected is None:
            raise ValueError(
                "unsupported image format"
            )

        if detected != self.mime_type:
            raise ValueError(
                "declared mime_type does not match image signature"
            )

        return self


# ============================================================================
# Generation Contracts
# ============================================================================


class LLMRequest(BaseModel):
    """Vendor-neutral multimodal generation request."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(
        min_length=1
    )

    images: list[ImagePart] = Field(
        default_factory=list
    )

    max_new_tokens: int | None = None
    trace_id: str | None = None

    @field_validator("max_new_tokens")
    @classmethod
    def validate_max_new_tokens(
        cls,
        value: int | None,
    ) -> int | None:
        if value is None:
            return None

        if isinstance(value, bool) or value <= 0:
            raise ValueError(
                "max_new_tokens must be > 0"
            )

        return value

    @model_validator(mode="after")
    def validate_request(self) -> LLMRequest:
        if not self.messages:
            raise ValueError(
                "messages must be non-empty"
            )

        has_images = bool(self.images)

        has_text = any(
            message.content.strip()
            for message in self.messages
        )

        if not has_text and not has_images:
            raise ValueError(
                "request must contain text or images"
            )

        for message in self.messages:
            if message.role == "system":
                if not message.content.strip():
                    raise ValueError(
                        "system message cannot be empty"
                    )

            if message.role == "user":
                if (
                    not message.content.strip()
                    and not has_images
                ):
                    raise ValueError(
                        "user message cannot be empty "
                        "without an image"
                    )

        return self


class GenerationChunk(BaseModel):
    """One streamed generation event."""

    model_config = ConfigDict(extra="forbid")

    text: str
    trace_id: str
    is_last: bool
    finish_reason: FinishReason | None = None

    @model_validator(mode="after")
    def validate_finish_reason(self) -> GenerationChunk:
        if self.is_last and self.finish_reason is None:
            raise ValueError(
                "last chunk requires finish_reason"
            )

        if (
            not self.is_last
            and self.finish_reason is not None
        ):
            raise ValueError(
                "intermediate chunk cannot have finish_reason"
            )

        return self


class GenerationResult(BaseModel):
    """Vendor-neutral final generation result."""

    model_config = ConfigDict(extra="forbid")

    text: str
    finish_reason: FinishReason
    output_token_count: int
    latency_ms: float
    model_id: str
    backend_id: str
    quantization: Quantization
    modality: Modality
    trace_id: str
    streamed: bool
    model_fingerprint: str

    @field_validator("output_token_count")
    @classmethod
    def validate_output_tokens(
        cls,
        value: int,
    ) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError(
                "output_token_count must be >= 0"
            )

        return value

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(
        cls,
        value: float,
    ) -> float:
        if value < 0:
            raise ValueError(
                "latency_ms must be >= 0"
            )

        return value


class BackendOutput(BaseModel):
    """Internal backend-neutral output."""

    model_config = ConfigDict(extra="forbid")

    text: str
    finish_reason: FinishReason
    output_token_count: int


# ============================================================================
# Observability
# ============================================================================


logger = logging.getLogger("rag.llm")

_trace_id_var: ContextVar[str] = ContextVar(
    "rag_llm_trace_id",
    default="",
)


def current_trace_id() -> str:
    return _trace_id_var.get()


def _log_event(
    event: str,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "trace_id": current_trace_id(),
        **fields,
    }

    logger.info("%s", payload)


def _safe_diag(
    **fields: Any,
) -> dict[str, Any]:
    return {
        key: fields[key]
        for key in _SAFE_DIAGNOSTIC_KEYS
        if key in fields
    }


# ============================================================================
# Runtime Utilities
# ============================================================================


def resolve_device(
    choice: DeviceChoice,
) -> ResolvedDevice:
    """Resolve requested execution device."""

    if choice == "cpu":
        return "cpu"

    if choice == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

        except Exception:
            pass

        raise InvalidLLMConfigurationError(
            "CUDA was explicitly requested but is unavailable"
        )

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except Exception:
        pass

    return "cpu"


def compute_model_fingerprint(
    *,
    backend_id: str,
    model_id: str,
    quantization: str,
    resolved_device: str,
    weights_dir: str,
) -> str:
    """Create an operational identity for the loaded model configuration."""

    root = Path(weights_dir)

    parts = [
        backend_id,
        model_id,
        quantization,
        resolved_device,
    ]

    config = root / "config.json"

    if config.is_file():
        parts.append(
            hashlib.sha256(
                config.read_bytes()
            ).hexdigest()
        )
    else:
        parts.append("no-config")

    snapshot: list[str] = []

    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            stat = path.stat()
            relative = path.relative_to(root).as_posix()

            snapshot.append(
                f"{relative}:{stat.st_size}:{int(stat.st_mtime)}"
            )

    parts.append(
        ";".join(snapshot)
        if snapshot
        else "no-weights"
    )

    return hashlib.sha256(
        "|".join(parts).encode("utf-8")
    ).hexdigest()


# ============================================================================
# Backend Abstraction
# ============================================================================


class IVLMBackend(ABC):
    """Vendor-neutral backend contract."""

    @abstractmethod
    def load(self) -> None:
        """Load model and processor."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release model resources."""

    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
        *,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> BackendOutput:
        """Run complete generation."""

    @abstractmethod
    def stream(
        self,
        request: LLMRequest,
        *,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> Iterator[
        tuple[str, bool, FinishReason | None]
    ]:
        """Produce generation chunks."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @property
    @abstractmethod
    def model_fingerprint(self) -> str:
        ...

    @property
    @abstractmethod
    def resolved_device(self) -> ResolvedDevice:
        ...


# ============================================================================
# Validation
# ============================================================================


def _validate_settings(
    settings: LLMSettings,
) -> LLMSettings:
    try:
        return LLMSettings.model_validate(
            settings.model_dump()
        )

    except Exception as exc:
        raise InvalidLLMConfigurationError(
            "invalid LLM settings",
            cause=exc,
        ) from exc


def _coerce_request(
    raw: LLMRequest | Mapping[str, Any],
) -> LLMRequest:
    if isinstance(raw, LLMRequest):
        return raw

    if isinstance(raw, Mapping):
        try:
            return LLMRequest.model_validate(raw)

        except ValidationError as exc:
            raise InvalidLLMInputError(
                "request does not match LLMRequest",
                cause=exc,
            ) from exc

    raise InvalidLLMInputError(
        "request must be LLMRequest or mapping"
    )


def derive_modality(
    messages: Sequence[ChatMessage],
    images: Sequence[ImagePart],
) -> Modality:
    has_images = bool(images)

    has_text = any(
        message.content.strip()
        for message in messages
    )

    if has_images and has_text:
        return "image_text"

    if has_images:
        return "image"

    return "text"


def validate_images(
    images: Sequence[ImagePart],
    settings: LLMSettings,
) -> None:
    if len(images) > settings.max_images:
        raise InvalidLLMInputError(
            f"image count exceeds max_images={settings.max_images}"
        )

    for image in images:
        if len(image.data) > settings.max_image_bytes:
            raise InvalidLLMInputError(
                "image exceeds max_image_bytes"
            )

        detected = sniff_image_mime(
            image.data
        )

        if detected is None:
            raise InvalidLLMInputError(
                "unsupported image format"
            )

        if detected != image.mime_type:
            raise InvalidLLMInputError(
                "image MIME does not match signature"
            )


def prepare_request(
    raw: LLMRequest | Mapping[str, Any],
    settings: LLMSettings,
) -> tuple[
    LLMRequest,
    Modality,
    int,
    str,
]:
    request = _coerce_request(raw)

    validate_images(
        request.images,
        settings,
    )

    modality = derive_modality(
        request.messages,
        request.images,
    )

    if modality == "text" and request.images:
        raise InvalidLLMInputError(
            "text modality cannot contain images"
        )

    if modality == "image" and not request.images:
        raise InvalidLLMInputError(
            "image modality requires images"
        )

    if modality == "image_text":
        if not request.images:
            raise InvalidLLMInputError(
                "image_text modality requires images"
            )

    max_new_tokens = (
        settings.max_new_tokens
        if request.max_new_tokens is None
        else request.max_new_tokens
    )

    if max_new_tokens > settings.max_new_tokens:
        raise InvalidLLMInputError(
            "requested max_new_tokens exceeds configured maximum"
        )

    trace_id = (
        request.trace_id.strip()
        if isinstance(request.trace_id, str)
        and request.trace_id.strip()
        else uuid.uuid4().hex
    )

    return (
        request,
        modality,
        max_new_tokens,
        trace_id,
    )


# ============================================================================
# Qwen3-VL Backend
# ============================================================================


class Qwen3VLBackend(IVLMBackend):
    """Concrete Qwen3-VL backend.

    Qwen-specific implementation details are intentionally isolated here.
    """

    def __init__(
        self,
        settings: LLMSettings,
    ) -> None:
        self._settings = settings
        self._loaded = False
        self._fingerprint = ""
        self._device: ResolvedDevice = "cpu"
        self._model: Any = None
        self._processor: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def resolved_device(self) -> ResolvedDevice:
        return self._device

    def load(self) -> None:
        if self._loaded:
            return

        self._device = resolve_device(
            self._settings.device
        )

        weights = Path(
            self._settings.local_weights_dir
        )

        if not weights.exists():
            raise ModelNotReadyError(
                "local VLM weights are not available",
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
            )

        if not weights.is_dir():
            raise ModelNotReadyError(
                "local VLM weights path is not a directory",
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
            )

        try:
            import torch

            from transformers import (
                AutoProcessor,
                Qwen3VLForConditionalGeneration,
            )

        except Exception as exc:
            raise ModelNotReadyError(
                "required VLM runtime dependencies are unavailable",
                cause=exc,
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
            ) from exc

        try:
            processor = AutoProcessor.from_pretrained(
                str(weights),
                local_files_only=True,
                trust_remote_code=False,
            )

            model_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": False,
            }

            if self._device == "cuda":
                model_kwargs["device_map"] = "auto"

                if self._settings.quantization == "none":
                    model_kwargs["torch_dtype"] = torch.bfloat16

            else:
                model_kwargs["torch_dtype"] = torch.float32

            if self._settings.quantization in {
                "4bit",
                "8bit",
            }:
                try:
                    from transformers import (
                        BitsAndBytesConfig,
                    )

                except Exception as exc:
                    raise ModelNotReadyError(
                        "bitsandbytes/transformers quantization support is unavailable",
                        cause=exc,
                        model_id=self._settings.model_id,
                        backend_id=self._settings.backend_id,
                    ) from exc

                quantization_kwargs: dict[str, Any] = {
                    "load_in_4bit": (
                        self._settings.quantization
                        == "4bit"
                    ),
                    "load_in_8bit": (
                        self._settings.quantization
                        == "8bit"
                    ),
                }

                if self._device == "cuda":
                    quantization_kwargs[
                        "bnb_4bit_compute_dtype"
                    ] = torch.float16

                    quantization_kwargs[
                        "bnb_4bit_quant_type"
                    ] = "nf4"

                    quantization_kwargs[
                        "bnb_4bit_use_double_quant"
                    ] = True

                model_kwargs[
                    "quantization_config"
                ] = BitsAndBytesConfig(
                    **quantization_kwargs
                )

            model = (
                Qwen3VLForConditionalGeneration
                .from_pretrained(
                    str(weights),
                    **model_kwargs,
                )
            )

            if self._device == "cpu":
                model = model.to("cpu")

            model.eval()

            self._processor = processor
            self._model = model

        except ModelNotReadyError:
            raise

        except Exception as exc:
            raise ModelNotReadyError(
                "Qwen3-VL model failed to load",
                cause=exc,
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
            ) from exc

        self._fingerprint = compute_model_fingerprint(
            backend_id=self._settings.backend_id,
            model_id=self._settings.model_id,
            quantization=self._settings.quantization,
            resolved_device=self._device,
            weights_dir=str(weights),
        )

        self._loaded = True

    def shutdown(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        self._fingerprint = ""

    def generate(
        self,
        request: LLMRequest,
        *,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> BackendOutput:
        if not self._loaded:
            raise ModelNotReadyError(
                "Qwen3-VL backend is not loaded"
            )

        try:
            (
                text,
                output_token_count,
                finish_reason,
            ) = self._generate_blocking(
                request,
                max_new_tokens=max_new_tokens,
            )

            return BackendOutput(
                text=text,
                finish_reason=finish_reason,
                output_token_count=output_token_count,
            )

        except LLMError:
            raise

        except Exception as exc:
            raise LLMBackendError(
                cause=exc,
                trace_id=trace_id,
                modality=modality,
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
            ) from exc

    def stream(
        self,
        request: LLMRequest,
        *,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> Iterator[
        tuple[str, bool, FinishReason | None]
    ]:
        """Compatibility streaming interface.

        Current Base RAG implementation performs one complete inference and
        emits one final chunk.

        True token-level streaming is intentionally deferred because it
        requires a dedicated generation streamer and cancellation contract.
        """

        output = self.generate(
            request,
            max_new_tokens=max_new_tokens,
            modality=modality,
            trace_id=trace_id,
        )

        yield (
            output.text,
            True,
            output.finish_reason,
        )

    def _build_messages(
        self,
        request: LLMRequest,
    ) -> list[dict[str, Any]]:
        """Convert public messages/images into Qwen multimodal messages."""

        image_payloads = [
            {
                "type": "image",
                "image": self._image_to_pil(image),
            }
            for image in request.images
        ]

        messages: list[dict[str, Any]] = []

        image_attached = False

        for message in request.messages:
            if (
                message.role == "user"
                and image_payloads
                and not image_attached
            ):
                content: list[dict[str, Any]] = []

                content.extend(image_payloads)

                if message.content.strip():
                    content.append(
                        {
                            "type": "text",
                            "text": message.content,
                        }
                    )

                messages.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )

                image_attached = True

            else:
                content = []

                if message.content.strip():
                    content.append(
                        {
                            "type": "text",
                            "text": message.content,
                        }
                    )

                messages.append(
                    {
                        "role": message.role,
                        "content": content,
                    }
                )

        return messages

    @staticmethod
    def _image_to_pil(
        image: ImagePart,
    ) -> Any:
        try:
            from io import BytesIO

            from PIL import Image

            with Image.open(
                BytesIO(image.data)
            ) as pil_image:
                return pil_image.convert("RGB")

        except Exception as exc:
            raise LLMBackendError(
                "failed to decode image",
                cause=exc,
            ) from exc

    def _generate_blocking(
        self,
        request: LLMRequest,
        *,
        max_new_tokens: int,
    ) -> tuple[
        str,
        int,
        FinishReason,
    ]:
        import torch

        if self._model is None:
            raise ModelNotReadyError(
                "Qwen3-VL model is unavailable"
            )

        if self._processor is None:
            raise ModelNotReadyError(
                "Qwen3-VL processor is unavailable"
            )

        messages = self._build_messages(
            request
        )

        inputs = (
            self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        if "token_type_ids" in inputs:
            inputs.pop(
                "token_type_ids",
                None,
            )

        inputs = self._move_inputs_to_device(
            inputs
        )

        input_ids = inputs.get(
            "input_ids"
        )

        if input_ids is None:
            raise LLMBackendError(
                "processor did not produce input_ids"
            )

        input_token_count = (
            input_ids.shape[-1]
        )

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                **generate_kwargs,
            )

        generated_ids_trimmed = [
            output_ids[input_token_count:]
            for output_ids in generated_ids
        ]

        if not generated_ids_trimmed:
            raise LLMBackendError(
                "model returned no generated sequence"
            )

        generated_sequence = (
            generated_ids_trimmed[0]
        )

        output_token_count = int(
            generated_sequence.shape[-1]
        )

        text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        text = text.strip()

        finish_reason: FinishReason = (
            "length"
            if output_token_count >= max_new_tokens
            else "stop"
        )

        return (
            text,
            output_token_count,
            finish_reason,
        )

    def _move_inputs_to_device(
        self,
        inputs: Any,
    ) -> Any:
        """Move processor tensors to the model execution device."""

        if self._model is None:
            raise ModelNotReadyError(
                "model is unavailable"
            )

        device = self._model.device

        if hasattr(inputs, "to"):
            return inputs.to(device)

        if isinstance(inputs, Mapping):
            return {
                key: (
                    value.to(device)
                    if hasattr(value, "to")
                    else value
                )
                for key, value in inputs.items()
            }

        return inputs


# ============================================================================
# Runtime
# ============================================================================


class LLMRuntime:
    """Owns lifecycle, concurrency, timeout, and execution boundaries."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        backend: IVLMBackend | None = None,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._settings = _validate_settings(
            settings or get_llm_settings()
        )

        self._backend = (
            backend
            if backend is not None
            else Qwen3VLBackend(
                self._settings
            )
        )

        if executor is not None:
            self._executor = executor
            self._owns_executor = False

        else:
            workers = max(
                self._settings.executor_max_workers,
                self._settings.max_concurrent,
            )

            self._executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="rag-llm",
            )

            self._owns_executor = True

        self._semaphore = asyncio.Semaphore(
            self._settings.max_concurrent
        )

        self._load_lock = asyncio.Lock()

        self._ready = False
        self._startup_called = False
        self._shutdown = False

    @property
    def settings(self) -> LLMSettings:
        return self._settings

    @property
    def backend(self) -> IVLMBackend:
        return self._backend

    @property
    def is_ready(self) -> bool:
        return (
            not self._shutdown
            and self._ready
            and self._backend.is_loaded
        )

    @property
    def model_fingerprint(self) -> str:
        if not self.is_ready:
            raise ModelNotReadyError(
                "model is not ready"
            )

        return self._backend.model_fingerprint

    async def startup(self) -> None:
        if self._shutdown:
            raise ModelNotReadyError(
                "LLM runtime has been shut down"
            )

        self._startup_called = True

        if not self._settings.require_model_at_startup:
            _log_event(
                "llm_startup_deferred",
                backend_id=self._settings.backend_id,
                model_id=self._settings.model_id,
            )

            return

        await self._load_backend()

    async def shutdown(self) -> None:
        if self._shutdown:
            return

        try:
            async with self._load_lock:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        self._backend.shutdown,
                    )

                except LLMError:
                    raise

                except Exception as exc:
                    raise LLMBackendError(
                        cause=exc
                    ) from exc

                finally:
                    self._ready = False

        finally:
            self._shutdown = True

            if self._owns_executor:
                self._executor.shutdown(
                    wait=True
                )

                self._owns_executor = False

    async def generate(
        self,
        request: LLMRequest | Mapping[str, Any],
    ) -> GenerationResult:
        (
            prepared,
            modality,
            max_new_tokens,
            trace_id,
        ) = prepare_request(
            request,
            self._settings,
        )

        token = _trace_id_var.set(
            trace_id
        )

        started_at = time.perf_counter()

        try:
            await self._ensure_loaded()

            max_new_tokens = (
                self._effective_max_new_tokens(
                    max_new_tokens
                )
            )

            diagnostics = _safe_diag(
                trace_id=trace_id,
                modality=modality,
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
                timeout_seconds=self._inference_timeout_seconds(),
                max_new_tokens=max_new_tokens,
                image_count=len(prepared.images),
                message_count=len(prepared.messages),
            )

            _log_event(
                "llm_generate_start",
                modality=modality,
                image_count=len(prepared.images),
                message_count=len(prepared.messages),
                max_new_tokens=max_new_tokens,
            )

            output = await self._execute_with_timeout(
                prepared,
                max_new_tokens,
                modality,
                trace_id,
                diagnostics,
            )

            latency_ms = (
                time.perf_counter()
                - started_at
            ) * 1000.0

            result = GenerationResult(
                text=output.text,
                finish_reason=output.finish_reason,
                output_token_count=output.output_token_count,
                latency_ms=latency_ms,
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
                quantization=self._settings.quantization,
                modality=modality,
                trace_id=trace_id,
                streamed=False,
                model_fingerprint=self._backend.model_fingerprint,
            )

            _log_event(
                "llm_generate_done",
                modality=modality,
                finish_reason=result.finish_reason,
                output_token_count=result.output_token_count,
                latency_ms=round(
                    latency_ms,
                    3,
                ),
            )

            return result

        finally:
            _trace_id_var.reset(token)

    def _inference_timeout_seconds(self) -> float:
        timeout = self._settings.timeout_seconds

        if self._backend.resolved_device == "cpu":
            return max(timeout, 600.0)

        return timeout

    def _effective_max_new_tokens(
        self,
        max_new_tokens: int,
    ) -> int:
        if self._backend.resolved_device == "cpu":
            return min(max_new_tokens, 32)

        return max_new_tokens

    async def _execute_with_timeout(
        self,
        request: LLMRequest,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
        diagnostics: dict[str, Any],
    ) -> BackendOutput:
        await self._semaphore.acquire()

        loop = asyncio.get_running_loop()

        future = loop.run_in_executor(
            self._executor,
            lambda: self._invoke_generate(
                request,
                max_new_tokens,
                modality,
                trace_id,
            ),
        )

        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._inference_timeout_seconds(),
            )

        except TimeoutError as exc:
            asyncio.create_task(
                self._release_after_completion(
                    future
                )
            )

            raise LLMTimeoutError(
                "vlm inference exceeded timeout",
                cause=exc,
                **diagnostics,
            ) from exc

        except asyncio.CancelledError:
            asyncio.create_task(
                self._release_after_completion(
                    future
                )
            )

            raise

        except LLMError:
            self._semaphore.release()
            raise

        except Exception as exc:
            self._semaphore.release()

            raise LLMBackendError(
                cause=exc,
                **diagnostics,
            ) from exc

        else:
            self._semaphore.release()

    async def _release_after_completion(
        self,
        future: asyncio.Future[Any],
    ) -> None:
        try:
            await asyncio.shield(
                future
            )

        except BaseException:
            pass

        finally:
            self._semaphore.release()

    async def stream(
        self,
        request: LLMRequest | Mapping[str, Any],
    ) -> AsyncIterator[GenerationChunk]:
        """Compatibility streaming surface.

        The current backend emits one final chunk after complete inference.

        True token streaming is intentionally deferred.
        """

        (
            prepared,
            modality,
            max_new_tokens,
            trace_id,
        ) = prepare_request(
            request,
            self._settings,
        )

        token = _trace_id_var.set(
            trace_id
        )

        try:
            await self._ensure_loaded()

            max_new_tokens = (
                self._effective_max_new_tokens(
                    max_new_tokens
                )
            )

            diagnostics = _safe_diag(
                trace_id=trace_id,
                modality=modality,
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
                timeout_seconds=self._inference_timeout_seconds(),
                max_new_tokens=max_new_tokens,
                image_count=len(prepared.images),
                message_count=len(prepared.messages),
            )

            await self._semaphore.acquire()

            loop = asyncio.get_running_loop()

            future = loop.run_in_executor(
                self._executor,
                lambda: self._invoke_generate(
                    prepared,
                    max_new_tokens,
                    modality,
                    trace_id,
                ),
            )

            try:
                output = await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=self._inference_timeout_seconds(),
                )

            except TimeoutError as exc:
                asyncio.create_task(
                    self._release_after_completion(
                        future
                    )
                )

                raise LLMTimeoutError(
                    "vlm inference exceeded timeout",
                    cause=exc,
                    **diagnostics,
                ) from exc

            except asyncio.CancelledError:
                asyncio.create_task(
                    self._release_after_completion(
                        future
                    )
                )

                raise

            except LLMError:
                self._semaphore.release()
                raise

            except Exception as exc:
                self._semaphore.release()

                raise LLMBackendError(
                    cause=exc,
                    **diagnostics,
                ) from exc

            else:
                self._semaphore.release()

            yield GenerationChunk(
                text=output.text,
                trace_id=trace_id,
                is_last=True,
                finish_reason=output.finish_reason,
            )

        finally:
            _trace_id_var.reset(token)

    async def _ensure_loaded(self) -> None:
        if self._shutdown:
            raise ModelNotReadyError(
                "LLM runtime has been shut down"
            )

        if (
            self._ready
            and self._backend.is_loaded
        ):
            return

        if (
            self._settings.require_model_at_startup
            and not self._startup_called
        ):
            raise ModelNotReadyError(
                "LLMRuntime.startup() must be called before generation"
            )

        if (
            self._settings.require_model_at_startup
            and self._startup_called
            and not self._ready
        ):
            raise ModelNotReadyError(
                "model failed to become ready during startup"
            )

        await self._load_backend()

    async def _load_backend(self) -> None:
        async with self._load_lock:
            if (
                self._ready
                and self._backend.is_loaded
            ):
                return

            try:
                await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    self._backend.load,
                )

            except ModelNotReadyError:
                raise

            except LLMError as exc:
                raise ModelNotReadyError(
                    "VLM backend failed during load",
                    cause=exc,
                    model_id=self._settings.model_id,
                    backend_id=self._settings.backend_id,
                ) from exc

            except Exception as exc:
                raise ModelNotReadyError(
                    "VLM backend failed during load",
                    cause=exc,
                    model_id=self._settings.model_id,
                    backend_id=self._settings.backend_id,
                ) from exc

            if not self._backend.is_loaded:
                raise ModelNotReadyError(
                    "backend.load() returned without becoming ready"
                )

            self._ready = True

            _log_event(
                "llm_model_ready",
                backend_id=self._settings.backend_id,
                model_id=self._settings.model_id,
                quantization=self._settings.quantization,
                resolved_device=self._backend.resolved_device,
                fingerprint_prefix=(
                    self._backend.model_fingerprint[:12]
                ),
            )

    def _invoke_generate(
        self,
        request: LLMRequest,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> BackendOutput:
        if not self._backend.is_loaded:
            raise ModelNotReadyError(
                "VLM backend is not loaded"
            )

        try:
            return self._backend.generate(
                request,
                max_new_tokens=max_new_tokens,
                modality=modality,
                trace_id=trace_id,
            )

        except LLMError:
            raise

        except Exception as exc:
            raise LLMBackendError(
                cause=exc,
                trace_id=trace_id,
                modality=modality,
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
            ) from exc


# ============================================================================
# Factory
# ============================================================================


def build_llm_runtime(
    settings: LLMSettings | None = None,
    backend: IVLMBackend | None = None,
) -> LLMRuntime:
    """Create an LLM runtime.

    Dependency injection is supported only for testing/replacement of the
    backend; normal production construction uses Qwen3VLBackend.
    """

    return LLMRuntime(
        settings=settings,
        backend=backend,
    )