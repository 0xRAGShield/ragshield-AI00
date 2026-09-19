"""LLM / VLM execution layer.

This module runs inference only. It does not retrieve, rerank, pack context,
build prompts, run OCR, decide refusal, or map API responses.

Public pipeline types do not name Qwen. The current backend is Qwen3-VL;
swapping implementations later must not change LLMRequest / GenerationResult.

Weights are expected at models/qwen3-vl-8b/ (not Git). This file does not
download models.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

MessageRole = Literal["system", "user"]
Modality = Literal["text", "image", "image_text"]
Quantization = Literal["4bit", "8bit", "none"]
DeviceChoice = Literal["auto", "cuda", "cpu"]
ResolvedDevice = Literal["cuda", "cpu"]
FinishReason = Literal["stop", "length"]
MimeType = Literal["image/jpeg", "image/png", "image/webp"]
BackendId = Literal["qwen3_vl"]

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


# ---------------------------------------------------------------------------
# Exceptions — fail closed; never swallowed into an empty generation
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for execution-layer failures. Diagnostics never include PHI."""

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
    """Vendor/runtime failure. Public message is generic; __cause__ is internal."""

    error_code = "llm_backend_error"

    def __init__(
        self,
        message: str = "vlm backend failed",
        *,
        cause: BaseException | None = None,
        **diagnostics: Any,
    ) -> None:
        super().__init__(message, cause=cause, **diagnostics)


class LLMConcurrencyError(LLMError):
    error_code = "llm_concurrency_error"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class LLMSettings(BaseModel):
    """Execution knobs. Quantization is load-time, not a per-request switch."""

    model_config = ConfigDict(extra="forbid")

    backend_id: BackendId = "qwen3_vl"
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    local_weights_dir: str = "models/qwen3-vl-8b"
    quantization: Quantization = "none"
    device: DeviceChoice = "auto"
    max_new_tokens: int = 1024
    timeout_seconds: float = 60.0
    max_concurrent: int = 1
    require_model_at_startup: bool = Field(
        default=True,
        description="True: startup() loads the model. False: first generate/stream lazy-loads once.",
    )
    max_image_bytes: int = 10 * 1024 * 1024
    max_images: int = 1
    executor_max_workers: int = 2
    tokenizer_encoding: str | None = None

    @field_validator("model_id", "local_weights_dir")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value.strip()

    @field_validator("max_new_tokens", "max_concurrent", "max_images", "max_image_bytes", "executor_max_workers")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("timeout_seconds must be > 0")
        return float(value)


def get_llm_settings() -> LLMSettings:
    return LLMSettings()


# ---------------------------------------------------------------------------
# Public contracts — no Qwen types
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """Text-only chat turn. Images belong on LLMRequest.images, never here.

    Empty/whitespace ``content`` is rejected for ``system``. A ``user`` turn
    may be empty only so an image-only LLMRequest can omit caption text.
    """

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str

    @field_validator("role")
    @classmethod
    def _system_or_user(cls, value: object) -> MessageRole:
        if value not in {"system", "user"}:
            raise ValueError("assistant/tool roles are not allowed")
        return value  # type: ignore[return-value]

    @field_validator("content")
    @classmethod
    def _content_is_str(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("content must be str")
        return value

    @model_validator(mode="after")
    def _system_not_empty(self) -> ChatMessage:
        if self.role == "system" and not self.content.strip():
            raise ValueError("system message content cannot be empty")
        return self


# Magic constants stay with image validation (not trusted MIME labels).
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_WEBP_RIFF = b"RIFF"
_WEBP_TAG = b"WEBP"


def sniff_image_mime(data: bytes) -> MimeType | None:
    """Return the MIME implied by magic bytes, or None if unrecognized."""

    if not data:
        return None
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    if len(data) >= 12 and data.startswith(_WEBP_RIFF) and data[8:12] == _WEBP_TAG:
        return "image/webp"
    return None


class ImagePart(BaseModel):
    """One image. Declared MIME must match magic bytes of ``data``."""

    model_config = ConfigDict(extra="forbid")

    data: bytes
    mime_type: MimeType

    @field_validator("data")
    @classmethod
    def _non_empty_bytes(cls, value: object) -> bytes:
        if isinstance(value, bytearray):
            value = bytes(value)
        if not isinstance(value, bytes) or not value:
            raise ValueError("data must be non-empty bytes")
        return value

    @model_validator(mode="after")
    def _bytes_match_declared_mime(self) -> ImagePart:
        sniffed = sniff_image_mime(self.data)
        if sniffed is None:
            raise ValueError("image magic bytes are not jpeg/png/webp")
        if sniffed != self.mime_type:
            raise ValueError("declared mime_type does not match image signature")
        return self


class LLMRequest(BaseModel):
    """Multimodal generation request.

    Fields: messages, images, max_new_tokens, trace_id.
    ``modality`` is derived (text | image | image_text) and is not an input field.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    images: list[ImagePart] = Field(default_factory=list)
    max_new_tokens: int | None = None
    trace_id: str | None = None

    @field_validator("max_new_tokens")
    @classmethod
    def _max_new_positive(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if isinstance(value, bool) or value <= 0:
            raise ValueError("max_new_tokens must be an int > 0")
        return value

    @model_validator(mode="after")
    def _lock_modality_relations(self) -> LLMRequest:
        if not self.messages:
            raise ValueError("messages must be a non-empty list")
        has_images = len(self.images) > 0
        has_text = any(isinstance(m.content, str) and m.content.strip() for m in self.messages)
        for message in self.messages:
            if message.role not in {"system", "user"}:
                raise ValueError("assistant/tool roles are not allowed")
            stripped = message.content.strip() if isinstance(message.content, str) else ""
            if message.role == "system" and not stripped:
                raise ValueError("system message content cannot be empty")
            if message.role == "user" and not stripped and not has_images:
                raise ValueError("user message content cannot be empty unless the request is image-only")
        if has_images and has_text:
            return self
        if has_images:
            return self
        if not has_text:
            raise ValueError("text request requires non-empty message content")
        return self


class GenerationChunk(BaseModel):
    """One streamed delta. ``finish_reason`` is required only on the last chunk."""

    model_config = ConfigDict(extra="forbid")

    text: str
    trace_id: str
    is_last: bool
    finish_reason: FinishReason | None = None

    @model_validator(mode="after")
    def _last_has_reason(self) -> GenerationChunk:
        if self.is_last and self.finish_reason is None:
            raise ValueError("finish_reason is required on the last chunk")
        if not self.is_last and self.finish_reason is not None:
            raise ValueError("finish_reason must be None on intermediate chunks")
        return self


class GenerationResult(BaseModel):
    """Unified generation output. Never a vendor object."""

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


class BackendOutput(BaseModel):
    """Internal backend return. Not a vendor object."""

    model_config = ConfigDict(extra="forbid")

    text: str
    finish_reason: FinishReason
    output_token_count: int


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

logger = logging.getLogger("rag.llm")
_trace_id_var: ContextVar[str] = ContextVar("rag_llm_trace_id", default="")


def current_trace_id() -> str:
    return _trace_id_var.get()


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "trace_id": current_trace_id(), **fields}
    logger.info("%s", payload)


def _safe_diag(**fields: Any) -> dict[str, Any]:
    return {key: fields[key] for key in _SAFE_DIAGNOSTIC_KEYS if key in fields}


def _map_backend_exception(exc: BaseException, **diag: Any) -> LLMError:
    if isinstance(exc, LLMError):
        return exc
    return LLMBackendError(cause=exc, **diag)


def resolve_device(choice: DeviceChoice) -> ResolvedDevice:
    """Backend/runtime responsibility. Callers must not assume CUDA exists."""

    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda"
    try:
        import torch  # type: ignore[import-not-found]

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
    """Deterministic identity of loaded model/config/weights — not model_id alone."""

    parts = [backend_id, model_id, quantization, resolved_device]
    root = Path(weights_dir)
    config = root / "config.json"
    if config.is_file():
        parts.append(hashlib.sha256(config.read_bytes()).hexdigest())
    else:
        parts.append("no-config")
    snapshot: list[str] = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".safetensors", ".bin", ".pt", ".index"}:
                if path.name not in {"config.json", "generation_config.json", "preprocessor_config.json"}:
                    continue
            stat = path.stat()
            rel = path.relative_to(root).as_posix()
            snapshot.append(f"{rel}:{stat.st_size}:{int(stat.st_mtime)}")
    parts.append(";".join(snapshot) if snapshot else "no-weights")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class IVLMBackend(ABC):
    """Vendor-agnostic VLM backend. Qwen is one implementation.

    Contract: load, generate, stream, shutdown, model_fingerprint,
    plus is_loaded and resolved_device.
    """

    @abstractmethod
    def load(self) -> None:
        """Blocking load. Runtime runs this on a worker thread."""

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
        *,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> BackendOutput: ...

    @abstractmethod
    def stream(
        self,
        request: LLMRequest,
        *,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> Iterator[tuple[str, bool, FinishReason | None]]:
        """Yield (delta, is_last, finish_reason). finish_reason only on last."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def model_fingerprint(self) -> str: ...

    @property
    @abstractmethod
    def resolved_device(self) -> ResolvedDevice: ...


# ---------------------------------------------------------------------------
# Validation (structural only — not medical/refusal policy)
# ---------------------------------------------------------------------------


def _validate_settings(settings: LLMSettings) -> LLMSettings:
    try:
        return LLMSettings.model_validate(settings.model_dump())
    except Exception as exc:
        raise InvalidLLMConfigurationError(str(exc), cause=exc) from exc


def _coerce_message(raw: Any) -> ChatMessage:
    if isinstance(raw, ChatMessage):
        return raw
    if isinstance(raw, Mapping):
        try:
            return ChatMessage.model_validate(raw)
        except ValidationError as exc:
            raise InvalidLLMInputError("message does not match ChatMessage", cause=exc) from exc
    role = getattr(raw, "role", None)
    content = getattr(raw, "content", None)
    try:
        return ChatMessage.model_validate({"role": role, "content": content})
    except ValidationError as exc:
        raise InvalidLLMInputError("message does not match ChatMessage", cause=exc) from exc


def _coerce_request(raw: LLMRequest | Mapping[str, Any] | Any) -> LLMRequest:
    if isinstance(raw, LLMRequest):
        return raw
    if isinstance(raw, Mapping):
        try:
            return LLMRequest.model_validate(raw)
        except ValidationError as exc:
            raise InvalidLLMInputError("request does not match LLMRequest", cause=exc) from exc
    raise InvalidLLMInputError("request must be LLMRequest")


def derive_modality(messages: Sequence[ChatMessage], images: Sequence[ImagePart]) -> Modality:
    has_image = len(images) > 0
    has_text = any(isinstance(m.content, str) and m.content.strip() for m in messages)
    if has_image and has_text:
        return "image_text"
    if has_image:
        return "image"
    return "text"


def validate_images(images: Sequence[ImagePart], settings: LLMSettings) -> None:
    if len(images) > settings.max_images:
        raise InvalidLLMInputError(
            f"image count {len(images)} exceeds max_images {settings.max_images}"
        )
    for image in images:
        if not image.data:
            raise InvalidLLMInputError("image data is empty")
        if len(image.data) > settings.max_image_bytes:
            raise InvalidLLMInputError("image exceeds max_image_bytes")
        sniffed = sniff_image_mime(image.data)
        if sniffed is None:
            raise InvalidLLMInputError("image magic bytes are not jpeg/png/webp")
        if sniffed != image.mime_type:
            raise InvalidLLMInputError("declared mime_type does not match image signature")


def prepare_request(
    raw: LLMRequest | Mapping[str, Any] | Any,
    settings: LLMSettings,
) -> tuple[LLMRequest, Modality, int, str]:
    request = _coerce_request(raw)
    messages = [_coerce_message(item) for item in request.messages]
    if not messages:
        raise InvalidLLMInputError("messages must be a non-empty list")
    request = request.model_copy(update={"messages": messages})
    validate_images(request.images, settings)
    modality = derive_modality(request.messages, request.images)
    for message in request.messages:
        if not isinstance(message.content, str):
            raise InvalidLLMInputError("ChatMessage.content must be str")
        if message.role not in {"system", "user"}:
            raise InvalidLLMInputError("assistant/tool roles are not allowed")
        stripped = message.content.strip()
        if message.role == "system" and not stripped:
            raise InvalidLLMInputError("system message content cannot be empty")
        if message.role == "user" and not stripped and modality != "image":
            raise InvalidLLMInputError(
                "user message content cannot be empty unless the request is image-only"
            )
    if modality == "text":
        if request.images:
            raise InvalidLLMInputError("text modality cannot include images")
        if not any(m.content.strip() for m in request.messages):
            raise InvalidLLMInputError("text request requires non-empty message content")
    elif modality == "image":
        if not request.images:
            raise InvalidLLMInputError("image-only request requires images")
    elif modality == "image_text":
        if not request.images:
            raise InvalidLLMInputError("image_text request requires images")
        if not any(m.content.strip() for m in request.messages):
            raise InvalidLLMInputError("image_text request requires non-empty message content")
    max_new = settings.max_new_tokens if request.max_new_tokens is None else request.max_new_tokens
    if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new <= 0:
        raise InvalidLLMInputError("max_new_tokens must be an int > 0")
    if max_new > settings.max_new_tokens:
        raise InvalidLLMInputError("max_new_tokens exceeds settings.max_new_tokens")
    trace_id = request.trace_id.strip() if isinstance(request.trace_id, str) and request.trace_id.strip() else uuid.uuid4().hex
    return request, modality, max_new, trace_id


# ---------------------------------------------------------------------------
# Qwen3-VL implementation — isolated; optional deps
# ---------------------------------------------------------------------------


class Qwen3VLBackend(IVLMBackend):
    """Current VLM implementation. Not referenced by LLMRequest / GenerationResult."""

    def __init__(self, settings: LLMSettings) -> None:
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
        self._device = resolve_device(self._settings.device)
        weights = Path(self._settings.local_weights_dir)
        if not weights.exists():
            raise ModelNotReadyError(
                "vlm weights are not available",
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
            )
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoProcessor  # type: ignore[import-not-found]
        except Exception as exc:
            raise ModelNotReadyError(
                "vlm runtime dependencies are not available",
                cause=exc,
                model_id=self._settings.model_id,
                backend_id=self._settings.backend_id,
            ) from exc

        try:
            processor = AutoProcessor.from_pretrained(str(weights), trust_remote_code=True)
            model_cls = self._resolve_model_class()
            kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "device_map": "auto" if self._device == "cuda" else None,
            }
            quant = self._settings.quantization
            if quant in {"4bit", "8bit"}:
                from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quant == "4bit",
                    load_in_8bit=quant == "8bit",
                )
            elif self._device == "cpu":
                kwargs["torch_dtype"] = torch.float32
            else:
                kwargs["torch_dtype"] = torch.bfloat16
            model = model_cls.from_pretrained(str(weights), **{k: v for k, v in kwargs.items() if v is not None})
            if self._device == "cpu":
                model = model.to("cpu")
            self._processor = processor
            self._model = model
        except LLMError:
            raise
        except Exception as exc:
            raise ModelNotReadyError(
                "vlm model failed to load",
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

    def _resolve_model_class(self) -> Any:
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore[import-not-found]

            return AutoModelForImageTextToText
        except Exception:
            from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

            return AutoModelForCausalLM

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
        pieces = list(self.stream(request, max_new_tokens=max_new_tokens, modality=modality, trace_id=trace_id))
        text = "".join(delta for delta, _, _ in pieces)
        last = pieces[-1] if pieces else ("", True, "stop")
        finish: FinishReason = last[2] if last[2] in {"stop", "length"} else "stop"
        return BackendOutput(text=text, finish_reason=finish, output_token_count=max(1, len(text.split())))

    def stream(
        self,
        request: LLMRequest,
        *,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> Iterator[tuple[str, bool, FinishReason | None]]:
        if not self._loaded or self._model is None or self._processor is None:
            raise ModelNotReadyError("vlm backend is not loaded")
        try:
            text = self._generate_blocking(request, max_new_tokens=max_new_tokens)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMBackendError(cause=exc) from exc
        if not text:
            yield "", True, "stop"
            return
        yield text, True, "stop"

    def _processor_inputs(self, request: LLMRequest) -> tuple[list[dict[str, Any]], list[Any] | None]:
        pil_images: list[Any] | None = None
        if request.images:
            try:
                from PIL import Image  # type: ignore[import-not-found]
            except Exception as exc:
                raise LLMBackendError(cause=exc) from exc
            pil_images = [Image.open(BytesIO(part.data)).convert("RGB") for part in request.images]

        conversation: list[dict[str, Any]] = []
        attached = False
        for message in request.messages:
            if pil_images and message.role == "user" and not attached:
                content: list[dict[str, Any]] = [{"type": "image"} for _ in pil_images]
                if message.content:
                    content.append({"type": "text", "text": message.content})
                conversation.append({"role": message.role, "content": content})
                attached = True
            else:
                conversation.append({"role": message.role, "content": message.content})
        return conversation, pil_images

    def _generate_blocking(self, request: LLMRequest, *, max_new_tokens: int) -> str:
        import torch  # type: ignore[import-not-found]

        conversation, pil_images = self._processor_inputs(request)
        processor = self._processor
        model = self._model
        prompt_text: Any = conversation
        if hasattr(processor, "apply_chat_template"):
            try:
                prompt_text = processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = conversation
        kwargs: dict[str, Any] = {"text": prompt_text, "return_tensors": "pt"}
        if pil_images:
            kwargs["images"] = pil_images
        inputs = processor(**kwargs)
        if self._device == "cuda":
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        return decoded[0] if decoded else ""


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class LLMRuntime:
    """Owns lifecycle, semaphore, timeout, and structural validation.

    Contract: startup, generate, stream, shutdown. One semaphore per instance.
    Timeout covers semaphore wait through the last generated chunk.
    """

    def __init__(
        self,
        settings: LLMSettings | None = None,
        backend: IVLMBackend | None = None,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._settings = _validate_settings(settings or get_llm_settings())
        self._backend = backend if backend is not None else Qwen3VLBackend(self._settings)
        if executor is not None:
            self._executor = executor
            self._owns_executor = False
        else:
            workers = max(self._settings.executor_max_workers, self._settings.max_concurrent)
            self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rag-llm")
            self._owns_executor = True
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent)
        self._load_lock = asyncio.Lock()
        self._ready = False
        self._startup_called = False

    @property
    def settings(self) -> LLMSettings:
        return self._settings

    @property
    def backend(self) -> IVLMBackend:
        return self._backend

    @property
    def is_ready(self) -> bool:
        return self._ready and self._backend.is_loaded

    @property
    def model_fingerprint(self) -> str:
        if not self.is_ready:
            raise ModelNotReadyError("model fingerprint is unavailable before load")
        return self._backend.model_fingerprint

    async def startup(self) -> None:
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
        try:
            async with self._load_lock:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        self._executor, self._backend.shutdown
                    )
                except LLMError:
                    raise
                except Exception as exc:
                    raise LLMBackendError(cause=exc) from exc
                finally:
                    self._ready = False
        finally:
            if self._owns_executor:
                self._executor.shutdown(wait=False)
                self._owns_executor = False

    async def generate(self, request: LLMRequest | Mapping[str, Any] | Any) -> GenerationResult:
        prepared, modality, max_new, trace_id = prepare_request(request, self._settings)
        token = _trace_id_var.set(trace_id)
        t0 = time.perf_counter()
        diag = _safe_diag(
            trace_id=trace_id,
            modality=modality,
            model_id=self._settings.model_id,
            backend_id=self._settings.backend_id,
            timeout_seconds=self._settings.timeout_seconds,
            max_new_tokens=max_new,
            image_count=len(prepared.images),
            message_count=len(prepared.messages),
        )
        try:
            await self._ensure_loaded()
            _log_event(
                "llm_generate_start",
                modality=modality,
                image_count=len(prepared.images),
                message_count=len(prepared.messages),
                max_new_tokens=max_new,
            )

            async def _body() -> BackendOutput:
                async with self._semaphore:
                    return await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        lambda: self._invoke_generate(prepared, max_new, modality, trace_id),
                    )

            try:
                output = await asyncio.wait_for(_body(), timeout=self._settings.timeout_seconds)
            except TimeoutError as exc:
                raise LLMTimeoutError("vlm inference exceeded timeout", cause=exc, **diag) from exc
            latency_ms = (time.perf_counter() - t0) * 1000.0
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
                image_count=len(prepared.images),
                message_count=len(prepared.messages),
                finish_reason=result.finish_reason,
                latency_ms=round(latency_ms, 3),
                streamed=False,
            )
            return result
        finally:
            _trace_id_var.reset(token)

    async def stream(self, request: LLMRequest | Mapping[str, Any] | Any) -> AsyncIterator[GenerationChunk]:
        prepared, modality, max_new, trace_id = prepare_request(request, self._settings)
        token = _trace_id_var.set(trace_id)
        diag = _safe_diag(
            trace_id=trace_id,
            modality=modality,
            model_id=self._settings.model_id,
            backend_id=self._settings.backend_id,
            timeout_seconds=self._settings.timeout_seconds,
            max_new_tokens=max_new,
            image_count=len(prepared.images),
            message_count=len(prepared.messages),
        )
        try:
            await self._ensure_loaded()
            _log_event(
                "llm_stream_start",
                modality=modality,
                image_count=len(prepared.images),
                message_count=len(prepared.messages),
                max_new_tokens=max_new,
            )
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Any] = asyncio.Queue()
            deadline = loop.time() + self._settings.timeout_seconds

            def worker() -> None:
                try:
                    for delta, is_last, reason in self._invoke_stream(prepared, max_new, modality, trace_id):
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("chunk", delta, is_last, reason)),
                            loop,
                        ).result()
                    asyncio.run_coroutine_threadsafe(queue.put(("done",)), loop).result()
                except Exception as exc:  # mapped below; never returns empty success
                    asyncio.run_coroutine_threadsafe(queue.put(("err", exc)), loop).result()

            try:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise LLMTimeoutError("vlm inference exceeded timeout", **diag)
                await asyncio.wait_for(self._semaphore.acquire(), timeout=remaining)
            except TimeoutError as exc:
                raise LLMTimeoutError("vlm inference exceeded timeout", cause=exc, **diag) from exc

            worker_future = loop.run_in_executor(self._executor, worker)
            last_emitted = False
            try:
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise LLMTimeoutError("vlm inference exceeded timeout", **diag)
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except TimeoutError as exc:
                        raise LLMTimeoutError("vlm inference exceeded timeout", cause=exc, **diag) from exc
                    kind = item[0]
                    if kind == "err":
                        raise _map_backend_exception(item[1], **diag)
                    if kind == "done":
                        if not last_emitted:
                            raise LLMBackendError(**diag)
                        break
                    _, delta, is_last, reason = item
                    chunk = GenerationChunk(
                        text=delta,
                        trace_id=trace_id,
                        is_last=bool(is_last),
                        finish_reason=reason if is_last else None,
                    )
                    last_emitted = chunk.is_last
                    yield chunk
                    if chunk.is_last:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise LLMTimeoutError("vlm inference exceeded timeout", **diag)
                        try:
                            await asyncio.wait_for(queue.get(), timeout=remaining)
                        except TimeoutError as exc:
                            raise LLMTimeoutError("vlm inference exceeded timeout", cause=exc, **diag) from exc
                        break
                await asyncio.wait_for(worker_future, timeout=max(0.001, deadline - loop.time()))
            finally:
                self._semaphore.release()
            _log_event(
                "llm_stream_done",
                modality=modality,
                image_count=len(prepared.images),
                message_count=len(prepared.messages),
                streamed=True,
            )
        finally:
            _trace_id_var.reset(token)

    async def _ensure_loaded(self) -> None:
        if self._ready and self._backend.is_loaded:
            return
        if self._settings.require_model_at_startup and not self._startup_called:
            raise ModelNotReadyError("LLMRuntime.startup() was not called")
        if self._settings.require_model_at_startup and self._startup_called and not self._ready:
            raise ModelNotReadyError("model failed to load at startup")
        await self._load_backend()

    async def _load_backend(self) -> None:
        async with self._load_lock:
            if self._ready and self._backend.is_loaded:
                return
            try:
                await asyncio.get_running_loop().run_in_executor(self._executor, self._backend.load)
            except ModelNotReadyError:
                raise
            except LLMError as exc:
                raise ModelNotReadyError(
                    "vlm model failed to load",
                    cause=exc,
                    model_id=self._settings.model_id,
                    backend_id=self._settings.backend_id,
                ) from exc
            except Exception as exc:
                raise ModelNotReadyError(
                    "vlm model failed to load",
                    cause=exc,
                    model_id=self._settings.model_id,
                    backend_id=self._settings.backend_id,
                ) from exc
            if not self._backend.is_loaded:
                raise ModelNotReadyError("backend.load() returned without becoming ready")
            self._ready = True
            _log_event(
                "llm_model_ready",
                backend_id=self._settings.backend_id,
                model_id=self._settings.model_id,
                quantization=self._settings.quantization,
                fingerprint_prefix=self._backend.model_fingerprint[:12],
            )

    def _invoke_generate(
        self,
        request: LLMRequest,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> BackendOutput:
        if not self._backend.is_loaded:
            raise ModelNotReadyError("vlm backend is not loaded")
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
            raise LLMBackendError(cause=exc) from exc

    def _invoke_stream(
        self,
        request: LLMRequest,
        max_new_tokens: int,
        modality: Modality,
        trace_id: str,
    ) -> Iterator[tuple[str, bool, FinishReason | None]]:
        if not self._backend.is_loaded:
            raise ModelNotReadyError("vlm backend is not loaded")
        try:
            yield from self._backend.stream(
                request,
                max_new_tokens=max_new_tokens,
                modality=modality,
                trace_id=trace_id,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMBackendError(cause=exc) from exc


def build_llm_runtime(
    settings: LLMSettings | None = None,
    backend: IVLMBackend | None = None,
) -> LLMRuntime:
    return LLMRuntime(settings, backend)
