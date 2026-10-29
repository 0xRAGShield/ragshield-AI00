"""Local English handwritten TrOCR. Loads one cached checkpoint on CPU."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ocr.config import TROCR_MODEL_ID, default_trocr_checkpoint_dir, inference_context, load_dotenv, resolve_device
from ocr.crops import CropSpec

# Files that a local microsoft/trocr-base-handwritten dump normally includes.
_CONFIG_FILES = ("config.json", "preprocessor_config.json")
_TOKENIZER_GROUPS = (
    ("tokenizer.json",),
    ("vocab.json", "merges.txt"),
    ("tokenizer.model",),
)
_WEIGHT_CANDIDATES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
)

STATUS_NOT_CONFIGURED = "not_configured"
STATUS_CHECKPOINT_MISSING = "checkpoint_missing"
STATUS_CHECKPOINT_INCOMPLETE = "checkpoint_incomplete"
STATUS_DEPENDENCIES_MISSING = "dependencies_missing"
STATUS_NOT_LOADED = "not_loaded"
STATUS_CROPS_MISSING = "crops_missing"
STATUS_CROPS_PATH_NOT_FOUND = "crops_path_not_found"
STATUS_CROPS_UNSUPPORTED = "crops_unsupported"
STATUS_CROPS_JSON_MALFORMED = "crops_json_malformed"
STATUS_CROPS_EMPTY = "crops_empty"
STATUS_CROP_IMAGE_MISSING = "crop_image_missing"
STATUS_CROP_BOX_INVALID = "crop_box_invalid"
STATUS_CROP_ENTRY_INVALID = "crop_entry_invalid"
STATUS_SKIPPED_ARABIC = "skipped_arabic"
STATUS_INFERENCE_FAILURE = "inference_failure"
STATUS_SUCCESS = "success"


class TrocrLocalError(RuntimeError):
    """Raised when the local TrOCR checkpoint cannot be used."""


@dataclass
class TrocrConfig:
    model: str = TROCR_MODEL_ID
    checkpoint_dir: Path = field(default_factory=default_trocr_checkpoint_dir)
    device: str = "cpu"
    max_new_tokens: int = 64
    # Separate from line-crop decoding. A page can be longer than one IAM line.
    full_page_max_new_tokens: int = 256
    configured: bool = True

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)

    @classmethod
    def from_env(cls) -> "TrocrConfig":
        load_dotenv()
        raw = os.environ.get("TROCR_CHECKPOINT_DIR")
        if raw is not None and not raw.strip():
            return cls(configured=False, checkpoint_dir=default_trocr_checkpoint_dir())
        checkpoint_dir = Path(raw.strip()) if raw and raw.strip() else default_trocr_checkpoint_dir()
        device = os.environ.get("TROCR_DEVICE", os.environ.get("HANDWRITTEN_OCR_DEVICE", "cpu")).strip() or "cpu"
        max_new_tokens = int(os.environ.get("TROCR_MAX_NEW_TOKENS", "64"))
        full_page_max_new_tokens = int(
            os.environ.get("TROCR_FULL_PAGE_MAX_NEW_TOKENS", "256")
        )
        model = os.environ.get("TROCR_MODEL_ID", TROCR_MODEL_ID).strip() or TROCR_MODEL_ID
        return cls(
            model=model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            max_new_tokens=max_new_tokens,
            full_page_max_new_tokens=full_page_max_new_tokens,
            configured=True,
        )


@dataclass
class TrocrCropResult:
    crop_id: str
    status: str
    text: str | None = None
    error: str | None = None
    latency_ms: float = 0.0
    box: list[int] | None = None
    language: str = "en"
    source: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "crop_id": self.crop_id,
            "status": self.status,
            "text": self.text,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "box": self.box,
            "language": self.language,
            "source": self.source,
        }
        if self.path:
            payload["path"] = self.path
        return payload


FULL_PAGE_WARNINGS = (
    "microsoft/trocr-base-handwritten is trained mainly on English line-level "
    "handwriting (IAM). A full prescription is accepted as one image. The ViT "
    "processor resizes that image to a fixed square (commonly 384x384 for this "
    "checkpoint), which can make small handwriting unreadable.",
    "This checkpoint is English-oriented. Arabic and mixed Arabic-English "
    "prescription text may be unreliable. Text is not translated or completed.",
    "status success means generate returned a string. It is not transcription "
    "accuracy and is not a confidence score.",
)


@dataclass
class TrocrFullPageResult:
    source: str
    model: str
    checkpoint_dir: str
    status: str
    text: str | None = None
    error: str | None = None
    latency_ms: float = 0.0
    image_width: int | None = None
    image_height: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "checkpoint_dir": self.checkpoint_dir,
            "status": self.status,
            "text": self.text,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "warnings": list(self.warnings),
        }


@dataclass
class TrocrBatchResult:
    model: str
    checkpoint_dir: str
    status: str
    error: str | None = None
    crops: list[TrocrCropResult] = field(default_factory=list)
    joined_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "checkpoint_dir": self.checkpoint_dir,
            "status": self.status,
            "error": self.error,
            "joined_text": self.joined_text,
            "crops": [item.to_dict() for item in self.crops],
        }


class TrocrClient:
    """Loads `microsoft/trocr-base-handwritten` from a local directory only."""

    def __init__(
        self,
        config: TrocrConfig | None = None,
        loader: Callable[[Path], tuple[Any, Any]] | None = None,
    ) -> None:
        self.config = config or TrocrConfig()
        self._loader = loader or self._default_loader
        self._model = None
        self._processor = None
        self.device = "cpu"

    def _default_loader(self, path: Path):
        self.device = resolve_device(self.config.device)
        return _load_from_disk(path, self.device)

    def inspect(self) -> TrocrBatchResult:
        """Describe readiness without loading weights."""
        if not self.config.configured:
            return TrocrBatchResult(
                model=self.config.model,
                checkpoint_dir=str(self.config.checkpoint_dir),
                status=STATUS_NOT_CONFIGURED,
                error="TROCR_CHECKPOINT_DIR is empty. TrOCR is not configured.",
            )
        path = self.config.checkpoint_dir
        if not path.is_dir():
            return TrocrBatchResult(
                model=self.config.model,
                checkpoint_dir=str(path),
                status=STATUS_CHECKPOINT_MISSING,
                error=(
                    f"Local TrOCR checkpoint directory is missing: {path}. "
                    "Place microsoft/trocr-base-handwritten files there. "
                    "This client does not download weights."
                ),
            )
        missing = missing_checkpoint_files(path)
        if missing:
            return TrocrBatchResult(
                model=self.config.model,
                checkpoint_dir=str(path),
                status=STATUS_CHECKPOINT_INCOMPLETE,
                error=(
                    f"Local TrOCR checkpoint is incomplete at {path}. "
                    f"Missing: {', '.join(missing)}. "
                    "A directory on disk is not enough; required files must be present."
                ),
            )
        dependency_error = _dependency_error()
        if dependency_error:
            return TrocrBatchResult(
                model=self.config.model,
                checkpoint_dir=str(path),
                status=STATUS_DEPENDENCIES_MISSING,
                error=dependency_error,
            )
        if self._model is None:
            return TrocrBatchResult(
                model=self.config.model,
                checkpoint_dir=str(path),
                status=STATUS_NOT_LOADED,
                error=None,
            )
        return TrocrBatchResult(
            model=self.config.model,
            checkpoint_dir=str(path),
            status=STATUS_SUCCESS,
        )

    def ensure_ready(self) -> None:
        state = self.inspect()
        if state.status == STATUS_NOT_LOADED:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self._model, self._processor = self._loader(self.config.checkpoint_dir)
            return
        if state.status == STATUS_SUCCESS:
            return
        raise TrocrLocalError(state.error or state.status)

    def transcribe_line(
        self,
        image: Image.Image,
        crop_id: str,
        source: str,
        box: tuple[int, int, int, int] | None = None,
    ) -> TrocrCropResult:
        """Recognize one line crop. Pads to a square; does not stretch the line."""
        ready = self.inspect()
        if ready.status not in {STATUS_NOT_LOADED, STATUS_SUCCESS}:
            return TrocrCropResult(
                crop_id=crop_id,
                status=ready.status,
                error=ready.error,
                box=list(box) if box else None,
                language="en",
                source=source,
            )
        try:
            self.ensure_ready()
        except Exception as exc:
            return TrocrCropResult(
                crop_id=crop_id,
                status=STATUS_INFERENCE_FAILURE,
                error=_public_error(exc),
                box=list(box) if box else None,
                language="en",
                source=source,
            )
        padded = pad_line_keep_aspect(image)
        spec = CropSpec(
            crop_id=crop_id,
            image=padded,
            source=source,
            box=box,
            language="en",
        )
        return self._transcribe_one(spec)

    def transcribe_full_page(
        self,
        image: Image.Image,
        source: str,
    ) -> TrocrFullPageResult:
        """Recognize one complete page. Does not use crop loading or a detector."""
        width, height = image.size
        ready = self.inspect()
        if ready.status not in {STATUS_NOT_LOADED, STATUS_SUCCESS}:
            return TrocrFullPageResult(
                source=source,
                model=self.config.model,
                checkpoint_dir=str(self.config.checkpoint_dir),
                status=ready.status,
                error=ready.error,
                image_width=width,
                image_height=height,
                warnings=list(FULL_PAGE_WARNINGS),
            )
        started = time.perf_counter()
        try:
            self.ensure_ready()
            text = self._generate(
                _as_rgb(image),
                max_new_tokens=self.config.full_page_max_new_tokens,
            )
        except Exception as exc:
            return TrocrFullPageResult(
                source=source,
                model=self.config.model,
                checkpoint_dir=str(self.config.checkpoint_dir),
                status=STATUS_INFERENCE_FAILURE,
                error=_public_error(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
                image_width=width,
                image_height=height,
                warnings=list(FULL_PAGE_WARNINGS),
            )
        return TrocrFullPageResult(
            source=source,
            model=self.config.model,
            checkpoint_dir=str(self.config.checkpoint_dir),
            status=STATUS_SUCCESS,
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000,
            image_width=width,
            image_height=height,
            warnings=list(FULL_PAGE_WARNINGS),
        )

    def transcribe_crops(self, crops: list[CropSpec]) -> TrocrBatchResult:
        if not crops:
            result = self.inspect()
            result.status = STATUS_CROPS_MISSING
            result.error = (
                "TrOCR needs line or word crops. This project has no text detector. "
                "Supply --crops as a JSON sidecar, a crop image, or a directory of crop images. "
                "Do not send a full prescription page as if it were a single text line."
            )
            return result

        ready = self.inspect()
        if ready.status not in {STATUS_NOT_LOADED, STATUS_SUCCESS}:
            ready.crops = [
                TrocrCropResult(
                    crop_id=crop.crop_id,
                    status=ready.status,
                    error=ready.error,
                    box=list(crop.box) if crop.box else None,
                    language=crop.language,
                    source=crop.source,
                    path=crop.path,
                )
                for crop in crops
            ]
            return ready

        try:
            self.ensure_ready()
        except Exception as exc:
            failure = self.inspect()
            failure.status = STATUS_INFERENCE_FAILURE if failure.status in {
                STATUS_NOT_LOADED,
                STATUS_SUCCESS,
            } else failure.status
            failure.error = _public_error(exc)
            return failure

        results: list[TrocrCropResult] = []
        for crop in crops:
            results.append(self._transcribe_one(crop))

        successful = [item.text for item in results if item.status == STATUS_SUCCESS and item.text]
        any_success = any(item.status == STATUS_SUCCESS for item in results)
        any_failure = any(
            item.status == STATUS_INFERENCE_FAILURE for item in results
        )
        skipped = all(
            item.status == STATUS_SKIPPED_ARABIC for item in results
        ) and results
        if skipped:
            status = STATUS_SKIPPED_ARABIC
            error = "Every supplied crop is marked Arabic. English TrOCR was not applied."
        elif any_success and not any_failure:
            status = STATUS_SUCCESS
            error = None
        elif any_success:
            status = STATUS_INFERENCE_FAILURE
            error = "Some English crops failed. Successful crop texts are kept separately."
        else:
            status = STATUS_INFERENCE_FAILURE
            error = "TrOCR did not produce text for any English crop."

        return TrocrBatchResult(
            model=self.config.model,
            checkpoint_dir=str(self.config.checkpoint_dir),
            status=status,
            error=error,
            crops=results,
            joined_text="\n".join(successful) if successful else None,
        )

    def _transcribe_one(self, crop: CropSpec) -> TrocrCropResult:
        language = (crop.language or "en").lower()
        box = list(crop.box) if crop.box else None
        if language.startswith("ar"):
            return TrocrCropResult(
                crop_id=crop.crop_id,
                status=STATUS_SKIPPED_ARABIC,
                error=(
                    "Skipped: microsoft/trocr-base-handwritten is English-oriented. "
                    "Arabic crops are not sent to this checkpoint."
                ),
                box=box,
                language=crop.language,
                source=crop.source,
                path=crop.path,
            )
        started = time.perf_counter()
        try:
            text = self._generate(_as_rgb(crop.image))
        except Exception as exc:
            return TrocrCropResult(
                crop_id=crop.crop_id,
                status=STATUS_INFERENCE_FAILURE,
                error=_public_error(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
                box=box,
                language=crop.language,
                source=crop.source,
                path=crop.path,
            )
        return TrocrCropResult(
            crop_id=crop.crop_id,
            status=STATUS_SUCCESS,
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000,
            box=box,
            language=crop.language,
            source=crop.source,
            path=crop.path,
        )

    def _generate(
        self,
        image: Image.Image,
        max_new_tokens: int | None = None,
    ) -> str:
        processor = self._processor
        model = self._model
        encoded = processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"]
        if hasattr(pixel_values, "to"):
            pixel_values = pixel_values.to(self.device)
        tokens = (
            self.config.max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )
        with inference_context():
            generated = model.generate(
                pixel_values,
                max_new_tokens=tokens,
            )
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        return decoded[0].strip()


def missing_checkpoint_files(directory: Path) -> list[str]:
    missing = [name for name in _CONFIG_FILES if not (directory / name).is_file()]
    if not any(all((directory / name).is_file() for name in group) for group in _TOKENIZER_GROUPS):
        missing.append("tokenizer.json or vocab.json+merges.txt")
    if not any((directory / name).is_file() for name in _WEIGHT_CANDIDATES):
        missing.append("model.safetensors or pytorch_model.bin")
    return missing


def _dependency_error() -> str | None:
    try:
        import torch  # noqa: F401
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: F401
    except ImportError as exc:
        return (
            f"dependencies_missing: {exc}. "
            "Install the existing local CPU stack from requirements-local.txt. "
            "This client does not install packages."
        )
    return None


def _load_from_disk(path: Path, device: str = "cpu"):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(str(path), local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(
        str(path),
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, processor


def pad_line_keep_aspect(image: Image.Image, fill: int = 255) -> Image.Image:
    """Pad a line crop to a square so ViT resize does not stretch handwriting."""
    rgb = _as_rgb(image)
    width, height = rgb.size
    side = max(width, height, 1)
    canvas = Image.new("RGB", (side, side), (fill, fill, fill))
    canvas.paste(rgb, ((side - width) // 2, (side - height) // 2))
    return canvas


def _as_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _public_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
