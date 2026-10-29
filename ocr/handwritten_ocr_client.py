"""Local Arabic-English handwritten OCR. Loads one cached checkpoint (GPU when available)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ocr.config import default_cache_dir, inference_context, load_dotenv, resolve_device, resolve_dtype
from ocr.confidence import TokenStats, make_recorder
from ocr.preprocessing import Variant
from ocr.tiling import needs_tiling, plan_bands

MODEL_ID = "sherif1313/Arabic-English-handwritten-OCR-v3"
_CHECKPOINT_DIRNAME = "sherif1313__Arabic-English-handwritten-OCR-v3"
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)

# Visible-text transcription only. The model is not asked to interpret the Rx.
TRANSCRIPTION_PROMPT = (
    "You are an OCR transcriber for one medical-prescription image. "
    "Copy only text that is visibly written in the image. "
    "Keep Arabic and English, mixed-language lines, numbers, dates, names, "
    "medical terms, dosages, punctuation, and the original reading order. "
    "Do not translate. Do not infer, complete, normalize, or invent words. "
    "Do not guess illegible handwriting. "
    "If a word, number, or region cannot be read, write [illegible] for that "
    "span and continue. "
    "Do not repeat lines or fragments unless they are written more than once. "
    "Do not add headings, labels, markdown, commentary, diagnoses, or extra "
    "blank lines. Reply with the transcription only."
)

# General documents: notes, forms, letters, exam papers. Less cautious than the
# prescription prompt: it asks for a best reading of every word instead of
# giving up on a whole line.
GENERAL_PROMPT = (
    "You are an OCR transcriber. Copy all text that is visibly written or printed "
    "in the image, in natural reading order. Arabic is written right to left; keep "
    "Arabic and English, mixed-language lines, numbers, dates, names, and punctuation "
    "in their original script. Do not translate. Do not summarize, explain, or add "
    "words that are not on the page. Give your best reading of every word; write "
    "[illegible] only for a single word that cannot be read at all, and keep "
    "transcribing the rest of the line. Do not repeat lines or fragments unless they "
    "are written more than once. Do not add headings, labels, markdown, commentary, "
    "or extra blank lines. Reply with the transcription only."
)

PROMPTS = {"general": GENERAL_PROMPT, "prescription": TRANSCRIPTION_PROMPT}


def prompt_for(mode: str, custom: str | None = None) -> str:
    if custom and custom.strip():
        return custom.strip()
    try:
        return PROMPTS[mode]
    except KeyError:
        raise ValueError(f"prompt mode must be one of {sorted(PROMPTS)}, got {mode!r}") from None


# Heuristic flags only. They do not rewrite text or change operational status.
_LONG_CHAR_LIMIT = 3500
_REPEAT_NGRAM = 6
_REPEAT_NGRAM_COUNT = 4


class LocalModelError(RuntimeError):
    """Raised when the local checkpoint is missing, incomplete, or unloadable."""


@dataclass
class HandwrittenOcrConfig:
    model: str = MODEL_ID
    cache_dir: Path = field(default_factory=default_cache_dir)
    # "auto" picks cuda, then mps, then cpu. dtype "auto" is bfloat16/float16 on
    # CUDA and float32 elsewhere. Override with HANDWRITTEN_OCR_DEVICE / _DTYPE.
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 512
    # Modest penalty from the checkpoint card. No n-gram ban: repeated doses
    # and instructions can be genuine. Greedy decoding stays deterministic.
    repetition_penalty: float = 1.05
    # Longest side of the tensor sent to the vision encoder. Source files and
    # saved variant PNGs are unchanged. Qwen2.5-VL cost scales with pixels.
    max_image_side: int = 1792
    # Pages taller than max_image_side at that scale are sent as several bands cut at
    # blank rows, so handwriting is not shrunk to fit one image.
    tile_pages: bool = True
    # "general" (any document) or "prescription" (conservative, medical wording).
    prompt_mode: str = "general"
    prompt_text: str | None = None
    # Local file URI only. Present so reports keep a base_url field without a host.
    base_url: str = ""

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        if not self.base_url:
            self.base_url = (self.cache_dir / _CHECKPOINT_DIRNAME).as_uri()

    @classmethod
    def from_env(cls) -> "HandwrittenOcrConfig":
        load_dotenv()
        model = os.environ.get("HANDWRITTEN_OCR_MODEL", MODEL_ID).strip()
        if model != MODEL_ID:
            raise LocalModelError(
                f"This client only loads {MODEL_ID}. Refusing {model}."
            )
        cache_dir = default_cache_dir()
        max_new_tokens = int(os.environ.get("HANDWRITTEN_OCR_MAX_NEW_TOKENS", "512"))
        repetition_penalty = float(
            os.environ.get("HANDWRITTEN_OCR_REPETITION_PENALTY", "1.05")
        )
        max_image_side = int(
            os.environ.get("HANDWRITTEN_OCR_MAX_IMAGE_SIDE", "1792")
        )
        tile_pages = os.environ.get("HANDWRITTEN_OCR_TILE_PAGES", "1").strip().lower() not in {"0", "false", "no", "off"}
        prompt_mode = os.environ.get("HANDWRITTEN_OCR_PROMPT_MODE", "general").strip().lower() or "general"
        prompt_for(prompt_mode)  # fail early on a typo
        checkpoint = cache_dir / _CHECKPOINT_DIRNAME
        return cls(
            model=model,
            cache_dir=cache_dir,
            device=os.environ.get("HANDWRITTEN_OCR_DEVICE", "auto").strip() or "auto",
            dtype=os.environ.get("HANDWRITTEN_OCR_DTYPE", "auto").strip() or "auto",
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            max_image_side=max_image_side,
            tile_pages=tile_pages,
            prompt_mode=prompt_mode,
            base_url=checkpoint.as_uri(),
        )


@dataclass
class Transcription:
    text: str | None
    model: str
    status: str
    preprocessing_method: str
    latency_ms: float
    error: str | None = None
    served_model: str | None = None
    raw_text: str | None = None
    quality_warnings: list[str] = field(default_factory=list)
    # Geometric-mean probability of the decoded tokens. A ranking signal for review,
    # not an accuracy. None when it could not be measured.
    confidence: float | None = None
    min_token_prob: float | None = None
    low_confidence_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "model": self.model,
            "status": self.status,
            "preprocessing_method": self.preprocessing_method,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }
        if self.raw_text is not None:
            payload["raw_text"] = self.raw_text
        if self.quality_warnings:
            payload["quality_warnings"] = list(self.quality_warnings)
        if self.confidence is not None:
            payload["confidence"] = round(self.confidence, 4)
            payload["min_token_prob"] = round(self.min_token_prob, 4) if self.min_token_prob is not None else None
            payload["low_confidence_fraction"] = (
                round(self.low_confidence_fraction, 4) if self.low_confidence_fraction is not None else None
            )
        return payload


class HandwrittenOcrClient:
    """Loads `sherif1313/Arabic-English-handwritten-OCR-v3` once, from disk."""

    def __init__(
        self,
        config: HandwrittenOcrConfig | None = None,
        loader: Callable[[Path], tuple[Any, Any]] | None = None,
    ) -> None:
        self.config = config or HandwrittenOcrConfig()
        if self.config.model != MODEL_ID:
            raise LocalModelError(
                f"This client only loads {MODEL_ID}. "
                f"Refusing {self.config.model}."
            )
        self._loader = loader or self._default_loader
        self._model = None
        self._processor = None
        self.device = "cpu"
        self._last_stats = TokenStats()

    def transcribe(
        self,
        image: Image.Image | np.ndarray | Variant,
        preprocessing_method: str,
    ) -> Transcription:
        started = time.perf_counter()
        notes: list[str] = []
        stats = TokenStats()
        try:
            self.ensure_ready()
            text = self._generate_page(_as_pil(image), notes, stats)
        except Exception as exc:
            return Transcription(
                text=None,
                model=self.config.model,
                status="error",
                preprocessing_method=preprocessing_method,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=_public_error(exc),
            )
        raw = text
        warnings = notes + diagnose_transcription(raw)
        return Transcription(
            text=raw,
            raw_text=raw,
            model=self.config.model,
            status="success",
            preprocessing_method=preprocessing_method,
            latency_ms=(time.perf_counter() - started) * 1000,
            quality_warnings=warnings,
            confidence=stats.confidence,
            min_token_prob=stats.min_token_prob,
            low_confidence_fraction=stats.low_fraction,
        )

    def ensure_ready(self) -> None:
        if self._model is not None:
            return
        path = self.checkpoint_dir()
        missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            listed = ", ".join(missing)
            raise LocalModelError(
                f"Local checkpoint is incomplete at {path}. Missing: {listed}. "
                "Place the Hugging Face files there. This client does not "
                "download weights and does not call an external service."
            )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(self.config.cache_dir)
        self._model, self._processor = self._loader(path)
        self._apply_greedy_generation_config()

    def _default_loader(self, path: Path):
        self.device = resolve_device(self.config.device)
        dtype = resolve_dtype(self.config.dtype, self.device)
        return _load_from_disk(path, self.device, dtype)

    def checkpoint_dir(self) -> Path:
        return Path(self.config.cache_dir) / _CHECKPOINT_DIRNAME

    def _apply_greedy_generation_config(self) -> None:
        """Keep greedy decoding. Neutralize sampling fields on the checkpoint.

        generation_config.json ships `do_sample: true` and `temperature: 1e-06`.
        Passing `do_sample=False` while those fields remain set makes
        transformers warn that temperature is ignored. Temperature 1.0 is the
        greedy default and is unused when sampling is off.
        """
        model = self._model
        gen = getattr(model, "generation_config", None)
        if gen is None:
            return
        gen.do_sample = False
        if hasattr(gen, "temperature"):
            gen.temperature = 1.0
        if hasattr(gen, "top_p"):
            gen.top_p = None
        if hasattr(gen, "top_k"):
            gen.top_k = None

    def _generate_page(self, image: Image.Image, notes: list[str], stats: TokenStats | None = None) -> str:
        """Run one image, or several bands when the page is too large for one call."""
        width, height = image.size
        if self.config.tile_pages and needs_tiling(width, height, self.config.max_image_side):
            bands = plan_bands(np.asarray(image.convert("L")), self.config.max_image_side)
            if len(bands) > 1:
                notes.append(f"large page split into {len(bands)} bands at blank rows")
                parts = []
                for top, bottom in bands:
                    parts.append(self._generate(image.crop((0, top, width, bottom))))
                    if stats is not None:
                        stats.merge(self._last_stats)
                return "\n".join(part for part in parts if part)
        text = self._generate(image)
        if stats is not None:
            stats.merge(self._last_stats)
        return text

    def _generate(self, image: Image.Image) -> str:
        model = self._model
        processor = self._processor
        prepared = _limit_image_side(image, self.config.max_image_side)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": prepared},
                    {"type": "text", "text": prompt_for(self.config.prompt_mode, self.config.prompt_text)},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt],
            images=[prepared],
            padding=True,
            return_tensors="pt",
        )
        target = getattr(model, "device", None) or self.device
        inputs = {
            key: value.to(target) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        recorder, processors = make_recorder()
        extra = {"logits_processor": processors} if processors is not None else {}
        self._last_stats = TokenStats()
        with inference_context():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                num_return_sequences=1,
                **extra,
            )
        if recorder is not None:
            self._last_stats = recorder.stats()
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


def diagnose_transcription(text: str | None) -> list[str]:
    """Flag suspicious shape. Does not edit the transcription or prove errors."""
    if text is None or not str(text).strip():
        return ["empty transcription"]

    warnings: list[str] = []
    body = str(text)
    if len(body) > _LONG_CHAR_LIMIT:
        warnings.append(
            f"unusually long transcription ({len(body)} characters)"
        )

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    consecutive = 0
    for current, following in zip(lines, lines[1:]):
        if current == following:
            consecutive += 1
    if consecutive >= 2:
        warnings.append("consecutive duplicate lines")

    tokens = body.split()
    n = _REPEAT_NGRAM
    if len(tokens) >= n * _REPEAT_NGRAM_COUNT:
        counts: dict[tuple[str, ...], int] = {}
        for index in range(len(tokens) - n + 1):
            gram = tuple(tokens[index : index + n])
            counts[gram] = counts.get(gram, 0) + 1
        peak = max(counts.values()) if counts else 0
        if peak >= _REPEAT_NGRAM_COUNT:
            warnings.append(
                f"repeated {_REPEAT_NGRAM}-gram appears {peak} times"
            )

    return warnings


def _limit_image_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side < 32:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


def _as_pil(image: Image.Image | np.ndarray | Variant) -> Image.Image:
    if isinstance(image, Variant):
        image = image.image
    if isinstance(image, Image.Image):
        if image.mode == "RGB":
            return image
        if image.mode == "L":
            return image.convert("RGB")
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    return Image.fromarray(array).convert("RGB")


def _load_from_disk(path: Path, device: str = "cpu", dtype: str = "float32"):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    return model, processor


def _public_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
