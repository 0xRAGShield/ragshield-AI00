from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.sensitive.labels import (
    DEFAULT_PRETRAINED_MODEL,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
)
from app.sensitive.offsets import TokenAnnotation, is_special_or_empty_offset

logger = logging.getLogger("ragshield.sensitive.model")

_WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")
_TOKENIZER_FILENAMES = ("tokenizer.json", "tokenizer_config.json")


@dataclass(frozen=True)
class ModelArtifacts:
    """Local files required for offline token-classification inference."""

    root_dir: Path
    weights_dir: Path
    tokenizer_dir: Path
    config_dir: Path


class SensitiveDataModelError(RuntimeError):
    """Raised when the NER backend cannot load or run inference."""


@dataclass(frozen=True)
class RawEntityPrediction:
    text: str
    label: str
    start: int
    end: int
    score: float

    @property
    def entity(self) -> str:
        return self.label

    @property
    def entity_type(self) -> str:
        return self.label

    @property
    def confidence(self) -> float:
        return self.score


class INERModel(Protocol):
    """Protocol for swappable NER backends (pretrained or fine-tuned)."""

    @property
    def model_name(self) -> str:
        ...

    def predict(self, text: str) -> list[RawEntityPrediction]:
        ...


@dataclass(frozen=True)
class SensitiveDataModelConfig:
    model_name_or_path: str = DEFAULT_PRETRAINED_MODEL
    device: str = "auto"
    confidence_threshold: float = 0.5
    max_length: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.model_name_or_path, str) or not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a non-empty string.")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        if not isinstance(self.max_length, int) or self.max_length <= 0:
            raise ValueError("max_length must be a positive integer.")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise SensitiveDataModelError(
            "CUDA was requested but is not available on this machine."
        )
    return resolved


def resolve_local_model_path(model_name_or_path: str) -> Path:
    candidate = Path(model_name_or_path)
    if candidate.is_dir():
        return _prefer_weights_directory(candidate.resolve())

    project_relative = Path(__file__).resolve().parents[3] / model_name_or_path
    if project_relative.is_dir():
        return _prefer_weights_directory(project_relative.resolve())

    raise SensitiveDataModelError(
        "Fine-tuned NER model directory was not found at "
        f"'{model_name_or_path}'. Place the local bilingual PII checkpoint "
        "under models/sensitive/ and do not download the base model."
    )


def _has_weight_file(directory: Path) -> bool:
    return any((directory / name).is_file() for name in _WEIGHT_FILENAMES)


def _prefer_weights_directory(directory: Path) -> Path:
    if _has_weight_file(directory):
        return directory

    nested = directory / directory.name
    if nested.is_dir() and _has_weight_file(nested):
        return nested

    nested_matches = [
        child
        for child in directory.iterdir()
        if child.is_dir()
        and not child.name.startswith("checkpoint-")
        and _has_weight_file(child)
    ]
    if len(nested_matches) == 1:
        return nested_matches[0]
    return directory


def _checkpoint_sort_key(name: str) -> int:
    suffix = name.split("-")[-1]
    return int(suffix) if suffix.isdigit() else -1


def _search_roots(model_dir: Path) -> list[Path]:
    roots = [model_dir]
    if model_dir.name.startswith("checkpoint-") and model_dir.parent.is_dir():
        roots.append(model_dir.parent)
    return roots


def _checkpoint_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        [
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.startswith("checkpoint-")
        ],
        key=lambda path: _checkpoint_sort_key(path.name),
    )


def _find_weights_dir(model_dir: Path) -> Path | None:
    for root in _search_roots(model_dir):
        if _has_weight_file(root):
            return root
        checkpoints = [path for path in _checkpoint_dirs(root) if _has_weight_file(path)]
        if checkpoints:
            return checkpoints[-1]
    return None


def _has_tokenizer_files(directory: Path) -> bool:
    return any((directory / name).is_file() for name in _TOKENIZER_FILENAMES)


def _find_tokenizer_dir(model_dir: Path, *, weights_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if weights_dir is not None:
        candidates.append(weights_dir)
    for root in _search_roots(model_dir):
        candidates.append(root)
        candidates.extend(reversed(_checkpoint_dirs(root)))
    for candidate in candidates:
        if _has_tokenizer_files(candidate):
            return candidate
    return None


def locate_model_artifacts(model_dir: Path) -> ModelArtifacts:
    """
    Resolve config, tokenizer, and weights for a local checkpoint tree.

    The V3 export stores ``config.json`` at the model root, tokenizer files in
    one ``checkpoint-*`` directory, and ``model.safetensors`` in another.
    Missing pieces are reported instead of falling back to a different model.
    """
    weights_dir = _find_weights_dir(model_dir)
    tokenizer_dir = _find_tokenizer_dir(model_dir, weights_dir=weights_dir)

    missing: list[str] = []
    if weights_dir is None:
        missing.append(" or ".join(_WEIGHT_FILENAMES))
    if tokenizer_dir is None:
        missing.append("tokenizer.json or tokenizer_config.json")
    if missing:
        raise SensitiveDataModelError(
            "Fine-tuned NER model is incomplete at "
            f"'{model_dir}'. Missing: {', '.join(missing)}. "
            "Searched the model directory and checkpoint-* subdirectories. "
            "Refusing to download Davlan/xlm-roberta-base-ner-hrl or load an older checkpoint."
        )

    config_dir = find_model_config_dir(model_dir)
    return ModelArtifacts(
        root_dir=model_dir,
        weights_dir=weights_dir,
        tokenizer_dir=tokenizer_dir,
        config_dir=config_dir,
    )


def _label_maps_from_config_payload(payload: dict[str, object]) -> tuple[dict[int, str], dict[str, int]]:
    raw_id2label = payload.get("id2label") or {}
    raw_label2id = payload.get("label2id") or {}

    id2label: dict[int, str] = {}
    if isinstance(raw_id2label, dict):
        id2label = {int(index): str(label) for index, label in raw_id2label.items()}

    label2id: dict[str, int] = {}
    if isinstance(raw_label2id, dict) and raw_label2id:
        label2id = {str(label): int(index) for label, index in raw_label2id.items()}
    elif id2label:
        label2id = {label: index for index, label in id2label.items()}

    return id2label, label2id


def _config_matches_canonical(payload: dict[str, object]) -> bool:
    id2label, label2id = _label_maps_from_config_payload(payload)
    return id2label == dict(ID2LABEL) and label2id == dict(LABEL2ID)


def find_model_config_dir(model_dir: Path) -> Path:
    config_candidates: list[Path] = []
    for root in _search_roots(model_dir):
        config_candidates.append(root / "config.json")
        for checkpoint in _checkpoint_dirs(root):
            config_candidates.append(checkpoint / "config.json")

    compatible: list[Path] = []
    mismatched: list[Path] = []
    for config_path in config_candidates:
        if not config_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _config_matches_canonical(payload):
            compatible.append(config_path.parent)
        else:
            mismatched.append(config_path)

    if compatible:
        return compatible[0]

    if mismatched:
        raise SensitiveDataModelError(
            f"Model config at '{mismatched[0]}' does not match the canonical 35-label mapping."
        )

    raise SensitiveDataModelError(
        "Missing Hugging Face config.json for the fine-tuned weights at "
        f"'{model_dir}'. Expected config.json in that directory or in a "
        "checkpoint-* subdirectory whose id2label matches the canonical 35-label mapping."
    )


def _config_label_maps(model: PreTrainedModel) -> tuple[dict[int, str], dict[str, int]]:
    config_id2label = getattr(model.config, "id2label", None)
    config_label2id = getattr(model.config, "label2id", None)
    payload = {
        "id2label": config_id2label if isinstance(config_id2label, dict) else {},
        "label2id": config_label2id if isinstance(config_label2id, dict) else {},
    }
    return _label_maps_from_config_payload(payload)


def _validate_label_schema(
    *,
    model: PreTrainedModel,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> None:
    expected_count = len(ID2LABEL)
    config_num_labels = getattr(model.config, "num_labels", None)
    if config_num_labels is not None and int(config_num_labels) != expected_count:
        raise SensitiveDataModelError(
            "Fine-tuned model num_labels does not match the canonical 35-label BIO schema."
        )

    if len(id2label) != expected_count or len(label2id) != expected_count:
        raise SensitiveDataModelError(
            "Fine-tuned model label maps do not match the canonical 35-label BIO schema."
        )

    if id2label != dict(ID2LABEL) or label2id != dict(LABEL2ID):
        raise SensitiveDataModelError(
            "Fine-tuned model id2label/label2id does not match the canonical 35-label mapping."
        )


class XLMRobertaNERModel:
    def __init__(self, config: SensitiveDataModelConfig | None = None) -> None:
        self._config = config or SensitiveDataModelConfig()
        model_dir = resolve_local_model_path(self._config.model_name_or_path)
        artifacts = locate_model_artifacts(model_dir)
        self._model_name = str(artifacts.root_dir)
        self._device = resolve_device(self._config.device)
        config_dir = artifacts.config_dir

        try:
            model_config = AutoConfig.from_pretrained(
                str(artifacts.config_dir),
                local_files_only=True,
            )
            self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                str(artifacts.tokenizer_dir),
                local_files_only=True,
                use_fast=True,
            )
            self._model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
                str(artifacts.weights_dir),
                config=model_config,
                local_files_only=True,
            )
        except SensitiveDataModelError:
            raise
        except Exception as error:
            raise SensitiveDataModelError(
                "Failed to load the local fine-tuned NER model from "
                f"'{self._model_name}' (config from '{config_dir}'). "
                "Offline loading is required; remote model download is disabled."
            ) from error

        if not getattr(self._tokenizer, "is_fast", False):
            raise SensitiveDataModelError(
                "A fast tokenizer with offset mapping is required for character spans."
            )

        config_id2label, config_label2id = _config_label_maps(self._model)
        if not config_id2label:
            raise SensitiveDataModelError(
                "Fine-tuned model does not expose id2label; refusing to guess label IDs."
            )
        _validate_label_schema(
            model=self._model,
            id2label=config_id2label,
            label2id=config_label2id,
        )
        self._id2label = config_id2label

        classifier = getattr(self._model, "classifier", None)
        out_features = getattr(classifier, "out_features", None)
        if out_features is not None and int(out_features) != NUM_LABELS:
            raise SensitiveDataModelError(
                "Model classifier output size does not match the canonical BIO label schema."
            )

        self._model.to(self._device)
        self._model.eval()

        logger.info(
            "%s",
            {
                "event": "sensitive_model_loaded",
                "model_name": self._model_name,
                "device": str(self._device),
                "num_labels": len(self._id2label),
                "local_files_only": True,
            },
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def id2label(self) -> dict[int, str]:
        return dict(self._id2label)

    @property
    def confidence_threshold(self) -> float:
        return self._config.confidence_threshold

    def run_token_classification(self, text: str) -> list[TokenAnnotation]:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text:
            return []

        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True,
            max_length=self._config.max_length,
            add_special_tokens=True,
        )

        if "offset_mapping" not in encoding:
            raise SensitiveDataModelError(
                "Tokenizer did not return offset_mapping; a fast tokenizer is required."
            )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        special_tokens_mask = encoding.pop("special_tokens_mask")[0].tolist()
        model_inputs = {
            key: value.to(self._device)
            for key, value in encoding.items()
        }

        with torch.inference_mode():
            outputs = self._model(**model_inputs)

        logits = outputs.logits[0]
        if logits.shape[-1] != len(self._id2label):
            raise SensitiveDataModelError(
                "Model logits do not match the canonical BIO label schema."
            )

        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        annotations: list[TokenAnnotation] = []
        for token_index, (start, end) in enumerate(offset_mapping):
            is_special = bool(special_tokens_mask[token_index]) or is_special_or_empty_offset(
                int(start),
                int(end),
            )
            label_id = int(predictions[token_index].item())
            raw_label = self._id2label.get(label_id, "O")
            token_score = float(probabilities[token_index, label_id].item())
            annotations.append(
                TokenAnnotation(
                    token_index=token_index,
                    start=int(start),
                    end=int(end),
                    label=raw_label,
                    confidence=token_score,
                    is_special=is_special,
                )
            )

        return annotations

    def predict(self, text: str) -> list[RawEntityPrediction]:
        from app.sensitive.offsets import reconstruct_entities

        annotations = self.run_token_classification(text)
        spans = reconstruct_entities(
            text,
            annotations,
            confidence_threshold=self._config.confidence_threshold,
        )
        return [
            RawEntityPrediction(
                text=span.text,
                label=span.entity,
                start=span.start,
                end=span.end,
                score=span.confidence,
            )
            for span in spans
        ]


@lru_cache(maxsize=4)
def get_shared_ner_model(
    model_name_or_path: str,
    device: str,
    confidence_threshold: float,
    max_length: int,
) -> XLMRobertaNERModel:
    config = SensitiveDataModelConfig(
        model_name_or_path=model_name_or_path,
        device=device,
        confidence_threshold=confidence_threshold,
        max_length=max_length,
    )
    return XLMRobertaNERModel(config=config)
