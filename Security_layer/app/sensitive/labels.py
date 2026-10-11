from __future__ import annotations

import json
from pathlib import Path
from typing import Final

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_LABEL_MAPPING_PATH: Final[Path] = (
    _PROJECT_ROOT / "data" / "sensitive" / "processed" / "combined" / "label_mapping.json"
)
DEFAULT_LOCAL_MODEL_DIR: Final[str] = (
    "models/sensitive/bilingual_pii_xlmr"
)

# Exact training BIO order: 17 entity types + O = 35 labels.
CANONICAL_BIO_LABELS: Final[list[str]] = [
    "O",
    "B-NATIONAL_ID",
    "I-NATIONAL_ID",
    "B-PASSPORT_NUMBER",
    "I-PASSPORT_NUMBER",
    "B-EMPLOYEE_ID",
    "I-EMPLOYEE_ID",
    "B-PHONE",
    "I-PHONE",
    "B-EMAIL",
    "I-EMAIL",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-CREDIT_CARD",
    "I-CREDIT_CARD",
    "B-MEDICAL_RECORD_NUMBER",
    "I-MEDICAL_RECORD_NUMBER",
    "B-PRIVATE_DATE",
    "I-PRIVATE_DATE",
    "B-PASSWORD",
    "I-PASSWORD",
    "B-SPOKEN_NATIONAL_ID",
    "I-SPOKEN_NATIONAL_ID",
    "B-PERSON",
    "I-PERSON",
    "B-USERNAME",
    "I-USERNAME",
    "B-SOCIAL_SECURITY_NUMBER",
    "I-SOCIAL_SECURITY_NUMBER",
    "B-ID_CARD_NUMBER",
    "I-ID_CARD_NUMBER",
    "B-DRIVER_LICENSE",
    "I-DRIVER_LICENSE",
    "B-IP_ADDRESS",
    "I-IP_ADDRESS",
]


def _entity_labels_from_bio(bio_labels: list[str]) -> list[str]:
    entity_labels: list[str] = []
    for label in bio_labels:
        if label.startswith("B-"):
            entity_labels.append(label[2:])
    return entity_labels


def _maps_from_bio_labels(bio_labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {label: index for index, label in enumerate(bio_labels)}
    id2label = {index: label for index, label in enumerate(bio_labels)}
    return label2id, id2label


def load_canonical_label_mapping(path: Path | None = None) -> dict[str, object]:
    mapping_path = path or CANONICAL_LABEL_MAPPING_PATH
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, list) or [str(item) for item in labels] != CANONICAL_BIO_LABELS:
        raise ValueError(
            f"Canonical label mapping at '{mapping_path}' does not match the trained 35-label schema."
        )

    raw_label2id = payload.get("label2id") or {}
    raw_id2label = payload.get("id2label") or {}
    label2id = {str(label): int(index) for label, index in raw_label2id.items()}
    id2label = {int(index): str(label) for index, label in raw_id2label.items()}
    expected_label2id, expected_id2label = _maps_from_bio_labels(CANONICAL_BIO_LABELS)
    if label2id != expected_label2id or id2label != expected_id2label:
        raise ValueError(
            f"Canonical label IDs at '{mapping_path}' do not match the trained 35-label schema."
        )
    return payload


def resolve_default_model_path() -> str:
    nested = (_PROJECT_ROOT / "models" / "sensitive" / "bilingual_pii_xlmr" / "bilingual_pii_xlmr")
    parent = _PROJECT_ROOT / "models" / "sensitive" / "bilingual_pii_xlmr"
    if (nested / "model.safetensors").is_file() or (nested / "pytorch_model.bin").is_file():
        return str(nested.resolve())
    if (parent / "model.safetensors").is_file() or (parent / "pytorch_model.bin").is_file():
        return str(parent.resolve())
    return str(nested.resolve())


if CANONICAL_LABEL_MAPPING_PATH.is_file():
    load_canonical_label_mapping(CANONICAL_LABEL_MAPPING_PATH)

_LABEL2ID, _ID2LABEL = _maps_from_bio_labels(CANONICAL_BIO_LABELS)
LABEL2ID: Final[dict[str, int]] = _LABEL2ID
ID2LABEL: Final[dict[int, str]] = _ID2LABEL

ENTITY_LABELS: Final[list[str]] = _entity_labels_from_bio(CANONICAL_BIO_LABELS)
NUM_LABELS: Final[int] = len(CANONICAL_BIO_LABELS)
SENSITIVE_LABELS: Final[frozenset[str]] = frozenset(ENTITY_LABELS)
DEFAULT_PRETRAINED_MODEL: Final[str] = resolve_default_model_path()


def normalize_entity_label(raw_label: str) -> str:
    if not isinstance(raw_label, str) or not raw_label.strip():
        raise ValueError("raw_label must be a non-empty string.")

    label = raw_label.strip().upper()
    if label == "O":
        return "O"
    if "-" in label:
        _, label = label.split("-", maxsplit=1)
    return label


def parse_bio_label(raw_label: str) -> tuple[str, str | None]:
    """Return ``("O", None)`` or ``("B"|"I", ENTITY_TYPE)``."""
    normalized = raw_label.strip().upper() if isinstance(raw_label, str) else ""
    if not normalized or normalized == "O":
        return "O", None

    if "-" in normalized:
        prefix, entity_type = normalized.split("-", maxsplit=1)
        prefix = prefix if prefix in {"B", "I"} else "B"
        return prefix, entity_type or None

    return "B", normalized


def is_sensitive_label(label: str) -> bool:
    entity_type = normalize_entity_label(label)
    return entity_type in SENSITIVE_LABELS
