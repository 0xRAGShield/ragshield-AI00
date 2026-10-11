from __future__ import annotations

from app.sensitive.labels import (
    CANONICAL_BIO_LABELS,
    CANONICAL_LABEL_MAPPING_PATH,
    ENTITY_LABELS,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    SENSITIVE_LABELS,
    load_canonical_label_mapping,
)


def test_load_canonical_label_mapping_json() -> None:
    payload = load_canonical_label_mapping()
    assert CANONICAL_LABEL_MAPPING_PATH.is_file()
    assert payload["labels"] == CANONICAL_BIO_LABELS
    assert int(payload["num_labels"]) == NUM_LABELS if "num_labels" in payload else NUM_LABELS == 35
    assert payload["label2id"]["O"] == 0
    assert payload["label2id"]["I-IP_ADDRESS"] == 34


def test_sensitive_labels_match_entity_types() -> None:
    assert SENSITIVE_LABELS == frozenset(ENTITY_LABELS)
    assert "PERSON" in SENSITIVE_LABELS
    assert "ORGANIZATION" not in SENSITIVE_LABELS
    assert "CITY" not in SENSITIVE_LABELS
    assert LABEL2ID[ID2LABEL[23]] == 23
