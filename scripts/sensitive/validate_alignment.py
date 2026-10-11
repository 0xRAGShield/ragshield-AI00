
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATASET_DIR = (
    PROJECT_ROOT.parent
    / "data"
    / "sensitive"
)

PROCESSED_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "sensitive"
    / "processed"
    / "sitr-arabic-pii"
)

MODEL_NAME = "Davlan/xlm-roberta-base-ner-hrl"

MAX_LENGTH = 512
CHECK_PER_SPLIT = 1000


# Labels: must match prepare_dataset.py exactly
LABEL_MAP = {
    "national_id": "NATIONAL_ID",
    "passport_number": "PASSPORT_NUMBER",
    "employee_id": "EMPLOYEE_ID",
    "private_phone": "PHONE",
    "private_email": "EMAIL",
    "private_address": "ADDRESS",
    "credit_card": "CREDIT_CARD",
    "medical_record_number": "MEDICAL_RECORD_NUMBER",
    "private_date": "PRIVATE_DATE",
    "password": "PASSWORD",
    "spoken_national_id": "SPOKEN_NATIONAL_ID",
}

ENTITY_LABELS = list(dict.fromkeys(LABEL_MAP.values()))

LABEL2ID = {"O": 0}
ID2LABEL = {0: "O"}

next_id = 1

for entity_label in ENTITY_LABELS:
    LABEL2ID[f"B-{entity_label}"] = next_id
    ID2LABEL[next_id] = f"B-{entity_label}"
    next_id += 1

    LABEL2ID[f"I-{entity_label}"] = next_id
    ID2LABEL[next_id] = f"I-{entity_label}"
    next_id += 1


# Read raw CSV annotations
def parse_entities(value) -> list[dict]:
    if isinstance(value, list):
        return value

    if not isinstance(value, str) or not value.strip():
        return []

    fixed_value = re.sub(r"}\s*{", "}, {", value.strip())

    try:
        entities = json.loads(fixed_value)
    except (json.JSONDecodeError, TypeError):
        try:
            entities = ast.literal_eval(fixed_value)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(
                "Could not parse a raw entities annotation."
            ) from exc

    if not isinstance(entities, list):
        raise ValueError("Expected entities to be a list.")

    return entities


def load_raw_split(split_name: str) -> pd.DataFrame:
    csv_path = (
        RAW_DATASET_DIR
        / f"sitr_arabic_pii_{split_name}.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if not {"text", "entities"}.issubset(df.columns):
        raise ValueError(
            f"Missing text/entities columns in {csv_path.name}"
        )

    return df


# Entity cleaning: mirror prepare_dataset.py

def validate_entity(entity: dict, text: str) -> bool:
    try:
        start = int(entity["start"])
        end = int(entity["end"])
        value = str(entity["text"])
    except (KeyError, TypeError, ValueError):
        return False

    if start < 0 or end <= start or end > len(text):
        return False

    return text[start:end] == value


def clean_entities(text: str, entities: list[dict]) -> list[dict]:
    valid_entities = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        if not validate_entity(entity, text):
            continue

        valid_entities.append({
            "start": int(entity["start"]),
            "end": int(entity["end"]),
            "label": entity.get("label"),
            "text": str(entity["text"]),
        })

    valid_entities.sort(
        key=lambda entity: (
            entity["start"],
            entity["end"] - entity["start"],
        )
    )

    selected = []

    for entity in valid_entities:
        overlaps = [
            previous
            for previous in selected
            if (
                entity["start"] < previous["end"]
                and previous["start"] < entity["end"]
            )
        ]

        if not overlaps:
            selected.append(entity)
            continue

        current_length = entity["end"] - entity["start"]

        should_keep = all(
            current_length
            < previous["end"] - previous["start"]
            for previous in overlaps
        )

        if should_keep:
            selected = [
                item for item in selected
                if item not in overlaps
            ]
            selected.append(entity)

    return sorted(
        selected,
        key=lambda entity: (
            entity["start"],
            entity["end"],
        ),
    )


# Recreate token labels
def create_expected_labels(
    text: str,
    entities: list[dict],
    offsets: list[tuple[int, int]],
) -> list[int]:

    labels = [
        -100 if start == end else LABEL2ID["O"]
        for start, end in offsets
    ]

    entities = clean_entities(text, entities)

    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        raw_label = entity["label"]

        matching_tokens = [
            index
            for index, (token_start, token_end)
            in enumerate(offsets)
            if (
                token_start != token_end
                and max(start, token_start)
                < min(end, token_end)
            )
        ]

        if not matching_tokens:
            continue

        if raw_label not in LABEL_MAP:
            # Other entity types are ignored by the training loss.
            for index in matching_tokens:
                labels[index] = -100
            continue

        unified_label = LABEL_MAP[raw_label]

        labels[matching_tokens[0]] = LABEL2ID[
            f"B-{unified_label}"
        ]

        for index in matching_tokens[1:]:
            labels[index] = LABEL2ID[
                f"I-{unified_label}"
            ]

    return labels


# Validate one example
def validate_example(
    raw_example: dict,
    processed_example: dict,
    tokenizer,
) -> list[str]:

    errors = []
    text = raw_example["text"]
    entities = parse_entities(raw_example["entities"])

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]

    expected_labels = create_expected_labels(
        text,
        entities,
        offsets,
    )

    actual_labels = processed_example["labels"]

    if len(offsets) != len(actual_labels):
        return [
            f"Token/label count mismatch: "
            f"tokens={len(offsets)}, "
            f"labels={len(actual_labels)}"
        ]

    for index, (expected, actual) in enumerate(
        zip(expected_labels, actual_labels)
    ):
        if expected != actual:
            # Do not print the actual token text because
            # it may contain sensitive information.
            errors.append(
                f"Token index {index}: "
                f"expected={ID2LABEL.get(expected, expected)}, "
                f"actual={ID2LABEL.get(actual, actual)}"
            )

    return errors


# Validate one split
def validate_split(
    processed_split,
    raw_df: pd.DataFrame,
    split_name: str,
    tokenizer,
) -> tuple[int, int]:

    print("\n" + "*" * 60)
    print(f"Validating split: {split_name}")
    print("*" * 60)

    check_count = min(CHECK_PER_SPLIT, len(processed_split))

    print(f"Processed examples: {len(processed_split)}")
    print(f"Examples to check: {check_count}")

    examples_with_errors = 0
    total_errors = 0

    for position in range(check_count):
        processed_example = processed_split[position]
        original_index = int(
            processed_example["original_index"]
        )

        if not 0 <= original_index < len(raw_df):
            examples_with_errors += 1
            total_errors += 1
            print(
                f"Invalid original_index at processed "
                f"position {position}"
            )
            continue

        raw_example = raw_df.iloc[original_index].to_dict()

        errors = validate_example(
            raw_example,
            processed_example,
            tokenizer,
        )

        if errors:
            examples_with_errors += 1
            total_errors += len(errors)

            # Limit output and avoid exposing sensitive text.
            if examples_with_errors <= 5:
                print(
                    f"\nMismatch at processed position "
                    f"{position}; original index "
                    f"{original_index}"
                )

                for error in errors[:10]:
                    print(f"  - {error}")

                if len(errors) > 10:
                    print(
                        f"  ... {len(errors) - 10} "
                        "additional mismatches"
                    )

    print(f"\nExamples checked: {check_count}")
    print(f"Examples with errors: {examples_with_errors}")
    print(f"Total label mismatches: {total_errors}")

    return examples_with_errors, total_errors


# Main
def main():
    print("*" * 70)
    print("SITR Arabic PII Dataset Alignment Validation")
    print("*" * 70)

    if not PROCESSED_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n"
            f"{PROCESSED_DATASET_DIR}"
        )

    processed_dataset = load_from_disk(
        str(PROCESSED_DATASET_DIR)
    )

    print("\nProcessed dataset:")
    print(processed_dataset)

    print("\nLoading raw CSV splits")
    raw_splits = {
        split: load_raw_split(split)
        for split in ("train", "validation", "test")
    }

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for offset mappings."
        )

    print(f"Tokenizer loaded: {MODEL_NAME}")

    total_checked = 0
    total_examples_with_errors = 0
    total_errors = 0

    for split_name in ("train", "validation", "test"):
        if split_name not in processed_dataset:
            raise ValueError(
                f"Missing processed split: {split_name}"
            )

        example_count = min(
            CHECK_PER_SPLIT,
            len(processed_dataset[split_name]),
        )

        examples_with_errors, errors = validate_split(
            processed_dataset[split_name],
            raw_splits[split_name],
            split_name,
            tokenizer,
        )

        total_checked += example_count
        total_examples_with_errors += examples_with_errors
        total_errors += errors

    print("\n" + "*" * 70)
    print("FINAL VALIDATION REPORT")
    print("*" * 70)
    print(f"Examples checked: {total_checked}")
    print(f"Examples with errors: {total_examples_with_errors}")
    print(f"Total label mismatches: {total_errors}")

    if total_errors == 0:
        print("\nALIGNMENT VALIDATION PASSED")
    else:
        print("\nALIGNMENT VALIDATION FAILED")
        print(
            "Review the preparation and validation logic "
            "before training."
        )


if __name__ == "__main__":
    main()