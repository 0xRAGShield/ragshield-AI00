
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from collections import Counter

from datasets import Dataset, DatasetDict
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

# Keep the existing train / validation / test splits.
RANDOM_SEED = 42


# Selected labels
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


# BIO label mapping
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


# Dataset loading
def parse_entities(value: str | list) -> list[dict]:
    #Parse the entities column from the SITR CSV files

    if isinstance(value, list):
        return value

    if not isinstance(value, str) or not value.strip():
        return []

    # CSV annotations may lack commas between dictionaries.
    fixed_value = re.sub(r"}\s*{", "}, {", value.strip())

    try:
        entities = json.loads(fixed_value)
    except (json.JSONDecodeError, TypeError):
        try:
            entities = ast.literal_eval(fixed_value)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(
                "Could not parse an entities annotation."
            ) from exc

    if not isinstance(entities, list):
        raise ValueError("Expected entities to be a list.")

    return entities


def load_split(split_name: str) -> Dataset:
    #Load one existing CSV split

    csv_path = (
        RAW_DATASET_DIR
        / f"sitr_arabic_pii_{split_name}.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{csv_path}"
        )

    import pandas as pd

    df = pd.read_csv(csv_path)

    required_columns = {"text", "entities"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # Preserve the row's original position within its split.
    df = df[["text", "entities"]].copy()
    df.insert(0, "original_index", range(len(df)))

    # Store annotations as valid JSON strings.
    df["entities"] = df["entities"].apply(
        lambda value: json.dumps(
            parse_entities(value),
            ensure_ascii=False,
        )
    )

    return Dataset.from_pandas(
        df,
        preserve_index=False,
    )


# Entity validation and cleaning
def validate_entity(entity: dict, text: str) -> bool:
    #Validate character offsets against the original text

    try:
        start = int(entity["start"])
        end = int(entity["end"])
        value = str(entity["text"])
    except (KeyError, TypeError, ValueError):
        return False

    if start < 0 or end <= start:
        return False

    if end > len(text):
        return False

    return text[start:end] == value


def clean_entities(
    text: str,
    entities: list[dict],
) -> list[dict]:
    #Validate entities and resolve overlapping spans

    valid_entities = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        if not validate_entity(entity, text):
            continue

        start = int(entity["start"])
        end = int(entity["end"])
        raw_label = entity.get("label")

        valid_entities.append({
            "start": start,
            "end": end,
            "label": raw_label,
            "text": str(entity["text"]),
        })

    # Prefer shorter spans when annotations overlap.
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

        entity_length = entity["end"] - entity["start"]

        if all(
            entity_length
            < previous["end"] - previous["start"]
            for previous in overlaps
        ):
            selected = [
                previous
                for previous in selected
                if previous not in overlaps
            ]
            selected.append(entity)

    return sorted(
        selected,
        key=lambda entity: (
            entity["start"],
            entity["end"],
        ),
    )



# Token-level BIO labels
def create_token_labels(
    text: str,
    entities: list[dict],
    offsets: list[tuple[int, int]],
) -> list[int]:
    """Convert character spans to BIO token labels.

    Tokens belonging to non-selected annotated entities are
    ignored (-100), rather than incorrectly labeled as O.
    """

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
            token_index
            for token_index, (token_start, token_end)
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
            # Do not train the model to treat another sensitive
            # entity as ordinary non-sensitive text
            for token_index in matching_tokens:
                labels[token_index] = -100
            continue

        unified_label = LABEL_MAP[raw_label]

        labels[matching_tokens[0]] = LABEL2ID[
            f"B-{unified_label}"
        ]

        for token_index in matching_tokens[1:]:
            labels[token_index] = LABEL2ID[
                f"I-{unified_label}"
            ]

    return labels



# Tokenization
def tokenize_example(
    example: dict,
    tokenizer,
) -> dict:
    text = example["text"]
    entities = parse_entities(example["entities"])

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    offsets = encoded.pop("offset_mapping")

    labels = create_token_labels(
        text,
        entities,
        offsets,
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "original_index": example["original_index"],
    }


def process_split(
    dataset: Dataset,
    tokenizer,
    split_name: str,
) -> Dataset:
    print()
    print("*" * 60)
    print(f"Processing split: {split_name}")
    print("*" * 60)

    # Keep only rows with at least one valid selected entity
    keep_indices = []

    for index, example in enumerate(dataset):
        entities = clean_entities(
            example["text"],
            parse_entities(example["entities"]),
        )

        if any(
            entity["label"] in LABEL_MAP
            for entity in entities
        ):
            keep_indices.append(index)

    filtered_dataset = dataset.select(keep_indices)

    print(f"Original rows: {len(dataset)}")
    print(f"Rows with selected labels: {len(filtered_dataset)}")
    print(f"Rows removed: {len(dataset) - len(filtered_dataset)}")

    def tokenize_wrapper(example):
        return tokenize_example(example, tokenizer)

    processed = filtered_dataset.map(
        tokenize_wrapper,
        remove_columns=[
            column
            for column in filtered_dataset.column_names
            if column != "original_index"
        ],
        desc=f"Tokenizing {split_name}",
    )

    return processed


# Main
def main():
    print("*" * 70)
    print("Preparing SITR Arabic PII dataset for XLM-R NER")
    print("*" * 70)

    print("\nSelected entity labels:")
    for label in LABEL_MAP:
        print(f"  {label} -> {LABEL_MAP[label]}")

    print(f"\nNumber of entity types: {len(ENTITY_LABELS)}")

    # Load the existing splits without creating a new random split.
    raw_dataset = DatasetDict({
        "train": load_split("train"),
        "validation": load_split("validation"),
        "test": load_split("test"),
    })

    print("\nRaw dataset:")
    print(raw_dataset)

    print("\nLoading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for offset mappings."
        )

    print(f"Tokenizer loaded: {MODEL_NAME}")

    processed_dataset = DatasetDict()

    for split_name in ["train", "validation", "test"]:
        processed_dataset[split_name] = process_split(
            raw_dataset[split_name],
            tokenizer,
            split_name,
        )

    # Save the processed dataset
    PROCESSED_DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSaving processed dataset")
    processed_dataset.save_to_disk(
        str(PROCESSED_DATASET_DIR)
    )

    # Summary
    print("\n" + "*" * 70)
    print("Dataset preparation completed!")
    print("*" * 70)

    print(f"\nOutput directory:\n{PROCESSED_DATASET_DIR}")

    print("\nSplit sizes:")
    for split_name, split_data in processed_dataset.items():
        print(f"  {split_name}: {len(split_data)} rows")

    print("\nColumns:")
    print(processed_dataset["train"].column_names)

    print("\nLabel mapping:")
    for label_id in sorted(ID2LABEL):
        print(f"  {label_id:2d} -> {ID2LABEL[label_id]}")

    print("\nSample original indices:")
    for split_name, split_data in processed_dataset.items():
        if len(split_data) > 0:
            print(
                f"  {split_name}:",
                split_data[0]["original_index"],
            )


if __name__ == "__main__":
    main()