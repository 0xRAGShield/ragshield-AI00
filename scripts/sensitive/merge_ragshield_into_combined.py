
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OLD_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sensitive"
    / "processed"
    / "combined_masked_sitr"
)

RAGSHIELD_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "sensitive"
    / "processed"
    / "ragshield_pii_dataset_audited"
    / "ragshield_pii_dataset_unified.csv"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "sensitive"
    / "bilingual_pii_xlmr"
   
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "sensitive"
    / "processed"
    / "V3OFcombined"
)

MAX_LENGTH = 512


# ============================================================
# 2. CANONICAL LABEL MAPPING
# Must match the existing model's label IDs exactly.
# ============================================================

LABELS = [
    "O",
    "B-NATIONAL_ID", "I-NATIONAL_ID",
    "B-PASSPORT_NUMBER", "I-PASSPORT_NUMBER",
    "B-EMPLOYEE_ID", "I-EMPLOYEE_ID",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-ADDRESS", "I-ADDRESS",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-MEDICAL_RECORD_NUMBER", "I-MEDICAL_RECORD_NUMBER",
    "B-PRIVATE_DATE", "I-PRIVATE_DATE",
    "B-PASSWORD", "I-PASSWORD",
    "B-SPOKEN_NATIONAL_ID", "I-SPOKEN_NATIONAL_ID",
    "B-PERSON", "I-PERSON",
    "B-USERNAME", "I-USERNAME",
    "B-SOCIAL_SECURITY_NUMBER", "I-SOCIAL_SECURITY_NUMBER",
    "B-ID_CARD_NUMBER", "I-ID_CARD_NUMBER",
    "B-DRIVER_LICENSE", "I-DRIVER_LICENSE",
    "B-IP_ADDRESS", "I-IP_ADDRESS",
]

LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


# Aliases in the audited RAGShield dataset.
ENTITY_MAP = {
    "NAME": "PERSON",
    "PERSON": "PERSON",

    "CARD": "CREDIT_CARD",
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",

    "DATE": "PRIVATE_DATE",
    "PRIVATE_DATE": "PRIVATE_DATE",
    "DOB": "PRIVATE_DATE",
    "BOD": "PRIVATE_DATE",

    "ID_CARD": "ID_CARD_NUMBER",
    "ID_CARD_NUMBER": "ID_CARD_NUMBER",

    "IP": "IP_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",

    "MRN": "MEDICAL_RECORD_NUMBER",
    "MEDICAL_RECORD_NUMBER": "MEDICAL_RECORD_NUMBER",

    "PASSPORT": "PASSPORT_NUMBER",
    "PASSPORT_NUMBER": "PASSPORT_NUMBER",

    "SSN": "SOCIAL_SECURITY_NUMBER",
    "SOCIAL_SECURITY_NUMBER": "SOCIAL_SECURITY_NUMBER",

    "TEL": "PHONE",
    "PHONE": "PHONE",

    "EMAIL": "EMAIL",
    "ADDRESS": "ADDRESS",
    "EMPLOYEE_ID": "EMPLOYEE_ID",
    "NATIONAL_ID": "NATIONAL_ID",
    "SPOKEN_NATIONAL_ID": "SPOKEN_NATIONAL_ID",
    "DRIVER_LICENSE": "DRIVER_LICENSE",
    "PASSWORD": "PASSWORD",
    "PASS": "PASSWORD",
    "USERNAME": "USERNAME",
}


# ============================================================
# 3. PARSE SPAN ANNOTATIONS
# ============================================================

def parse_spans(value):
    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        raise ValueError(
            f"Expected serialized span list, got {type(value).__name__}"
        )

    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        result = ast.literal_eval(value)

    if not isinstance(result, list):
        raise ValueError("Span annotations must be a list.")

    return result


# ============================================================
# 4. VALIDATE LABEL MAPPING
# ============================================================

def validate_label_mapping():
    if len(LABELS) != 35:
        raise ValueError(
            f"Expected 35 labels, found {len(LABELS)}."
        )

    config_path = TOKENIZER_PATH / "config.json"

    # If the local model config is available, check its label IDs.
    if config_path.exists():
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)

        model_label2id = config.get("label2id")

        if model_label2id:
            normalized = {
                str(label): int(index)
                for label, index in model_label2id.items()
            }

            if normalized != LABEL2ID:
                raise ValueError(
                    "The configured label IDs do not match the "
                    "existing model mapping. Stop before merging."
                )

    print(f"Label mapping validated: {len(LABELS)} labels.")


# ============================================================
# 5. ENCODE ONE RAGSHIELD EXAMPLE
# ============================================================

def encode_example(row, tokenizer):
    text = row["source_text"]

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"Empty source_text for ID {row['id']}."
        )

    spans = parse_spans(row["unified_span_labels"])

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    offsets = encoded.pop("offset_mapping")

    # Special tokens are ignored by the token classification loss.
    labels = [
        -100 if start == end else LABEL2ID["O"]
        for start, end in offsets
    ]

    visible_end = max(
        (
            end
            for start, end in offsets
            if end > start
        ),
        default=0,
    )

    aligned_spans = []

    for span in spans:
        try:
            start = int(span["start"])
            end = int(span["end"])
            original_label = str(span["label"]).strip().upper()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed annotation in {row['id']}: {span}"
            ) from exc

        if start < 0 or end <= start or end > len(text):
            raise ValueError(
                f"Invalid character offsets in {row['id']}: "
                f"start={start}, end={end}, text_length={len(text)}"
            )

        entity_type = ENTITY_MAP.get(original_label)

        if entity_type is None:
            raise ValueError(
                f"Unknown entity label {original_label} "
                f"in sample {row['id']}."
            )

        if end > visible_end:
            raise ValueError(
                f"Entity was truncated in sample {row['id']}. "
                "Increase MAX_LENGTH or inspect the annotation."
            )

        token_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if (
                token_start < end
                and token_end > start
                and token_start != token_end
            )
        ]

        if not token_indices:
            raise ValueError(
                f"Could not align entity to tokens in {row['id']}: "
                f"{span}"
            )

        # Ensure every token of an entity is covered consistently.
        covered_start = min(offsets[i][0] for i in token_indices)
        covered_end = max(offsets[i][1] for i in token_indices)

        if covered_start > start or covered_end < end:
            raise ValueError(
                f"Token alignment does not fully cover entity "
                f"in {row['id']}: {span}"
            )

        aligned_spans.append(
            (start, end, token_indices, entity_type)
        )

    claimed_tokens = set()

    for start, end, token_indices, entity_type in sorted(
        aligned_spans,
        key=lambda item: (item[0], item[1]),
    ):
        if any(index in claimed_tokens for index in token_indices):
            raise ValueError(
                f"Overlapping token annotations in sample {row['id']}."
            )

        for position, token_index in enumerate(token_indices):
            prefix = "B" if position == 0 else "I"
            label_name = f"{prefix}-{entity_type}"

            if label_name not in LABEL2ID:
                raise ValueError(
                    f"Label {label_name} missing from unified mapping."
                )

            labels[token_index] = LABEL2ID[label_name]
            claimed_tokens.add(token_index)

    if not (
        len(encoded["input_ids"])
        == len(encoded["attention_mask"])
        == len(labels)
    ):
        raise ValueError(
            f"Token/label length mismatch for {row['id']}."
        )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


# ============================================================
# 6. LOAD AND PREPARE RAGSHIELD SPLITS
# ============================================================

def prepare_ragshield(tokenizer):
    df = pd.read_csv(
        RAGSHIELD_CSV_PATH,
        encoding="utf-8-sig",
    )

    required_columns = {
        "id",
        "language",
        "source_text",
        "unified_span_labels",
        "set",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["id"] = df["id"].astype(str)
    df["language"] = (
        df["language"].astype(str).str.strip().str.lower()
    )
    df["set"] = (
        df["set"].astype(str).str.strip().str.lower()
    )

    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs in RAGShield CSV.")

    if df["source_text"].isna().any():
        raise ValueError("Missing source_text values.")

    if not set(df["set"]).issubset({"train", "val", "test"}):
        raise ValueError("Unexpected split names in RAGShield CSV.")

    if not set(df["language"]).issubset(
        {"ar", "en", "franco", "mixed"}
    ):
        raise ValueError("Unexpected language value in RAGShield CSV.")

    prepared = {}
    metadata = {}

    split_mapping = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }

    for csv_split, output_split in split_mapping.items():
        part = df[df["set"] == csv_split].reset_index(drop=True)

        rows = []
        metadata_rows = []

        print(
            f"\nPreparing RAGShield {output_split}: "
            f"{len(part):,} samples"
        )

        for index, row in enumerate(part.to_dict(orient="records")):
            encoded = encode_example(row, tokenizer)

            # Existing original_index values are retained.
            # Negative values identify new RAGShield examples.
            encoded["original_index"] = -(index + 1)

            # Reorder columns to match the existing Dataset.
            rows.append({
                "original_index": encoded["original_index"],
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encoded["labels"],
            })

            metadata_rows.append({
                "original_index": -(index + 1),
                "id": row["id"],
                "language": row["language"],
                "source": "ragshield",
            })

        prepared[output_split] = Dataset.from_list(rows)
        metadata[output_split] = metadata_rows

        print(f"  Prepared: {len(rows):,}")
        print(
            "  Languages:",
            dict(Counter(part["language"].tolist())),
        )

    return DatasetDict(prepared), metadata


# ============================================================
# 7. VALIDATE AND MERGE
# ============================================================

def validate_dataset(dataset, split_name):
    required_columns = {
        "original_index",
        "input_ids",
        "attention_mask",
        "labels",
    }

    if set(dataset.column_names) != required_columns:
        raise ValueError(
            f"Unexpected columns in {split_name}: "
            f"{dataset.column_names}"
        )

    for index, row in enumerate(dataset):
        if not (
            len(row["input_ids"])
            == len(row["attention_mask"])
            == len(row["labels"])
        ):
            raise ValueError(
                f"Length mismatch in {split_name}, row {index}."
            )

        for label_id in row["labels"]:
            if label_id != -100 and label_id not in ID2LABEL:
                raise ValueError(
                    f"Invalid label ID {label_id} in "
                    f"{split_name}, row {index}."
                )


def main():
    print("=" * 65)
    print("MERGE RAGSHIELD INTO EXISTING COMBINED DATASET")
    print("=" * 65)

    if not OLD_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Existing combined dataset not found:\n{OLD_DATASET_PATH}"
        )

    if not RAGSHIELD_CSV_PATH.exists():
        raise FileNotFoundError(
            f"RAGShield CSV not found:\n{RAGSHIELD_CSV_PATH}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output directory already exists:\n{OUTPUT_PATH}\n"
            "Choose a new output directory or inspect the existing "
            "output. The old dataset will not be overwritten."
        )

    validate_label_mapping()

    print("\nLoading the existing combined dataset...")
    old = load_from_disk(str(OLD_DATASET_PATH))

    if not isinstance(old, DatasetDict):
        raise TypeError("Expected an existing DatasetDict.")

    expected_splits = {"train", "validation", "test"}

    if set(old.keys()) != expected_splits:
        raise ValueError(
            f"Unexpected old splits: {list(old.keys())}"
        )

    for name, dataset in old.items():
        validate_dataset(dataset, name)
        print(f"Old {name}: {len(dataset):,}")

    print("\nLoading tokenizer from the local model directory...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH),
        use_fast=True,
        local_files_only=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for character offsets."
        )

    ragshield, metadata = prepare_ragshield(tokenizer)

    print("\nMerging the corresponding splits...")

    combined = DatasetDict({
        split: concatenate_datasets([
            old[split],
            ragshield[split],
        ])
        for split in ("train", "validation", "test")
    })

    print("\nValidating merged dataset...")

    for split, dataset in combined.items():
        validate_dataset(dataset, split)

    # Save only after every split passes validation.
    OUTPUT_PATH.mkdir(parents=True, exist_ok=False)
    combined.save_to_disk(str(OUTPUT_PATH))

    # Preserve language and original sample IDs for later evaluation.
    metadata_path = OUTPUT_PATH / "ragshield_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "base_dataset": str(OLD_DATASET_PATH),
        "added_dataset": str(RAGSHIELD_CSV_PATH),
        "tokenizer": str(TOKENIZER_PATH),
        "max_length": MAX_LENGTH,
        "label_count": len(LABELS),
        "splits": {},
    }

    for split, dataset in combined.items():
        label_counts = Counter()
        ignored_count = 0

        for row in dataset:
            for label_id in row["labels"]:
                if label_id == -100:
                    ignored_count += 1
                else:
                    label_counts[ID2LABEL[label_id]] += 1

        summary["splits"][split] = {
            "total_examples": len(dataset),
            "added_ragshield_examples": len(ragshield[split]),
            "old_examples": len(old[split]),
            "ignored_tokens": ignored_count,
            "label_counts": dict(label_counts),
        }

        print(
            f"{split}: {len(old[split]):,} old + "
            f"{len(ragshield[split]):,} new = "
            f"{len(dataset):,} total"
        )

    summary_path = OUTPUT_PATH / "dataset_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print("Output:", OUTPUT_PATH)
    print("Metadata:", metadata_path)
    print("Summary:", summary_path)
    print("\nThe original combined dataset was not modified.")


if __name__ == "__main__":
    main()
