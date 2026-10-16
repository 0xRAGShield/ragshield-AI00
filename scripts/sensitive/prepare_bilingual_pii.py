
import ast
import json
from collections import Counter
from pathlib import Path

from datasets import (
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import AutoTokenizer


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SITR_PATH = (
    PROJECT_ROOT
    / "data"
    / "sensitive"
    / "processed"
    / "sitr-arabic-pii"
)

# Use a NEW output folder to preserve the existing dataset.
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "sensitive"
    / "processed"
    / "combined_masked_sitr"
)

DATASET_NAME = "ai4privacy/pii-masking-300k"
MODEL_NAME = "Davlan/xlm-roberta-base-ner-hrl"

MAX_LENGTH = 512
RANDOM_SEED = 42
ENGLISH_TEST_SIZE = 0.10

# Experimental change:
# SITR has no PERSON labels, so ignore its O tokens during training.
MASK_SITR_O_LABELS = True


# ==================================================
# 2. UNIFIED BIO LABELS
# ==================================================

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

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# OpenPII label -> unified label.
ENTITY_MAP = {
    "EMAIL": "EMAIL",
    "TEL": "PHONE",
    "PASSPORT": "PASSPORT_NUMBER",
    "PASS": "PASSWORD",
    "BOD": "PRIVATE_DATE",
    "DATE": "PRIVATE_DATE",

    "GIVENNAME1": "PERSON",
    "GIVENNAME2": "PERSON",
    "GIVENNAME3": "PERSON",
    "LASTNAME1": "PERSON",
    "LASTNAME2": "PERSON",
    "LASTNAME3": "PERSON",

    "USERNAME": "USERNAME",
    "SOCIALNUMBER": "SOCIAL_SECURITY_NUMBER",
    "IDCARD": "ID_CARD_NUMBER",
    "DRIVERLICENSE": "DRIVER_LICENSE",
    "IP": "IP_ADDRESS",

    "STREET": "ADDRESS",
    "BUILDING": "ADDRESS",
    "SECADDRESS": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "COUNTRY": "ADDRESS",
    "POSTCODE": "ADDRESS",
}


# Original processed SITR label order.
SITR_ID2LABEL = {
    0: "O",
    1: "B-NATIONAL_ID",
    2: "I-NATIONAL_ID",
    3: "B-PASSPORT_NUMBER",
    4: "I-PASSPORT_NUMBER",
    5: "B-EMPLOYEE_ID",
    6: "I-EMPLOYEE_ID",
    7: "B-PHONE",
    8: "I-PHONE",
    9: "B-EMAIL",
    10: "I-EMAIL",
    11: "B-ADDRESS",
    12: "I-ADDRESS",
    13: "B-CREDIT_CARD",
    14: "I-CREDIT_CARD",
    15: "B-MEDICAL_RECORD_NUMBER",
    16: "I-MEDICAL_RECORD_NUMBER",
    17: "B-PRIVATE_DATE",
    18: "I-PRIVATE_DATE",
    19: "B-PASSWORD",
    20: "I-PASSWORD",
    21: "B-SPOKEN_NATIONAL_ID",
    22: "I-SPOKEN_NATIONAL_ID",
}


# ==================================================
# 3. PARSE OPENPII ANNOTATIONS
# ==================================================

def parse_entities(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return []

        return parsed if isinstance(parsed, list) else []

    return []


# ==================================================
# 4. REMAP SITR LABEL IDS
# ==================================================

def remap_sitr_dataset(dataset, split_name):
    required_columns = {"input_ids", "attention_mask", "labels"}

    if not required_columns.issubset(dataset.column_names):
        raise ValueError(
            "Processed SITR must contain input_ids, attention_mask, labels. "
            f"Found: {dataset.column_names}"
        )

    def convert(example):
        remapped_labels = []

        for old_id in example["labels"]:
            old_id = int(old_id)

            if old_id == -100:
                remapped_labels.append(-100)
                continue

            if old_id not in SITR_ID2LABEL:
                raise ValueError(
                    f"Unknown original SITR label ID: {old_id}"
                )

            label_name = SITR_ID2LABEL[old_id]

            if label_name not in LABEL2ID:
                raise ValueError(f"Missing unified label: {label_name}")

            remapped_labels.append(LABEL2ID[label_name])

        if len(remapped_labels) != len(example["input_ids"]):
            raise ValueError(
                "SITR input_ids and labels have different lengths."
            )

        # EXPERIMENTAL FIX:
        # SITR does not annotate PERSON entities.
        # Ignore its O tokens instead of treating every unannotated
        # token as a confirmed negative example.
        if MASK_SITR_O_LABELS:
            remapped_labels = [
                -100 if label_id == LABEL2ID["O"] else label_id
                for label_id in remapped_labels
            ]

        return {"labels": remapped_labels}

    processed = dataset.map(
        convert,
        desc=f"Remapping SITR labels ({split_name})",
    )

    # Print counts so we can verify the masking effect.
    counts = Counter(
        label_id
        for example in processed
        for label_id in example["labels"]
    )

    print(f"\nSITR {split_name} label counts after remapping:")
    print(f"  Ignored (-100): {counts.get(-100, 0):,}")
    print(f"  O:             {counts.get(LABEL2ID['O'], 0):,}")

    for label in LABELS:
        if label == "O":
            continue
        count = counts.get(LABEL2ID[label], 0)
        if count:
            print(f"  {label}: {count:,}")

    return processed


# ==================================================
# 5. TOKENIZE OPENPII WITH CHARACTER OFFSETS
# ==================================================

def encode_openpii_example(example, tokenizer):
    text = example["source_text"]
    entities = parse_entities(example["privacy_mask"])

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    offsets = encoded.pop("offset_mapping")

    # Special tokens receive -100; ordinary tokens start as O.
    token_labels = [
        -100 if start == end else LABEL2ID["O"]
        for start, end in offsets
    ]

    ignored_tokens = set()
    selected_spans = []

    for entity in entities:
        try:
            start = int(entity["start"])
            end = int(entity["end"])
            source_label = str(entity["label"]).strip().upper()
        except (KeyError, TypeError, ValueError):
            continue

        if start < 0 or end <= start or end > len(text):
            continue

        overlapping_tokens = [
            i
            for i, (token_start, token_end) in enumerate(offsets)
            if (
                token_start < end
                and token_end > start
                and token_start != token_end
            )
        ]

        if not overlapping_tokens:
            continue

        unified_entity = ENTITY_MAP.get(source_label)

        if unified_entity is None:
            # Do not treat other annotated PII as O.
            ignored_tokens.update(overlapping_tokens)
        else:
            selected_spans.append(
                (start, end, overlapping_tokens, unified_entity)
            )

    for token_index in ignored_tokens:
        token_labels[token_index] = -100

    # Process spans in text order.
    # The first selected annotation claims overlapping tokens.
    claimed_tokens = set()

    for start, end, token_indices, entity_label in sorted(
        selected_spans,
        key=lambda item: (item[0], item[1]),
    ):
        available = [
            index
            for index in token_indices
            if index not in claimed_tokens
        ]

        if not available:
            continue

        for position, token_index in enumerate(available):
            prefix = "B" if position == 0 else "I"
            label_id = LABEL2ID[f"{prefix}-{entity_label}"]

            token_labels[token_index] = label_id
            claimed_tokens.add(token_index)

    result = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": token_labels,
    }

    if not (
        len(result["input_ids"])
        == len(result["attention_mask"])
        == len(result["labels"])
    ):
        raise ValueError("OpenPII token/label alignment failed.")

    return result


# ==================================================
# 6. LOAD ENGLISH DATA AND CREATE A FIXED TEST SET
# ==================================================

def load_english_splits(tokenizer):
    english_splits = {}

    for split_name in ("train", "validation"):
        print(f"\nDownloading OpenPII {split_name}...")

        dataset = load_dataset(
            DATASET_NAME,
            split=split_name,
        )

        required = {"source_text", "privacy_mask", "language", "id"}
        missing = required - set(dataset.column_names)

        if missing:
            raise ValueError(
                f"Missing OpenPII columns in {split_name}: {missing}"
            )

        dataset = dataset.filter(
            lambda row: (
                str(row["language"]).strip().lower() == "english"
            ),
            desc=f"Filtering English {split_name}",
        )

        ids = [str(value) for value in dataset["id"]]

        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Duplicate OpenPII IDs found in {split_name}."
            )

        print(f"English {split_name}: {len(dataset):,}")
        english_splits[split_name] = dataset

    # Reserve 10% of English TRAIN for testing.
    divided = english_splits["train"].train_test_split(
        test_size=ENGLISH_TEST_SIZE,
        seed=RANDOM_SEED,
        shuffle=True,
    )

    raw_train = divided["train"]
    raw_test = divided["test"]
    raw_validation = english_splits["validation"]

    train_ids = set(str(value) for value in raw_train["id"])
    validation_ids = set(str(value) for value in raw_validation["id"])
    test_ids = set(str(value) for value in raw_test["id"])

    if train_ids & validation_ids:
        raise ValueError("English Train/Validation ID overlap detected.")

    if train_ids & test_ids:
        raise ValueError("English Train/Test ID overlap detected.")

    if validation_ids & test_ids:
        raise ValueError("English Validation/Test ID overlap detected.")

    print("\nEnglish split sizes before tokenization:")
    print(f"Train:      {len(raw_train):,}")
    print(f"Validation: {len(raw_validation):,}")
    print(f"Test:       {len(raw_test):,}")

    def tokenize_split(raw_dataset, name):
        print(f"\nTokenizing English {name}...")

        return raw_dataset.map(
            lambda row: encode_openpii_example(row, tokenizer),
            remove_columns=raw_dataset.column_names,
            desc=f"Tokenizing English {name}",
        )

    return {
        "train": tokenize_split(raw_train, "train"),
        "validation": tokenize_split(raw_validation, "validation"),
        "test": tokenize_split(raw_test, "test"),
    }


# 7. VALIDATE AND COMBINE

def validate_tokenized_dataset(dataset, split_name):
    for index, example in enumerate(dataset):
        input_length = len(example["input_ids"])

        if (
            input_length != len(example["attention_mask"])
            or input_length != len(example["labels"])
        ):
            raise ValueError(
                f"Length mismatch in {split_name}, example {index}."
            )

        for label_id in example["labels"]:
            if label_id != -100 and label_id not in ID2LABEL:
                raise ValueError(
                    f"Invalid label ID {label_id} in {split_name}."
                )


def main():
    if not SITR_PATH.exists():
        raise FileNotFoundError(
            f"Processed SITR dataset not found: {SITR_PATH}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Rename or remove that output folder before rerunning. "
            "The original datasets will not be modified."
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for offset alignment."
        )

    print("\nLoading processed SITR...")
    sitr = load_from_disk(str(SITR_PATH))

    if not isinstance(sitr, DatasetDict):
        raise TypeError(
            "Expected SITR to be saved as a DatasetDict."
        )

    required_splits = {"train", "validation", "test"}

    if not required_splits.issubset(set(sitr.keys())):
        raise ValueError(
            f"SITR must contain {sorted(required_splits)}. "
            f"Found: {list(sitr.keys())}"
        )

    print("SITR split sizes:")
    for name, split in sitr.items():
        print(f"  {name}: {len(split):,}")

    sitr_processed = {
        name: remap_sitr_dataset(split, name)
        for name, split in sitr.items()
    }

    english = load_english_splits(tokenizer)

    combined = DatasetDict({
        "train": concatenate_datasets([
            sitr_processed["train"],
            english["train"],
        ]),
        "validation": concatenate_datasets([
            sitr_processed["validation"],
            english["validation"],
        ]),
        "test": concatenate_datasets([
            sitr_processed["test"],
            english["test"],
        ]),
    })

    print("\nValidating final splits...")
    for split_name, split in combined.items():
        validate_tokenized_dataset(split, split_name)

    print("\nSaving combined dataset...")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(OUTPUT_PATH))

    mapping_path = OUTPUT_PATH / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "labels": LABELS,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "model_name": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "seed": RANDOM_SEED,
                "english_test_size_from_original_train": ENGLISH_TEST_SIZE,
                "entity_map": ENTITY_MAP,
                "sitr_o_tokens_masked": MASK_SITR_O_LABELS,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "dataset": DATASET_NAME,
        "english_test_source": (
            "A held-out 10% subset of the original English train split"
        ),
        "seed": RANDOM_SEED,
        "sitr_o_tokens_masked": MASK_SITR_O_LABELS,
        "splits": {
            name: {
                "total": len(split),
                "labelled_token_counts": dict(
                    Counter(
                        ID2LABEL[label_id]
                        for example in split
                        for label_id in example["labels"]
                        if label_id != -100
                    )
                ),
                "ignored_token_count": sum(
                    label_id == -100
                    for example in split
                    for label_id in example["labels"]
                ),
            }
            for name, split in combined.items()
        },
    }

    summary_path = OUTPUT_PATH / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========== FINAL DATASET ==========")
    for name, split in combined.items():
        print(f"{name}: {len(split):,} examples")

    print("Unified labels:", len(LABELS))
    print("SITR O masking:", MASK_SITR_O_LABELS)
    print("Output:", OUTPUT_PATH)
    print("Label mapping:", mapping_path)
    print("Summary:", summary_path)
    print("\nPreparation completed successfully.")


if __name__ == "__main__":
    main()
