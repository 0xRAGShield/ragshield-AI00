
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# 1. PATHS - LOCAL WINDOWS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# New merged dataset
PROCESSED_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "sensitive"
    / "processed"
    / "V3OFcombined"
)

# Keep the original label mapping as the source of truth.
# The merge output currently contains metadata and summary;
# do not assume it contains label_mapping.json.
LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "sensitive"
    / "processed"
    / "V3OFcombined"
    / "label_mapping.json"
)

# Save the newly trained model separately.
MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "sensitive"
    /  "bilingual_pii_xlmr_V3OFcombined_run2"
)

# Keep this unchanged until we verify tokenizer compatibility.
MODEL_NAME = "Davlan/xlm-roberta-base-ner-hrl"


# ============================================================
# 2. TRAINING CONFIGURATION
# ============================================================

NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 2
MAX_LENGTH = 512


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 4. LOAD AND VALIDATE LABEL MAPPING
# ============================================================

def load_label_mapping():
    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping not found:\n{LABEL_MAPPING_PATH}"
        )

    with open(
        LABEL_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        mapping = json.load(file)

    label2id = {
    str(label): int(label_id)
    for label, label_id in mapping["label2id"].items()
    }

    id2label = {
    int(label_id): label
    for label_id, label in mapping["id2label"].items()
     }

    num_labels = len(label2id)

    if len(id2label) != num_labels:
        raise ValueError(
            "label2id and id2label have different sizes."
        )

    if label2id.get("O") != 0:
        raise ValueError("The O label must have ID 0.")

    if set(id2label) != set(range(num_labels)):
        raise ValueError(
            "Label IDs must be consecutive, starting at 0."
        )

    for label, label_id in label2id.items():
        if id2label.get(label_id) != label:
            raise ValueError(
                f"Inconsistent mapping for label: {label}"
            )

    entity_labels = [
        label[2:]
        for label in label2id
        if label.startswith("B-")
    ]

    for entity in entity_labels:
        if f"I-{entity}" not in label2id:
            raise ValueError(
                f"Missing I-{entity} label."
            )

    print("\n" + "=" * 65)
    print("BILINGUAL PII LABEL MAPPING")
    print("=" * 65)
    print(f"Number of labels: {num_labels}")
    print(f"Number of entity types: {len(entity_labels)}")

    for label_id in range(num_labels):
        print(f"{label_id:2d} -> {id2label[label_id]}")

    print("=" * 65)

    return label2id, id2label, entity_labels, num_labels, mapping


# ============================================================
# 5. VALIDATE DATASET STRUCTURE
# ============================================================

def validate_dataset(dataset, num_labels: int) -> None:
    required_splits = {"train", "validation", "test"}

    missing_splits = required_splits - set(dataset.keys())

    if missing_splits:
        raise ValueError(
            f"Missing splits: {sorted(missing_splits)}"
        )

    for split_name in ("train", "validation", "test"):
        split = dataset[split_name]

        required_columns = {
            "input_ids",
            "attention_mask",
            "labels",
        }

        missing_columns = required_columns - set(
            split.column_names
        )

        if missing_columns:
            raise ValueError(
                f"{split_name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        print(f"{split_name:<12}: {len(split):,} examples")

        for row_index, row in enumerate(split):
            input_ids = row["input_ids"]
            attention_mask = row["attention_mask"]
            labels = row["labels"]

            if not (
                len(input_ids)
                == len(attention_mask)
                == len(labels)
            ):
                raise ValueError(
                    f"Sequence length mismatch in {split_name}, "
                    f"example {row_index}."
                )

            for label_id in labels:
                if label_id == -100:
                    continue

                if not 0 <= label_id < num_labels:
                    raise ValueError(
                        f"Invalid label ID {label_id} in "
                        f"{split_name}, example {row_index}."
                    )

    print("\nDataset structure and label IDs are valid.")


# ============================================================
# 6. LABEL DISTRIBUTION
# ============================================================

def print_label_distribution(
    dataset,
    id2label: dict[int, str],
    num_labels: int,
) -> None:
    for split_name in ("train", "validation", "test"):
        counts = Counter(
            int(label_id)
            for row in dataset[split_name]["labels"]
            for label_id in row
            if label_id != -100
        )

        ignored_count = sum(
            1
            for row in dataset[split_name]["labels"]
            for label_id in row
            if label_id == -100
        )

        print(f"\nLabel distribution: {split_name}")
        print("-" * 50)

        for label_id in range(num_labels):
            print(
                f"{id2label[label_id]:30s}: "
                f"{counts.get(label_id, 0):,}"
            )

        print(f"Ignored tokens (-100): {ignored_count:,}")


# ============================================================
# 7. TOKEN-LEVEL EVALUATION METRICS
# ============================================================

def build_compute_metrics(
    id2label: dict[int, str],
    entity_labels: list[str],
    num_labels: int,
):
    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.argmax(predictions, axis=-1)

        true_labels = []
        pred_labels = []

        for prediction_row, label_row in zip(
            predictions, labels
        ):
            for prediction_id, label_id in zip(
                prediction_row, label_row
            ):
                if label_id == -100:
                    continue

                true_labels.append(int(label_id))
                pred_labels.append(int(prediction_id))

        if not true_labels:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "f1_sensitive_macro": 0.0,
                "entity_f1": 0.0,
            }

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                true_labels,
                pred_labels,
                average="weighted",
                zero_division=0,
            )
        )

        sensitive_ids = list(range(1, num_labels))

        _, _, sensitive_f1, _ = (
            precision_recall_fscore_support(
                true_labels,
                pred_labels,
                labels=sensitive_ids,
                average="macro",
                zero_division=0,
            )
        )

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f1_sensitive_macro": float(sensitive_f1),
        }

        all_ids = list(range(num_labels))

        per_precision, per_recall, per_f1, _ = (
            precision_recall_fscore_support(
                true_labels,
                pred_labels,
                labels=all_ids,
                average=None,
                zero_division=0,
            )
        )

        for label_id in all_ids:
            label_name = id2label[label_id].replace("-", "_")

            metrics[f"precision_{label_name}"] = float(
                per_precision[label_id]
            )
            metrics[f"recall_{label_name}"] = float(
                per_recall[label_id]
            )
            metrics[f"f1_{label_name}"] = float(
                per_f1[label_id]
            )

        # Average B/I token F1 is not exact entity-span F1.
        entity_f1_values = []

        for entity in entity_labels:
            b_f1 = metrics[f"f1_B_{entity}"]
            i_f1 = metrics[f"f1_I_{entity}"]
            average_f1 = (b_f1 + i_f1) / 2.0

            metrics[f"f1_entity_{entity}"] = float(average_f1)
            entity_f1_values.append(average_f1)

        metrics["entity_f1"] = float(
            np.mean(entity_f1_values)
        )

        return metrics

    return compute_metrics


# ============================================================
# 8. INFERENCE HELPER
# ============================================================

def predict_entities(
    text: str,
    model,
    tokenizer,
    device,
    id2label: dict[int, str],
):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    model.eval()

    with torch.no_grad():
        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        predictions = torch.argmax(
            outputs.logits,
            dim=-1,
        )[0].tolist()

    token_predictions = []

    for token_index, (label_id, offset) in enumerate(
        zip(predictions, offsets)
    ):
        start, end = offset

        if start == end:
            continue

        token_predictions.append({
            "label": id2label[int(label_id)],
            "start": start,
            "end": end,
            "confidence": float(
                probabilities[token_index, label_id].item()
            ),
        })

    entities = []
    current_entity = None

    def close_entity():
        nonlocal current_entity

        if current_entity is not None:
            scores = current_entity.pop("_scores")

            current_entity["text"] = text[
                current_entity["start"]:current_entity["end"]
            ]

            current_entity["score"] = float(np.mean(scores))

            entities.append(current_entity)
            current_entity = None

    for item in token_predictions:
        label = item["label"]

        if label == "O":
            close_entity()
            continue

        if label.startswith("B-"):
            close_entity()

            current_entity = {
                "label": label[2:],
                "start": item["start"],
                "end": item["end"],
                "_scores": [item["confidence"]],
            }

        elif label.startswith("I-"):
            entity_type = label[2:]

            if (
                current_entity is not None
                and current_entity["label"] == entity_type
                and current_entity["end"] <= item["start"]
            ):
                current_entity["end"] = item["end"]
                current_entity["_scores"].append(
                    item["confidence"]
                )
            else:
                close_entity()

                current_entity = {
                    "label": entity_type,
                    "start": item["start"],
                    "end": item["end"],
                    "_scores": [item["confidence"]],
                }

    close_entity()

    return entities


# ============================================================
# 9. MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("BILINGUAL SENSITIVE DATA NER TRAINING")
    print("SITR Arabic PII + OpenPII English")
    print("=" * 65)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {PROCESSED_DATASET_DIR}")
    print(f"Output: {MODEL_OUTPUT_DIR}")
    print(f"Base model: {MODEL_NAME}")

    set_seed(SEED)
    set_all_seeds(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not PROCESSED_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{PROCESSED_DATASET_DIR}\n"
            "Run prepare_bilingual_pii.py and check its output path."
        )

    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping not found:\n{LABEL_MAPPING_PATH}"
        )

    # Avoid accidentally overwriting an existing output directory.
    if MODEL_OUTPUT_DIR.exists() and any(MODEL_OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty:\n"
            f"{MODEL_OUTPUT_DIR}\n"
            "Choose a new output directory before training."
        )

    # --------------------------------------------------------
    # Load label mapping
    # --------------------------------------------------------

    (
        label2id,
        id2label,
        entity_labels,
        num_labels,
        mapping,
    ) = load_label_mapping()

    if num_labels != 35:
        raise ValueError(
            f"Expected 35 labels, found {num_labels}."
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading combined dataset...")

    dataset = load_from_disk(str(PROCESSED_DATASET_DIR))

    print("\nDataset sizes:")
    validate_dataset(dataset, num_labels)

    print_label_distribution(
        dataset,
        id2label,
        num_labels,
    )

    # --------------------------------------------------------
    # Verify data has supervised tokens
    # --------------------------------------------------------

    for split_name in ("train", "validation", "test"):
        supervised_count = sum(
            1
            for row in dataset[split_name]["labels"]
            for label_id in row
            if label_id != -100
        )

        if supervised_count == 0:
            raise ValueError(
                f"No supervised tokens remain in {split_name}."
            )

        print(
            f"{split_name}: "
            f"{supervised_count:,} supervised tokens"
        )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for offset mapping."
        )

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # Load pretrained model
    # --------------------------------------------------------

    print("\nLoading pretrained token classification model...")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Data collator
    # --------------------------------------------------------

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,

        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",

        load_best_model_at_end=True,
        metric_for_best_model="f1_sensitive_macro",
        greater_is_better=True,

        fp16=torch.cuda.is_available(),

        report_to="none",

        seed=SEED,
        data_seed=SEED,

        remove_unused_columns=True,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    compute_metrics = build_compute_metrics(
        id2label=id2label,
        entity_labels=entity_labels,
        num_labels=num_labels,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STARTING TRAINING")
    print("=" * 65)

    print(f"Train examples: {len(dataset['train']):,}")
    print(f"Validation examples: {len(dataset['validation']):,}")
    print(f"Test examples: {len(dataset['test']):,}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Labels: {num_labels}")

    train_result = trainer.train()

    print("\nTraining completed.")

    print("\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # Final validation evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL VALIDATION EVALUATION")
    print("=" * 65)

    validation_results = trainer.evaluate(
        eval_dataset=dataset["validation"],
    )

    for key, value in validation_results.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # --------------------------------------------------------
    # Save best model and tokenizer
    # --------------------------------------------------------

    print("\nSaving best model...")

    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    saved_mapping = dict(mapping)
    saved_mapping["base_model"] = MODEL_NAME
    saved_mapping["dataset"] = (
        "SITR Arabic PII + OpenPII English"
    )
    saved_mapping["label2id"] = label2id
    saved_mapping["id2label"] = {
        str(key): value
        for key, value in id2label.items()
    }
    saved_mapping["num_labels"] = num_labels
    saved_mapping["sitr_o_tokens_masked"] = True

    with open(
        MODEL_OUTPUT_DIR / "label_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            saved_mapping,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Model saved to: {MODEL_OUTPUT_DIR}")

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL TEST SET EVALUATION")
    print("=" * 65)

    test_results = trainer.evaluate(
        eval_dataset=dataset["test"],
        metric_key_prefix="test",
    )

    for key, value in test_results.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    with open(
        MODEL_OUTPUT_DIR / "evaluation_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "validation": validation_results,
                "test": test_results,
            },
            file,
            ensure_ascii=False,
            indent=2,
            default=float,
        )

    # --------------------------------------------------------
    # Reload model and run sample inference
    # --------------------------------------------------------

    print("\nReloading saved model...")

    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_OUTPUT_DIR),
        use_fast=True,
    )

    reloaded_model = (
        AutoModelForTokenClassification.from_pretrained(
            str(MODEL_OUTPUT_DIR)
        )
    )

    inference_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    reloaded_model.to(inference_device)
    reloaded_model.eval()

    test_texts = [
        "My email is demo.user@example.com.",
        "Call the fictional number 01000000000.",
        "رقم البطاقة 12345678901234",
        "رقم جواز السفر A1234567",
        "Employee ID: EMP-2048",
        "عنوان تجريبي: 15 Tahrir Street, Cairo.",
        "My name is Manar",
        "My name is Ahmed Mohamed.",
        "My name is Sara Ali.",
        "اسمي منار عبدالباقي حسن.",
        "My birth date is 07/07/2005",
        "My password is ExamplePass!234",
        "Medical record number: MRN-2026-001",
        "تاريخ الميلاد 15/03/2000",
        "رقم بطاقتي الائتمانية 4111111111111111",
    ]

    print("\n" + "=" * 65)
    print("INFERENCE EXAMPLES")
    print("=" * 65)

    for sample in test_texts:
        entities = predict_entities(
            sample,
            reloaded_model,
            reloaded_tokenizer,
            inference_device,
            id2label,
        )

        print(f"\nInput: {sample}")

        if not entities:
            print("No entities detected.")
            continue

        for entity in entities:
            print(
                f"Text: {entity['text']} | "
                f"Label: {entity['label']} | "
                f"Confidence: {entity['score']:.4f}"
            )

    print("\n" + "=" * 65)
    print("TRAINING AND EVALUATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
