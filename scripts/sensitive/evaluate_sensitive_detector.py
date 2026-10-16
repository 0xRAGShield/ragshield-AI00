
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# 1. PATHS AND CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sensitive"
    / "processed"
    / "ragshield_pii_dataset_audited"
    / "ragshield_pii_dataset_unified.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "sensitive_detector"

# Limit the number of error examples saved per category.
MAX_ERROR_EXAMPLES = 500

# Keep labels consistent between gold annotations and predictions.
LABEL_ALIASES = {
    "NAME": "PERSON",
    "IP": "IP_ADDRESS",
    # Unified identity-number label
    "ID_CARD_NUMBER": "NATIONAL_ID",


}


# 2. GENERAL HELPERS

def get_field(obj: Any, *names: str, default=None):
    """Read a field from either a dict or an object."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]

        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def normalize_label(label: Any) -> str:
    if label is None:
        return ""

    label = str(label).strip().upper()
    return LABEL_ALIASES.get(label, label)


def parse_gold_spans(value: Any) -> list[dict]:
    """Parse the unified_span_labels JSON column."""
    if value is None:
        return []

    if isinstance(value, float):
        return []

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse unified_span_labels: {value[:200]}"
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError("Expected unified_span_labels to be a JSON list.")

    spans = []

    for item in parsed:
        if not isinstance(item, dict):
            continue

        start = int(item["start"])
        end = int(item["end"])
        label = normalize_label(item["label"])
        span_text = str(item.get("text", ""))

        if start < 0 or end <= start:
            raise ValueError(f"Invalid gold span: {item}")

        spans.append({
            "start": start,
            "end": end,
            "label": label,
            "text": span_text,
        })

    return spans


def span_key(span: dict) -> tuple[int, int, str]:
    return (
        int(span["start"]),
        int(span["end"]),
        normalize_label(span["label"]),
    )


def load_test_rows() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n\n"
            "Check DATASET_PATH and make sure you are running "
            "the script from the correct project."
        )

    rows = []

    with DATASET_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "id",
            "language",
            "source_text",
            "unified_span_labels",
            "set",
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            split = str(row.get("set", "")).strip().lower()

            if split != "test":
                continue

            text = row.get("source_text") or ""

            rows.append({
                "id": row.get("id", ""),
                "language": (row.get("language") or "unknown").strip().lower(),
                "text": text,
                "gold": parse_gold_spans(
                    row.get("unified_span_labels")
                ),
            })

    if not rows:
        raise ValueError(
            "No rows with set='test' were found in the CSV."
        )

    return rows


# ============================================================
# 3. RUN THE END-TO-END DETECTOR
# ============================================================

def predict_entities(detector, text: str) -> list[dict]:
    """
    Evaluate the public detector interface, not just the raw model.
    Expected interface: detector.detect(text).
    """
    result = detector.detect(text)

    entities = get_field(result, "entities", default=None)

    if entities is None:
        raise ValueError(
            "detector.detect(text) did not return an 'entities' field. "
            f"Returned type: {type(result).__name__}"
        )

    predictions = []

    for entity in entities:
        start = get_field(entity, "start")
        end = get_field(entity, "end")
        label = get_field(
            entity, "entity_type", "label", "entity"
        )
        entity_text = get_field(entity, "text", default="")

        if start is None or end is None or label is None:
            raise ValueError(
                "Unexpected detected entity format: "
                f"{entity!r}"
            )

        predictions.append({
            "start": int(start),
            "end": int(end),
            "label": normalize_label(label),
            "text": str(entity_text or ""),
        })

    return predictions


# ============================================================
# 4. EXACT-SPAN METRICS
# ============================================================

def calculate_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_group(records: list[dict]) -> dict:
    """
    Exact match requires start, end, and entity label to match.
    Matching is one-to-one; duplicate predictions are not double-counted.
    """
    tp = fp = fn = 0
    per_label = defaultdict(Counter)
    errors = []

    for record in records:
        gold_by_key = Counter(
            span_key(span) for span in record["gold"]
        )
        pred_by_key = Counter(
            span_key(span) for span in record["predictions"]
        )

        matched = gold_by_key & pred_by_key
        record_tp = sum(matched.values())
        record_fp = sum((pred_by_key - gold_by_key).values())
        record_fn = sum((gold_by_key - pred_by_key).values())

        tp += record_tp
        fp += record_fp
        fn += record_fn

        for key, count in matched.items():
            per_label[key[2]]["tp"] += count

        for key, count in (pred_by_key - gold_by_key).items():
            label = key[2]
            per_label[label]["fp"] += count

            if len(errors) < MAX_ERROR_EXAMPLES:
                errors.append({
                    "error_type": "FALSE_POSITIVE",
                    "id": record["id"],
                    "language": record["language"],
                    "text": record["text"],
                    "start": key[0],
                    "end": key[1],
                    "label": label,
                    "span_text": next(
                        (
                            p["text"]
                            for p in record["predictions"]
                            if span_key(p) == key
                        ),
                        "",
                    ),
                })

        for key, count in (gold_by_key - pred_by_key).items():
            label = key[2]
            per_label[label]["fn"] += count

            if len(errors) < MAX_ERROR_EXAMPLES:
                errors.append({
                    "error_type": "FALSE_NEGATIVE",
                    "id": record["id"],
                    "language": record["language"],
                    "text": record["text"],
                    "start": key[0],
                    "end": key[1],
                    "label": label,
                    "span_text": next(
                        (
                            g["text"]
                            for g in record["gold"]
                            if span_key(g) == key
                        ),
                        "",
                    ),
                })

    label_metrics = {}

    for label, counts in sorted(per_label.items()):
        label_metrics[label] = calculate_metrics(
            counts["tp"],
            counts["fp"],
            counts["fn"],
        )

    return {
        "overall": calculate_metrics(tp, fp, fn),
        "per_entity_type": label_metrics,
        "errors": errors,
    }


# ============================================================
# 5. OUTPUT REPORTS
# ============================================================

def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Import after setting up paths so import errors are easier to diagnose.
    sys.path.insert(0, str(PROJECT_ROOT))

    from app.sensitive import SensitiveDataDetector

    print("=" * 72)
    print("SENSITIVE DATA DETECTOR - END-TO-END EVALUATION")
    print("=" * 72)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_PATH}")

    rows = load_test_rows()
    print(f"Test samples: {len(rows)}")

    detector = SensitiveDataDetector()

    records = []
    prediction_rows = []
    inference_errors = []

    for index, row in enumerate(rows, start=1):
        try:
            predictions = predict_entities(detector, row["text"])

            record = {
                **row,
                "predictions": predictions,
            }
            records.append(record)

            for prediction in predictions:
                prediction_rows.append({
                    "id": row["id"],
                    "language": row["language"],
                    "text": row["text"],
                    **prediction,
                })

        except Exception as exc:
            inference_errors.append({
                "id": row["id"],
                "language": row["language"],
                "text": row["text"],
                "error": repr(exc),
            })

        if index % 100 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)}")

    if not records:
        raise RuntimeError(
            "No samples were evaluated successfully. "
            "Check detector errors and model loading."
        )

    overall = evaluate_group(records)

    languages = sorted({
        record["language"] for record in records
    })

    language_metrics = {}

    for language in languages:
        subset = [
            record for record in records
            if record["language"] == language
        ]
        language_metrics[language] = {
            "samples": len(subset),
            **evaluate_group(subset)["overall"],
        }

    # Macro F1 across entity types that appear in gold or predictions.
    macro_f1 = (
        sum(
            item["f1"]
            for item in overall["per_entity_type"].values()
        )
        / len(overall["per_entity_type"])
        if overall["per_entity_type"]
        else 0.0
    )

    summary = {
        "evaluation": "end_to_end_detector",
        "matching": "exact start + end + entity type",
        "dataset": str(DATASET_PATH),
        "requested_test_samples": len(rows),
        "evaluated_samples": len(records),
        "failed_samples": len(inference_errors),
        "overall": overall["overall"],
        "entity_type_macro_f1": macro_f1,
        "per_entity_type": overall["per_entity_type"],
        "per_language": language_metrics,
        "notes": [
            "Only rows whose set column equals 'test' were evaluated.",
            "A true positive requires exact span boundaries and entity type.",
            "Failed inference samples are excluded from metric denominators; "
            "review inference_errors.csv before interpreting results.",
            "These are detector-level results, not raw token-classification metrics.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_entity_rows = [
        {"entity_type": label, **metrics}
        for label, metrics in overall["per_entity_type"].items()
    ]

    write_csv(
        OUTPUT_DIR / "per_entity_type.csv",
        per_entity_rows,
        ["entity_type", "tp", "fp", "fn", "precision", "recall", "f1"],
    )

    per_language_rows = [
        {"language": language, **metrics}
        for language, metrics in language_metrics.items()
    ]

    write_csv(
        OUTPUT_DIR / "per_language.csv",
        per_language_rows,
        ["language", "samples", "tp", "fp", "fn", "precision", "recall", "f1"],
    )

    write_csv(
        OUTPUT_DIR / "predictions.csv",
        prediction_rows,
        ["id", "language", "text", "start", "end", "label", "span_text"],
    )

    write_csv(
        OUTPUT_DIR / "errors.csv",
        overall["errors"],
        [
            "error_type", "id", "language", "text",
            "start", "end", "label", "span_text",
        ],
    )

    write_csv(
        OUTPUT_DIR / "inference_errors.csv",
        inference_errors,
        ["id", "language", "text", "error"],
    )

    print("\n" + "=" * 72)
    print("OVERALL RESULTS")
    print("=" * 72)
    print(f"Evaluated samples : {len(records)}")
    print(f"Failed samples    : {len(inference_errors)}")
    print(f"TP                : {overall['overall']['tp']}")
    print(f"FP                : {overall['overall']['fp']}")
    print(f"FN                : {overall['overall']['fn']}")
    print(f"Precision         : {overall['overall']['precision']:.4f}")
    print(f"Recall            : {overall['overall']['recall']:.4f}")
    print(f"F1                : {overall['overall']['f1']:.4f}")
    print(f"Entity Macro F1   : {macro_f1:.4f}")

    print("\nPer entity type:")
    for label, metrics in overall["per_entity_type"].items():
        print(
            f"{label:28s} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"(TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})"
        )

    print("\nPer language:")
    for language, metrics in language_metrics.items():
        print(
            f"{language:12s} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"samples={metrics['samples']}"
        )

    print(f"\nReports saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
