
import json
from pathlib import Path
from collections import Counter

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "sensitive" / "processed" / "combined"

with open(DATASET_PATH / "label_mapping.json", encoding="utf-8") as f:
    mapping = json.load(f)

labels = mapping["labels"]
id2label = {int(k): v for k, v in mapping["id2label"].items()}

assert len(labels) == 35, f"Expected 35 labels, found {len(labels)}"
assert mapping["label2id"]["O"] == 0
assert len(set(labels)) == len(labels)

dataset = load_from_disk(str(DATASET_PATH))

total_errors = 0

for split_name, split in dataset.items():
    errors = Counter()
    label_counts = Counter()
    ignored_tokens = 0

    for row_idx, row in enumerate(split):
        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]
        tag_ids = row["labels"]

        if not (len(input_ids) == len(attention_mask) == len(tag_ids)):
            errors["length_mismatch"] += 1
            continue

        previous_entity = None

        for tag_id in tag_ids:
            if tag_id == -100:
                ignored_tokens += 1
                previous_entity = None
                continue

            if tag_id not in id2label:
                errors["invalid_label_id"] += 1
                previous_entity = None
                continue

            tag = id2label[tag_id]
            label_counts[tag] += 1

            if tag == "O":
                previous_entity = None
                continue

            prefix, entity = tag.split("-", 1)

            if prefix == "B":
                previous_entity = entity

            elif prefix == "I":
                if previous_entity != entity:
                    errors["invalid_bio_transition"] += 1

                previous_entity = entity

    total_split_errors = sum(errors.values())
    total_errors += total_split_errors

    print(f"\n {split_name.upper()} ")
    print(f"Examples: {len(split):,}")
    print(f"Ignored tokens (-100): {ignored_tokens:,}")
    print(f"BIO / ID / length errors: {total_split_errors:,}")

    if errors:
        print("Error details:", dict(errors))

    print("Top labels:")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count:,}")

print("\nFINAL")
print(f"Total errors: {total_errors:,}")
print("Mapping and dataset structure check completed.")