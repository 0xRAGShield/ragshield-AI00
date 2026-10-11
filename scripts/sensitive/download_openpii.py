
from pathlib import Path
from datasets import load_dataset
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "sensitive" / "openpii-300k"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading PII Masking 300K")

dataset = load_dataset(
    "ai4privacy/pii-masking-300k",
    split="train",
    streaming=True,
)

print("Dataset columns:", dataset.column_names)

english_examples = []
english_limit = 1000

for row in dataset:
    if str(row.get("language", "")).lower() in {
        "en", "english"
    }:
        english_examples.append(row)

        if len(english_examples) >= english_limit:
            break

if not english_examples:
    raise RuntimeError(
        "No English examples found. "
        "Inspect language values in the dataset."
    )

output_file = OUTPUT_DIR / "openpii_300k_english_sample.csv"

pd.DataFrame(english_examples).to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig",
)

print(f"Saved {len(english_examples)} English examples.")
print(f"File: {output_file}")
print("\nColumns:", list(english_examples[0].keys()))
print("\nFirst example:")
print(english_examples[0])