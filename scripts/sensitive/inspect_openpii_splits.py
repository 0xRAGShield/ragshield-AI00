
from datasets import load_dataset

DATASET_NAME = "ai4privacy/pii-masking-300k"

for split in ["train", "validation"]:
    print(f"\n{'=' * 15} {split.upper()} {'=' * 15}")

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
    )

    print("Total rows:", len(dataset))
    print("Columns:", dataset.column_names)

    if "language" in dataset.column_names:
        from collections import Counter

        languages = Counter(dataset["language"])
        print("Languages:")
        for language, count in languages.most_common():
            print(f"  {language}: {count:,}")

    if "set" in dataset.column_names:
        print("Set values:", Counter(dataset["set"]))

    print("First example language:", dataset[0].get("language"))