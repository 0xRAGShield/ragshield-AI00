
import json
from pathlib import Path

root = Path(__file__).resolve().parent
dataset_path = root / "evaluation" / "datasets" / "evaluation_template.jsonl"

text = dataset_path.read_text(encoding="utf-8-sig")

# Remove comment lines and keep the JSON objects.
text = "\n".join(
    line for line in text.splitlines()
    if not line.lstrip().startswith("#")
)

decoder = json.JSONDecoder()
rows = []
position = 0

while position < len(text):
    # Skip whitespace between JSON objects.
    while position < len(text) and text[position].isspace():
        position += 1

    if position >= len(text):
        break

    if text[position] != "{":
        raise ValueError(
            f"Unexpected content at character {position}: "
            f"{text[position:position + 100]!r}"
        )

    obj, end_position = decoder.raw_decode(text, position)
    rows.append(obj)
    position = end_position

print("Number of JSON records:", len(rows))
print("Record IDs:", [row.get("query_id") for row in rows])

if rows:
    print("\nFields in first record:")
    print(json.dumps(list(rows[0].keys()), ensure_ascii=False, indent=2))

print("\nAnnotation summary:")
for row in rows:
    judgments = row.get("relevance_judgments", [])
    claims = row.get("claim_annotations", [])

    print(f"\nQuery: {row.get('query_id')}")
    print("  Query:", row.get("query"))
    print("  Relevant chunks:", len(row.get("relevant_chunk_ids", [])))
    print("  Relevance judgments:", len(judgments))
    print("  Key facts:", len(row.get("key_facts", [])))
    print("  Claim annotations:", len(claims))

    labels = {}
    for claim in claims:
        label = claim.get("label", "MISSING")
        labels[label] = labels.get(label, 0) + 1

    print("  Claim labels:", labels)

print("\nGeneration human-rating fields:")
rating_fields = [
    "human_correctness",
    "human_faithfulness",
    "human_relevance",
    "human_completeness",
    "human_abstention",
    "human_medical_safety",
]

for row in rows:
    print(f"\n{row.get('query_id')}:")
    for field in rating_fields:
        print(f"  {field}: {row.get(field, 'NOT PRESENT')}")
