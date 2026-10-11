# Sensitive Data Detection

## 1. Overview

The Sensitive Data Detection Layer is a standalone bilingual PII detection module developed as part of RAGShield.

Its primary purpose is to identify sensitive information in text and return the detected entities, their canonical labels, character offsets, and confidence scores when available.

The module is designed to operate independently of the RAG retrieval and generation pipeline, allowing it to be tested and integrated separately.

### Main Capabilities

- Bilingual sensitive entity detection for Arabic and English.
- Named Entity Recognition (NER) using a fine-tuned XLM-RoBERTa model.
- Rule-based detection using regular expressions and contextual patterns.
- Character-level span alignment and validation.
- Handling of overlapping entity predictions.
- Local inference without downloading model weights at runtime.
- Docker-based distribution.

## 2. Model Architecture

The detection model is based on XLM-RoBERTa and is fine-tuned for token classification.

- **Model directory:** `models/sensitive/bilingual_pii_xlmr`
- **Architecture:** XLM-RoBERTa Token Classification (NER)
- **Supported languages:** Arabic and English
- **Label configuration:** 35 BIO labels representing 17 entity types, including `O`.

The BIO tagging scheme identifies entity boundaries:

- `B-`: Beginning of an entity.
- `I-`: Continuation of an entity.
- `O`: Token outside the target entity types.

The model, tokenizer, and configuration are loaded from local files. The runtime does not download model weights from Hugging Face.

The canonical label mapping is maintained at:

`data/sensitive/processed/combined/label_mapping.json`

The model configuration, tokenizer, classifier weights, and canonical label mapping must remain compatible.

## 3. Training Data

The bilingual model was prepared using general PII data and Arabic PII data.

### General PII Dataset

General PII data was used to provide examples of sensitive entities across common categories.

### SITR Arabic PII

The `mabahboh/sitr-arabic-pii` dataset was used to strengthen Arabic PII coverage. Selected categories include national IDs, passport numbers, phone numbers, email addresses, addresses, passwords, and other supported sensitive information.

The source labels were mapped into a unified entity-label vocabulary.

### Combined Dataset Splits

The recorded combined dataset contains:

| Split           |         Examples |
| --------------- | ---------------: |
| Training        |           34,032 |
| Validation      |            8,849 |
| Test            |            3,890 |
| **Total** | **46,771** |

These counts describe the prepared dataset splits. Model performance must be evaluated separately using metrics measured on the test set.

## 4. Detection Pipeline

The detection process consists of the following stages:

1. **Input text:** Receive the text to be analyzed.
2. **NER inference:** Use XLM-RoBERTa to predict entity labels and token offsets.
3. **Rule-based detection:** Apply regular-expression and contextual rules for supported sensitive patterns.
4. **Span validation:** Validate entity boundaries against the original text.
5. **Overlap handling:** Resolve overlapping predictions according to the detection logic.
6. **Result generation:** Return detected entities with their labels, character offsets, and confidence information when available.

The model and rule-based component complement each other. Rules can identify recognizable patterns, while the NER model learns contextual and linguistic patterns.

Detection results do not guarantee that an identifier is valid or belongs to a particular person.

## 5. Supported Entity Categories

The canonical label vocabulary includes categories such as:

- `PERSON`
- `EMAIL`
- `PHONE`
- `ADDRESS`
- `NATIONAL_ID`
- `PASSPORT_NUMBER`
- `EMPLOYEE_ID`
- `CREDIT_CARD`
- `MEDICAL_RECORD_NUMBER`
- `PRIVATE_DATE`
- `PASSWORD`
- `SPOKEN_NATIONAL_ID`

The complete label vocabulary is defined by the canonical label mapping.

Actual detection performance may vary by entity category, language, input format, and surrounding context.

## 6. Configuration and Runtime

The detector supports configuration for:

- Model path.
- Device selection: CPU, CUDA, or automatic selection.
- Confidence threshold.
- Maximum token sequence length.
- Shared model reuse to avoid repeatedly loading the same weights.

The current Docker image runs inference on CPU by default.

The local runtime uses the packaged model assets. No model download is required during normal inference when the required files are present.

## 7. Testing and Validation

The implementation has been tested through unit tests and local-model inference checks.

### Unit Tests

Run from the project root:

    python -m pytest tests/sensitive -q

Latest recorded result:

- 22 tests passed.
- 3 tests skipped.

Skipped tests are not counted as passed and should be reviewed to determine whether additional validation is required.

### Local Model Validation

The local-model smoke test confirmed that:

- The local model loads successfully.
- The model exposes 35 labels.
- The model runs on CPU.
- An English sample containing a person's name and email address produced the expected entity types.

### Docker Validation

The Docker smoke test confirmed that:

- The model loads from the image's local model directory.
- Inference runs without downloading the model.
- A sample input identifies `Ahmed` as `PERSON` and `ahmed.test@example.com` as `EMAIL`.

These smoke tests validate basic model loading and inference. They do not replace a complete evaluation using precision, recall, F1-score, and per-entity metrics.

## 8. Docker Hub

Pull the Sensitive Data Detection image from Docker Hub:

    docker pull manarabdelbaky16/ragshield-sensitive:v1

Run the container:

    docker run --rm manarabdelbaky16/ragshield-sensitive:v1

The current image runs a model-loading and inference smoke test, then exits. It does not expose a persistent API.

If the repository is private, users must authenticate and have permission to access it before pulling the image.

## 9. Module Structure

The core implementation is located in `Security_layer/app/sensitive/`.

- `detector.py` — Main detection interface and orchestration.
- `model.py` — XLM-RoBERTa model loading and token-classification inference.
- `labels.py` — Label vocabulary, mapping validation, and model-path resolution.
- `offsets.py` — Character-offset alignment helpers.
- `schemas.py` — Detection result and entity data structures.
- `rules.py` — Rule-based sensitive pattern detection.
- `__init__.py` — Public module exports.

## 10. Current Status and Next Steps

### Completed

- Bilingual PII dataset preparation and label harmonization.
- Fine-tuned XLM-RoBERTa model integration.
- Local model loading and inference validation.
- Rule-based detection integration.
- Unit-test execution.
- Docker image build and local smoke testing.

### Next Steps

- Evaluate precision, recall, and F1-score on the held-out test set.
- Measure performance separately for Arabic and English.
- Evaluate per-entity detection performance and false positives.
- Review the three skipped tests.
- Complete integration with the RAGShield security workflow.

The current implementation provides a locally executable, containerized PII detection component. Further evaluation and integration testing are required before considering it production-read

# Sensitive Data Detection

Standalone bilingual NER module for detecting PII in text chunks. It is intentionally independent from the RAG retrieval/generation pipeline.

## Quick start

From `Security_layer/`:

```python
from app.sensitive import SensitiveDataDetector

detector = SensitiveDataDetector()
result = detector.detect("My name is Ahmed Mohamed and my email is ahmed@gmail.com.")

for entity in result.entities:
    print(entity.text, entity.label, entity.start, entity.end, entity.score)
```

Each entity includes:

- `text` — detected span text
- `label` / `entity` — canonical sensitive label (`PERSON`, `EMAIL`, `PHONE`, ...)
- `start` / `end` — character offsets in the original input (`text[start:end] == entity.text`)
- `score` / `confidence` — model confidence

## Model

Default checkpoint (local only):

`models/sensitive/bilingual_pii_xlmr/bilingual_pii_xlmr`

- Architecture: XLM-RoBERTa token classification (`Davlan/xlm-roberta-base-ner-hrl` fine-tune)
- Languages: Arabic + English (SITR Arabic PII + OpenPII English)
- 35 BIO labels / 17 entity types
- Canonical mapping: `data/sensitive/processed/combined/label_mapping.json`

The runtime loads this directory with `local_files_only=True`. It does not download the base model.

If `config.json` is missing next to `model.safetensors`, the loader uses a compatible `checkpoint-*/config.json` whose `id2label` matches the canonical 35-label mapping.

## Configuration

```python
from app.sensitive import SensitiveDataDetector, SensitiveDataDetectorConfig

detector = SensitiveDataDetector(
    config=SensitiveDataDetectorConfig(
        model_name_or_path="models/sensitive/bilingual_pii_xlmr/bilingual_pii_xlmr",
        device="auto",              # "cpu", "cuda", or "auto"
        confidence_threshold=0.5,
        max_length=512,
        reuse_shared_model=True,    # load once, reuse across instances
    )
)
```

## Tests

Unit tests (no model download):

```bash
cd Security_layer
pytest ../tests/sensitive/ -v
```

Local-model smoke tests:

```bash
cd Security_layer
set RUN_SENSITIVE_NER=1
pytest ../tests/sensitive/ -v
```

## Module layout

```text
app/sensitive/
├── detector.py   # SensitiveDataDetector public API
├── model.py      # XLM-RoBERTa backend + shared loader
├── schemas.py    # Pydantic result contracts
├── labels.py     # Label vocabulary + normalization
└── offsets.py    # Character offset alignment helpers
```
