# RAGShield OCR

Local, offline OCR toolkit for handwritten documents in Arabic and English.
Originally built for medical prescriptions; it also works on general papers.

- **Recognition:** `sherif1313/Arabic-English-handwritten-OCR-v3` (Qwen2.5-VL, Arabic + English) and,
  optionally, English-only `microsoft/trocr-base-handwritten` for line crops.
- **Preprocessing:** grayscale, denoise, CLAHE, illumination correction, deskew, binarisation variants.
- **Hybrid mode:** detect text lines, route each line to an engine, merge in reading order.
- **Tools:** CLI, a small local upload page, CER/WER evaluation, TrOCR fine-tuning scaffolding.

Nothing is sent to an external service. Model files are read from a local folder.

> OCR output is a draft. For prescriptions and other safety-critical documents a person must check every
> line against the image.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                         # image processing + tests
# Model inference (CPU build; for GPU install the CUDA build of torch instead):
pip install -r requirements-local.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Download the checkpoints once:

```bash
python scripts/download_models.py            # both models
python scripts/download_models.py --only qwen
```

Models go to `OCR_MODELS_DIR` (or `HANDWRITTEN_OCR_CACHE`) if set, otherwise
`~/.cache/ragshield-ocr/hf_cache`. Copy `.env.example` to `.env` to change settings.

## Usage

```bash
python -m ocr preprocess page.jpg --out outputs/prep            # write preprocessing variants
python -m ocr transcribe page.jpg --out outputs/run             # Qwen, default
python -m ocr transcribe page.jpg --out outputs/run --engine hybrid
python -m ocr run-images --input data/images --out outputs/batch
python -m ocr serve                                             # http://127.0.0.1:8765
python -m ocr build-dataset --out data/eval                     # synthetic evaluation set
python -m ocr evaluate --predictions preds.json --out outputs/eval
```

Single-image helpers live in `scripts/` (`ocr_single_image.py`, `enhance_image.py`, `download_models.py`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite uses fakes for the models, so it runs without `torch` or any downloaded weights.

## Layout

| Path | Purpose |
| --- | --- |
| `ocr/preprocessing.py` | image validation, variants, deskew, illumination |
| `ocr/line_detection.py` | classical line segmentation |
| `ocr/hybrid.py`, `ocr/routing.py` | per-line engine routing and merge |
| `ocr/handwritten_ocr_client.py` | Qwen2.5-VL client |
| `ocr/trocr_client.py` | English TrOCR client |
| `ocr/evaluation.py` | CER / WER |
| `ocr/server.py` | local upload page |
| `ocr/trocr_finetune/` | TrOCR fine-tuning (English medication-name data only) |
| `docs/TROCR_FINETUNE.md` | fine-tuning notes |

## Data privacy

`data/`, `dataset/` and `outputs/` are git-ignored on purpose. Do not commit real patient documents.
