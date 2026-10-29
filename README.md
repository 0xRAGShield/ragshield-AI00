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

## What the pipeline does to a page

1. **Page crop and flatten** (photos): the sheet of paper is cut out of the photo and its perspective is
   corrected. Conservative: only when the sheet is clearly brighter than its surroundings, otherwise untouched.
2. **Orientation:** pages scanned sideways are detected (tilt-robust) and turned. Which way to turn is
   decided by recognizer confidence when a model is available; without one the turn is a guess and is reported.
   `--check-upside-down` also compares 0° and 180°.
3. **Deskew** up to ±25°, then optional upscaling for small images only.
4. **Line detection:** thin ruling lines and table borders are removed, the grouping kernel follows the
   text size, regions that swallowed several lines are split, and boxes are grouped into rows.
5. **Recognition:** original-colour crops (aligned to the processed page) go to the model, which also reports
   token confidence. Pages too large for one image are sent as bands cut at blank rows.
6. **Merge:** rows top to bottom; inside a row left to right, or right to left when the text is mostly Arabic
   (`--reading-direction auto|ltr|rtl`).

Every line carries `confidence` and `review_reasons` (failed or empty, low confidence, numbers or dosage,
`[illegible]`, repetition, engine fallback). The page as a whole is always flagged `review_required`.
`--min-confidence` (default 0.75) is an **uncalibrated** starting point: pick yours on a labelled set.
Confidence is the model's own token probability, not an accuracy.

Useful switches: `--prompt-mode general|prescription`, `--ocr-source aligned|processed`,
`--max-image-side`, `--no-tiling`, `--no-page-crop`, `--no-orientation`, `--review-policy risk|all`,
`--trocr-line-prep pad_square|stretch|chunk`.

## Finding out why pages fail

```bash
python -m ocr triage --input data/images --out outputs/triage            # no model needed
python -m ocr triage --input data/images --out outputs/triage --run-ocr  # adds confidence and CER
python scripts/compare_triage.py outputs/triage_before outputs/triage_after
```

To measure whether a change helped, run the same folder through the old and the new code and diff the results.
This works against the original `OCR` branch too:

```bash
python -m ocr run-images --input data/images --out outputs/before --variant original   # on the old branch
python -m ocr run-images --input data/images --out outputs/after  --variant original   # on this branch
python scripts/compare_run_images.py outputs/before outputs/after
```

Put a `<image>.txt` with the correct text next to an image to get its CER. Triage writes `triage.csv`,
`triage_summary.md` (images grouped as `input_quality`, `detection`, `recognition`) and an overlay image per page
with the detected boxes. The thresholds are rules of thumb. Open the overlays and judge for yourself.

## Other engines and other Qwen-VL models

Engines share one small interface (`ocr/engines/`): a page image in, text, optional line boxes and an
optional confidence out.

```bash
python -m ocr engines                                        # what is installed and ready
python -m ocr compare-engines --input data/images --out outputs/engines --engines qwen,paddleocr,surya
python -m ocr compare-engines ... --preprocess aligned       # give every engine this project's cropped, turned, deskewed page
```

Put `<image>.txt` (the correct text) next to each image to get error rates. The summary ranks engines by
`cer_all`, which counts a failed page as completely wrong, so an engine cannot look good by skipping hard pages.
Confidence means different things per engine (token probability, recognition score): compare it within an engine only.

| engine | install | notes |
| --- | --- | --- |
| `qwen` | `requirements-local.txt` | default; handwriting, Arabic and English |
| `paddleocr` | `requirements-paddle.txt` | PaddleOCR 3.x, `--paddle-lang ar`; built mostly for printed text, so expect it to be weaker on handwriting. Verify on your pages. |
| `surya` | `requirements-surya.txt` | needs a vLLM (NVIDIA, Docker) or llama.cpp backend. **Model weights are free for research, personal use and small startups; broader commercial use needs a licence.** |

A cloud OCR API can be added by registering one more `Engine` (see `ocr/engines/qwen.py`). None is included.

**Other Qwen-VL checkpoints.** The client is no longer tied to one model:

```bash
python scripts/download_models.py --model-id Qwen/Qwen2.5-VL-7B-Instruct
export HANDWRITTEN_OCR_MODEL=Qwen/Qwen2.5-VL-7B-Instruct          # folder: <models dir>/Qwen__Qwen2.5-VL-7B-Instruct
export HANDWRITTEN_OCR_CHECKPOINT_DIR=/path/to/any/checkpoint     # or point at a folder directly
export HANDWRITTEN_OCR_ATTN=sdpa                                  # optional: sdpa or flash_attention_2
export HANDWRITTEN_OCR_MAX_PIXELS=1003520                         # optional: processor resize limit
```

A general model is not fine-tuned for Arabic handwriting. Whether it beats the default is a question for
`compare-engines` on your own pages, not an assumption. Weights are still read from disk only; nothing is downloaded at run time.

## GPU

`HANDWRITTEN_OCR_DEVICE=auto` uses CUDA when torch can see it (bfloat16, or float16 on older GPUs) and falls
back to CPU. Install the CUDA build of torch instead of the CPU one in `requirements-local.txt`.

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
| `ocr/page_crop.py`, `ocr/orientation.py` | find the paper in a photo; detect sideways pages |
| `ocr/line_detection.py` | classical line segmentation, rows and reading order |
| `ocr/tiling.py` | split large pages and long lines at blank rows |
| `ocr/confidence.py` | token-probability confidence |
| `ocr/triage.py` | per-page diagnosis of failures |
| `ocr/engines/`, `ocr/engine_benchmark.py` | pluggable engines (Qwen, PaddleOCR, Surya) and side-by-side comparison |
| `ocr/hybrid.py`, `ocr/routing.py` | per-line engine routing and merge |
| `ocr/handwritten_ocr_client.py` | Qwen2.5-VL client |
| `ocr/trocr_client.py` | English TrOCR client |
| `ocr/evaluation.py` | CER / WER |
| `ocr/server.py` | local upload page |
| `ocr/trocr_finetune/` | TrOCR fine-tuning (English medication-name data only) |
| `docs/TROCR_FINETUNE.md` | fine-tuning notes |

## Data privacy

`data/`, `dataset/` and `outputs/` are git-ignored on purpose. Do not commit real patient documents.
