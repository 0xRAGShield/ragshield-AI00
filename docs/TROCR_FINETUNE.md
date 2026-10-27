# TrOCR fine-tuning (static notes)

This document describes a **prepared** pipeline. Nothing in this file was executed when the pipeline was added. Commands below are for later use.

Qwen2.5-VL (`sherif1313/Arabic-English-handwritten-OCR-v3`) is **not** trained here and its default application path is unchanged.

## 1. Dataset format found in the repository

`dataset/RxHandBD-ML` contains four items:

| Path | Role |
| --- | --- |
| `dataset/RxHandBD-ML/Train_Set/` | Official training images |
| `dataset/RxHandBD-ML/Test_Set/` | Official test images (held out) |
| `dataset/RxHandBD-ML/Train_Label.csv` | Columns `Images`, `Text` — train filenames and Ground Truth (`P1116.jpg` … `P5578.jpg`, 4463 data rows by line count) |
| `dataset/RxHandBD-ML/Test_Label.csv` | Columns `Images`, `Text` — test filenames and Ground Truth (`P0001.jpg` … `P1115.jpg`, 1115 data rows by line count) |

Filename mapping (Windows paths, relative to the project root):

- Train CSV `P1116.jpg` → `dataset/RxHandBD-ML/Train_Set/P1116.jpg` (spot-checked; label `Dexter`)
- Test CSV `P0001.jpg` → `dataset/RxHandBD-ML/Test_Set/P0001.jpg` (spot-checked; label `Nexcital`)

The loader looks up each CSV `Images` value only in the matching split directory. It matches the exact filename, then a case-insensitive name, then a case-insensitive stem so `.jpg` / `.JPG` (and other on-disk suffixes) can resolve without assuming one extension. Train rows are never resolved against `Test_Set`. Test rows are never resolved against `Train_Set`. Relative config paths are resolved from the OCR project root (`ocr/trocr_finetune/dataset.py` → two parents), not from the process working directory.

Labels are handwritten **medication-name** strings in Latin script, with digits and punctuation (examples: `Dexter`, `m-lucas 10`, `5% dns`, `T-Cef`). A static search of the CSVs found **no Arabic-script characters**. A few test/raw rows use placeholders such as `???`, `??r??`, `?icrogent`.

`dataset/RxHandBD-Raw/Prescription_Labels.csv` remains an optional alternate label list. It is not the default ML source.

The original CSVs and images are not modified. Fine-tuned weights go under `outputs/trocr_finetune/`.

## 2. Is the dataset ready for supervised fine-tuning?

| Item | Status from static inspection |
| --- | --- |
| CSV columns | `Images`, `Text` (first two header cells of both ML CSVs) |
| Image folders | `Train_Set` and `Test_Set` exist as directories beside the CSVs |
| Spot-checked pairs | `Train_Set/P1116.jpg` ↔ `Dexter`; `Test_Set/P0001.jpg` ↔ `Nexcital` |
| Every image-label pair | **Not verified.** No full directory listing or validation script was run |
| Reliable Ground Truth | Human CSV labels for medication names. Placeholders (`???`) are flagged, not invented, and not dropped unless `--drop-uncertain` |
| Language labels | None in the CSVs |
| Arabic GT | None found in labels |

Model predictions are never used as Ground Truth.

## 3. Exact TrOCR checkpoint

| Item | Value from repository / checkpoint files |
| --- | --- |
| Hub id | `microsoft/trocr-base-handwritten` (`ocr/config.py` `TROCR_MODEL_ID`) |
| Local directory | `D:\OCR_Models\hf_cache\microsoft__trocr-base-handwritten` |
| Architecture | `VisionEncoderDecoderModel` (`config.json`) |
| Encoder | ViT, `image_size` 384, `patch_size` 16 |
| Processor | `TrOCRProcessor` + `ViTImageProcessor` (`do_resize`, `do_normalize`, mean/std 0.5) |
| Tokenizer | `RobertaTokenizer` (`tokenizer_config.json`, `name_or_path`: `roberta-large`) |
| Vocab size | 50265 (`config.json` decoder) |
| Special ids | `pad_token_id` 1, `eos` / `decoder_start_token_id` 2, `bos_token_id` 0 (`generation_config.json`) |
| Classes used at inference | `TrOCRProcessor`, `VisionEncoderDecoderModel` (`ocr/trocr_client.py`) |

Fine-tuned weights are written only under `outputs/trocr_finetune/...`. The source directory is not overwritten.

## 4. Arabic and multilingual tokenizer concerns

Static inspection only (the tokenizer was **not** loaded):

- The checkpoint is the English IAM handwritten line model used by `TrocrClient`.
- The decoder tokenizer is RoBERTa BPE, not an Arabic OCR tokenizer.
- Encoding an image does not mean the decoder can emit Arabic.
- This pipeline **does not** replace the tokenizer or resize embeddings. Doing that would be a different project (new vocab, embedding init, much more data).
- RxHandBD labels inspected here are Latin-script drug names, so they are in-domain for this tokenizer’s *script*, not for IAM English prose. That is still a domain shift (prescription crops vs IAM lines).
- If Arabic GT were added later, byte-level BPE might still tokenize it, but the decoder would not have learned Arabic handwriting. Treat Arabic OCR with this checkpoint as unsupported.

## 5. Dependencies (declared, not re-installed here)

Already listed in `requirements-local.txt` / `requirements.txt`:

- `torch`, `torchvision`, `transformers>=4.57.1`, `accelerate`
- `pillow`, `numpy`

Not added: `peft` / LoRA. Encoder-decoder LoRA would need an extra package and extra wiring; it is omitted.

`fp16` / `bf16` default to false. This repository’s README describes a CPU machine; those flags are optional and **unverified**.

## 6. How to launch later (do not run these as part of this task)

Prepare split + data-quality JSON (no model load):

```text
python -m ocr prepare-trocr-dataset --config configs/trocr_finetune.json --out outputs/trocr_finetune/prepare
```

Train (after images exist):

```text
python -m ocr finetune-trocr --config configs/trocr_finetune.json --output-dir outputs/trocr_finetune/run
```

Resume:

```text
python -m ocr finetune-trocr --config configs/trocr_finetune.json --resume-from-checkpoint outputs/trocr_finetune/run/hf_trainer/checkpoint-XXXX
```

Equivalent module entry points:

```text
python -m ocr.trocr_finetune.train --config configs/trocr_finetune.json
python -m ocr.trocr_finetune.evaluate --original-checkpoint D:\OCR_Models\hf_cache\microsoft__trocr-base-handwritten --finetuned-checkpoint outputs/trocr_finetune/run/best --out outputs/trocr_finetune/eval
```

## 7. Where outputs go

| Path | Contents |
| --- | --- |
| `outputs/trocr_finetune/run/split_manifest.json` | All records, issues, assigned splits |
| `outputs/trocr_finetune/run/data_quality_report.json` | Counts and tokenizer notes |
| `outputs/trocr_finetune/run/train_config.json` | Copy of settings used |
| `outputs/trocr_finetune/run/train_log.jsonl` | Trainer logs |
| `outputs/trocr_finetune/run/best/` | Selected checkpoint + processor + `finetune_meta.json` |
| `outputs/trocr_finetune/run/last/` | Final epoch |
| `outputs/trocr_finetune/run/processor/` | Processor/tokenizer copy |
| `outputs/trocr_finetune/eval/trocr_eval.json` | CER/WER and per-sample strings |
| `outputs/trocr_finetune/eval/trocr_eval.csv` | Same rows as CSV |

The default application checkpoint remains `D:\OCR_Models\hf_cache\microsoft__trocr-base-handwritten`.

To **opt in** to a fine-tuned directory without changing Qwen:

```text
set TROCR_CHECKPOINT_DIR=outputs/trocr_finetune/run/best
```

or:

```text
python -m ocr transcribe path\to\image.png --out outputs\one --engine trocr --crops path\to\crops --trocr-checkpoint outputs/trocr_finetune/run/best
```

## 8. Compare original vs fine-tuned

Evaluation uses the **same official test IDs** (`Test_Label.csv`), the same `pad_line_keep_aspect` + processor preprocessing as `TrocrClient`, and greedy decoding. Scoring uses `ocr.evaluation` CER/WER (NFKC / whitespace collapse **only for the metric copy**; saved references stay as loaded from CSV, aside from optional `strip_text` at dataset load).

```text
python -m ocr evaluate-trocr --original-checkpoint D:\OCR_Models\hf_cache\microsoft__trocr-base-handwritten --finetuned-checkpoint outputs/trocr_finetune/run/best --out outputs/trocr_finetune/eval --split-manifest outputs/trocr_finetune/run/split_manifest.json
```

Qwen is not included in this comparison.

## 9. Hardware and memory (estimates only)

These are **not** measured in this task:

- ViT-B/16 + TrOCR decoder in fp32 on CPU will be slow; batch size 2 and accumulation 4 are conservative guesses.
- CUDA/fp16 may fit a consumer GPU with small batches; VRAM was not probed.
- `dataloader_num_workers` defaults to 0 for Windows.

## 10. Training details

Order for each sample:

1. Load the file resolved from the CSV name (source files are not written).
2. Optional square pad (`pad_line_keep_aspect`), matching inference. The full-page `combined_gray` stack is **not** applied: RxHandBD items are already word/line crops, not full prescriptions.
3. `TrOCRProcessor` resize/normalize to 384.
4. `RobertaTokenizer` encodes the CSV text; pad positions are set to `-100` so they are ignored by the loss.
5. Hugging Face `Seq2SeqTrainer` trains `VisionEncoderDecoderModel`. Validation loss selects `best/`. Early stopping uses `early_stopping_patience` when a val split exists.

Split:

- Official test CSV + `Test_Set` are **never** used for training or early stopping.
- Official train CSV + `Train_Set` are hashed (`sha256(seed:filename)`) into train/val (`val_fraction_of_official_train`, default 0.10). Validation images still come from `Train_Set` only.
- Duplicate filenames keep the first row usable; later rows are reported.
- IDs that appear in the official test list are excluded from train/val.

Optional text changes (off unless enabled, except strip):

- `strip_text` default true (CSV trailing spaces such as `Zymef  `).
- `collapse_internal_whitespace` default false.
- `unicode_nfkc` default false.

## 11. Unresolved without execution

- Whether `Seq2SeqTrainingArguments` on the installed `transformers` build accepts every keyword (code falls back between `eval_strategy` / `evaluation_strategy` and `processing_class` / `tokenizer`).
- Whether every CSV filename has a matching file in `Train_Set` / `Test_Set` (only two files were opened read-only).
- Tokenizer encode/decode of RxHandBD strings (not run).
- Training loss, CER, or any accuracy claim (not run).
- CUDA availability (config default is `cpu`).
