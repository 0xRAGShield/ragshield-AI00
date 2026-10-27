
from __future__ import annotations

import json
from pathlib import Path

from ocr.config import DEFAULT_TROCR_CHECKPOINT_DIR
from ocr.trocr_client import missing_checkpoint_files


def assert_not_overwriting_source(output_dir: Path, source_dir: Path) -> None:
    output = output_dir.resolve()
    source = source_dir.resolve()
    if output == source or source in output.parents or output == DEFAULT_TROCR_CHECKPOINT_DIR.resolve():
        raise ValueError(
            f"Refusing to write fine-tuned weights into the source checkpoint {source}"
        )


def load_processor_and_model(source_dir: Path, local_files_only: bool = True):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    missing = missing_checkpoint_files(source_dir)
    if missing:
        raise FileNotFoundError(
            f"Incomplete TrOCR checkpoint at {source_dir}. Missing: {', '.join(missing)}"
        )
    processor = TrOCRProcessor.from_pretrained(
        str(source_dir),
        local_files_only=local_files_only,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        str(source_dir),
        local_files_only=local_files_only,
    )
    tokenizer = processor.tokenizer
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    model.config.use_cache = False
    return processor, model


def save_finetuned(
    output_dir: Path,
    model,
    processor,
    train_config: dict,
    extra: dict | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    payload = {
        "train_config": train_config,
        "source_model": train_config.get("model_id"),
        "source_checkpoint_dir": train_config.get("source_checkpoint_dir"),
        "note": (
            "Fine-tuned TrOCR weights. The application default remains "
            "microsoft/trocr-base-handwritten until TROCR_CHECKPOINT_DIR or "
            "--trocr-checkpoint points here."
        ),
    }
    if extra:
        payload.update(extra)
    (output_dir / "finetune_meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_dir
