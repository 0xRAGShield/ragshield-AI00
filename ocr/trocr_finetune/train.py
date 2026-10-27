#Fine-tune microsoft/trocr-base-handwritten

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

from ocr.trocr_finetune.config import TrocrFinetuneConfig
from ocr.trocr_finetune.data import TrocrCollator, TrocrLineDataset
from ocr.trocr_finetune.dataset import load_rxhandbd
from ocr.trocr_finetune.modeling import (
    assert_not_overwriting_source,
    load_processor_and_model,
    save_finetuned,
)


def train(config: TrocrFinetuneConfig) -> dict:
    output_dir = Path(config.output_dir)
    source_dir = Path(config.source_checkpoint_dir)
    assert_not_overwriting_source(output_dir, source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.write_json(output_dir / "train_config.json")

    bundle = load_rxhandbd(config)
    bundle.write_manifest(output_dir / "split_manifest.json")
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(
            {
                "counts": bundle.counts,
                "tokenizer_notes": bundle.tokenizer_notes,
                "issue_codes": _count_codes(bundle.issues),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    train_rows = bundle.split_records("train")
    val_rows = bundle.split_records("val")
    if config.max_train_samples is not None:
        train_rows = train_rows[: config.max_train_samples]
    if config.max_eval_samples is not None:
        val_rows = val_rows[: config.max_eval_samples]

    summary = {
        "status": "blocked",
        "output_dir": str(output_dir),
        "counts": bundle.counts,
        "usable_train": len(train_rows),
        "usable_val": len(val_rows),
        "usable_test": len(bundle.split_records("test")),
        "tokenizer_notes": bundle.tokenizer_notes,
    }
    if not train_rows:
        summary["error"] = (
            "No usable training pairs. Typical cause: a Train_Label.csv "
            "filename did not resolve under Train_Set, or every remaining row "
            "was flagged empty/duplicate. Ground-truth CSV labels were not invented."
        )
        (output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _set_seed(config.seed)

    processor, model = load_processor_and_model(
        source_dir,
        local_files_only=config.local_files_only,
    )
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    import torch
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    device = _resolve_device(config.device, torch)
    model.to(device)

    train_dataset = TrocrLineDataset(train_rows, pad_to_square=config.pad_line_keep_aspect)
    eval_dataset = TrocrLineDataset(val_rows, pad_to_square=config.pad_line_keep_aspect)
    collator = TrocrCollator(processor, config.max_target_length)

    args_kwargs = dict(
        output_dir=str(output_dir / "hf_trainer"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        report_to="none",
        seed=config.seed,
        predict_with_generate=False,
        load_best_model_at_end=bool(val_rows),
        metric_for_best_model="eval_loss" if val_rows else None,
        greater_is_better=False if val_rows else None,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    args_kwargs.update(_strategy_kwargs(config, bool(val_rows)))
    training_args = Seq2SeqTrainingArguments(
        **{key: value for key, value in args_kwargs.items() if value is not None}
    )

    log_callback = _make_jsonl_callback(TrainerCallback, output_dir / "train_log.jsonl")
    callbacks = [log_callback]
    if val_rows and config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )

    trainer = _make_trainer(
        Seq2SeqTrainer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if val_rows else None,
        collator=collator,
        processor=processor,
        callbacks=callbacks,
    )

    resume = config.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=resume if resume else None)
    metrics = dict(train_result.metrics)
    if val_rows:
        metrics.update(trainer.evaluate())

    best_dir = output_dir / "best"
    last_dir = output_dir / "last"
    save_finetuned(last_dir, trainer.model, processor, config.to_dict(), extra={"kind": "last"})
    save_finetuned(best_dir, trainer.model, processor, config.to_dict(), extra={"kind": "best"})
    processor.save_pretrained(output_dir / "processor")

    summary.update(
        {
            "status": "completed",
            "metrics": metrics,
            "best_checkpoint": str(best_dir),
            "last_checkpoint": str(last_dir),
            "source_checkpoint_dir": str(source_dir),
            "device": device,
        }
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def prepare_only(config: TrocrFinetuneConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_rxhandbd(config)
    manifest = bundle.write_manifest(output_dir / "split_manifest.json")
    report = {
        "status": "prepared",
        "manifest": str(manifest),
        "counts": bundle.counts,
        "tokenizer_notes": bundle.tokenizer_notes,
        "issue_codes": _count_codes(bundle.issues),
    }
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune local microsoft/trocr-base-handwritten on RxHandBD"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/trocr_finetune.json"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--drop-uncertain", action="store_true")
    parser.add_argument("--source", choices=("ml", "raw"), default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> TrocrFinetuneConfig:
    config_path = Path(getattr(args, "config", "configs/trocr_finetune.json"))
    if config_path.is_file():
        config = TrocrFinetuneConfig.from_json(config_path)
    else:
        config = TrocrFinetuneConfig()
    output_dir = getattr(args, "output_dir", None) or getattr(args, "out", None)
    if output_dir is not None:
        config.output_dir = str(output_dir)
    if getattr(args, "source_checkpoint", None) is not None:
        config.source_checkpoint_dir = str(args.source_checkpoint)
    if getattr(args, "resume_from_checkpoint", None) is not None:
        config.resume_from_checkpoint = str(args.resume_from_checkpoint)
    if getattr(args, "device", None) is not None:
        config.device = args.device
    if getattr(args, "epochs", None) is not None:
        config.epochs = args.epochs
    if getattr(args, "train_batch_size", None) is not None:
        config.train_batch_size = args.train_batch_size
    if getattr(args, "eval_batch_size", None) is not None:
        config.eval_batch_size = args.eval_batch_size
    if getattr(args, "learning_rate", None) is not None:
        config.learning_rate = args.learning_rate
    if getattr(args, "max_train_samples", None) is not None:
        config.max_train_samples = args.max_train_samples
    if getattr(args, "max_eval_samples", None) is not None:
        config.max_eval_samples = args.max_eval_samples
    if getattr(args, "drop_uncertain", False):
        config.drop_uncertain = True
    if getattr(args, "source", None) is not None:
        config.source = args.source
    return config


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    if args.prepare_only:
        report = prepare_only(config)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["counts"].get("usable_train", 0) or report["counts"].get("listed_test", 0) else 1
    summary = train(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("status") == "completed" else 1


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_device(requested: str, torch_module) -> str:
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("device=cuda was requested but torch.cuda.is_available() is false")
        return "cuda"
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return requested


def _strategy_kwargs(config: TrocrFinetuneConfig, has_val: bool) -> dict:
    import inspect

    from transformers import Seq2SeqTrainingArguments

    eval_strategy = config.eval_strategy if has_val else "no"
    save_strategy = config.save_strategy
    names = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in names else "evaluation_strategy"
    return {eval_key: eval_strategy, "save_strategy": save_strategy}


def _make_trainer(
    trainer_cls,
    model,
    args,
    train_dataset,
    eval_dataset,
    collator,
    processor,
    callbacks,
):
    common = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": callbacks,
    }
    try:
        return trainer_cls(**common, processing_class=processor)
    except TypeError:
        return trainer_cls(**common, tokenizer=processor)


def _make_jsonl_callback(base_cls, path: Path):
    class JsonlLogCallback(base_cls):
        def __init__(self) -> None:
            super().__init__()
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": state.global_step, **logs}) + "\n")
            return control

    return JsonlLogCallback()


def _count_codes(issues) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in issues:
        counts[item.code] = counts.get(item.code, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
