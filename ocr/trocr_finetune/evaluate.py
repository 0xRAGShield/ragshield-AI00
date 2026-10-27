#Compare original and fine-tuned TrOCR on the held-out test split



from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ocr.evaluation import NORMALIZATION, character_error_rate, word_error_rate
from ocr.trocr_client import TrocrClient, TrocrConfig, pad_line_keep_aspect
from ocr.trocr_finetune.config import TrocrFinetuneConfig
from ocr.trocr_finetune.dataset import SampleRecord, load_rxhandbd, open_sample_image


def evaluate_checkpoints(
    original_dir: Path,
    finetuned_dir: Path,
    records: list[SampleRecord],
    output_dir: Path,
    max_new_tokens: int = 64,
    pad_to_square: bool = True,
    device: str = "cpu",
    model_id: str = "microsoft/trocr-base-handwritten",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = _client(original_dir, max_new_tokens, device, model_id)
    finetuned = _client(finetuned_dir, max_new_tokens, device, model_id)
    rows = []
    for record in records:
        image = open_sample_image(record)
        if pad_to_square:
            image = pad_line_keep_aspect(image)
        original_result = original.transcribe_line(
            image,
            crop_id=record.sample_id,
            source=record.image_path or record.image_name,
        )
        finetuned_result = finetuned.transcribe_line(
            image,
            crop_id=record.sample_id,
            source=record.image_path or record.image_name,
        )
        rows.append(
            _score_pair(record, original_result, finetuned_result)
        )

    report = {
        "original_checkpoint": str(original_dir),
        "finetuned_checkpoint": str(finetuned_dir),
        "n_test": len(rows),
        "decoding": {
            "pad_line_keep_aspect": pad_to_square,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "skip_special_tokens": True,
            "device": device,
        },
        "scoring_normalization": NORMALIZATION,
        "language_labels": (
            "RxHandBD CSVs have no language column. Optional Arabic/English "
            "breakdown is omitted unless contains_arabic is true on a row."
        ),
        "aggregates": {
            "original": _aggregate(rows, "original"),
            "finetuned": _aggregate(rows, "finetuned"),
        },
        "items": rows,
    }
    (output_dir / "trocr_eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "trocr_eval.csv", rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CER/WER original vs fine-tuned TrOCR on the RxHandBD test split"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/trocr_finetune.json"))
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    if args.split_manifest and args.split_manifest.is_file():
        records = _records_from_manifest(args.split_manifest)
    else:
        config = (
            TrocrFinetuneConfig.from_json(args.config)
            if args.config.is_file()
            else TrocrFinetuneConfig()
        )
        bundle = load_rxhandbd(config)
        records = bundle.split_records("test")
    records = [row for row in records if row.usable]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        payload = {
            "status": "blocked",
            "error": (
                "No usable test pairs. Images may be missing, or the split "
                "manifest has no test rows with Ground Truth. CER/WER was not computed."
            ),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "trocr_eval.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    report = evaluate_checkpoints(
        args.original_checkpoint,
        args.finetuned_checkpoint,
        records,
        args.out,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "n_test": report["n_test"],
                "aggregates": report["aggregates"],
                "output": str(args.out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _client(checkpoint: Path, max_new_tokens: int, device: str, model_id: str) -> TrocrClient:
    return TrocrClient(
        TrocrConfig(
            model=model_id,
            checkpoint_dir=checkpoint,
            device=device,
            max_new_tokens=max_new_tokens,
            configured=True,
        )
    )


def _score_pair(record: SampleRecord, original, finetuned) -> dict:
    reference = record.text
    original_text = original.text if original.status == "success" else None
    finetuned_text = finetuned.text if finetuned.status == "success" else None
    return {
        "sample_id": record.sample_id,
        "image_name": record.image_name,
        "image_path": record.image_path,
        "reference": reference,
        "contains_arabic": record.contains_arabic,
        "uncertain": record.uncertain,
        "original_status": original.status,
        "original_prediction": original_text,
        "original_error": original.error,
        "original_cer": _maybe_cer(reference, original_text),
        "original_wer": _maybe_wer(reference, original_text),
        "finetuned_status": finetuned.status,
        "finetuned_prediction": finetuned_text,
        "finetuned_error": finetuned.error,
        "finetuned_cer": _maybe_cer(reference, finetuned_text),
        "finetuned_wer": _maybe_wer(reference, finetuned_text),
    }


def _maybe_cer(reference: str, hypothesis: str | None) -> float | None:
    if hypothesis is None:
        return None
    return round(character_error_rate(reference, hypothesis), 6)


def _maybe_wer(reference: str, hypothesis: str | None) -> float | None:
    if hypothesis is None:
        return None
    return round(word_error_rate(reference, hypothesis), 6)


def _aggregate(rows: list[dict], prefix: str) -> dict:
    scored_cer = [row[f"{prefix}_cer"] for row in rows if row[f"{prefix}_cer"] is not None]
    scored_wer = [row[f"{prefix}_wer"] for row in rows if row[f"{prefix}_wer"] is not None]
    failures = sum(1 for row in rows if row[f"{prefix}_status"] != "success")
    arabic_rows = [row for row in rows if row.get("contains_arabic")]
    payload = {
        "count": len(rows),
        "scored": len(scored_cer),
        "failure_rate": failures / len(rows) if rows else 0.0,
        "mean_cer": (sum(scored_cer) / len(scored_cer)) if scored_cer else None,
        "mean_wer": (sum(scored_wer) / len(scored_wer)) if scored_wer else None,
    }
    if arabic_rows:
        arabic_cer = [
            row[f"{prefix}_cer"] for row in arabic_rows if row[f"{prefix}_cer"] is not None
        ]
        payload["arabic_count"] = len(arabic_rows)
        payload["arabic_mean_cer"] = (
            sum(arabic_cer) / len(arabic_cer) if arabic_cer else None
        )
    return payload


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _records_from_manifest(path: Path) -> list[SampleRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in payload.get("records", []):
        if row.get("split") != "test":
            continue
        records.append(
            SampleRecord(
                sample_id=row["sample_id"],
                image_name=row["image_name"],
                text=row["text"],
                official_split=row.get("official_split", "official_test"),
                split="test",
                image_path=row.get("image_path"),
                issues=list(row.get("issues") or []),
                contains_arabic=bool(row.get("contains_arabic")),
                uncertain=bool(row.get("uncertain")),
                usable=bool(row.get("usable")),
            )
        )
    return records


if __name__ == "__main__":
    raise SystemExit(main())
