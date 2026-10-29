from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocr.batch import run_real_images
from ocr.config import RECOGNITION_MODES, VARIANT_NAMES
from ocr.engine_compare import recognize_image
from ocr.evaluation import load_dataset, score_predictions
from ocr.handwritten_ocr_client import HandwrittenOcrClient, HandwrittenOcrConfig
from ocr.pipeline import compare_image, prepare
from ocr.preprocessing import PreprocessConfig
from ocr.report import write_engine_report, write_report
from ocr.sample_dataset import write_dataset
from ocr.trocr_client import STATUS_SUCCESS, TrocrClient, TrocrConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ocr", description="Local Arabic-English handwritten OCR")
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess", help="Write preprocessing variants for one image")
    preprocess.add_argument("image", type=Path)
    preprocess.add_argument("--out", type=Path, required=True)
    preprocess.add_argument("--variants", default=",".join(VARIANT_NAMES))
    _add_preprocess_flags(preprocess)

    transcribe = sub.add_parser("transcribe", help="Transcribe one image with the local model")
    transcribe.add_argument("image", type=Path)
    transcribe.add_argument("--out", type=Path, required=True)
    transcribe.add_argument("--variants", default="original")
    transcribe.add_argument(
        "--engine",
        default="qwen",
        choices=list(RECOGNITION_MODES),
        help="qwen (default), trocr (crops), trocr-full-page, compare, or hybrid",
    )
    transcribe.add_argument(
        "--crops",
        type=Path,
        default=None,
        help="Crop JSON, crop image, or directory of crops for TrOCR. Not a detector.",
    )
    transcribe.add_argument(
        "--hybrid-policy",
        default="english_line_heuristic",
        choices=("english_line_heuristic", "qwen_default"),
        help="Hybrid routing policy. Heuristic only; not a language classifier.",
    )
    transcribe.add_argument(
        "--ocr-source",
        default="aligned",
        choices=("aligned", "processed"),
        help="Hybrid: recognize original-colour crops (aligned) or the processed grayscale crops",
    )
    transcribe.add_argument(
        "--reading-direction",
        default="auto",
        choices=("auto", "ltr", "rtl"),
        help="Hybrid: order of boxes on one row. auto = right to left when the text is mostly Arabic",
    )
    transcribe.add_argument(
        "--review-policy",
        default="risk",
        choices=("risk", "all"),
        help="Hybrid per-line review flags: only lines with a concrete reason, or every line",
    )
    transcribe.add_argument(
        "--min-confidence",
        type=float,
        default=0.75,
        help="Hybrid: lines below this token confidence are flagged. Uncalibrated; tune on labelled data",
    )
    transcribe.add_argument(
        "--hybrid-dual",
        action="store_true",
        help="Also run Qwen on TrOCR-routed lines without replacing TrOCR text.",
    )
    transcribe.add_argument(
        "--trocr-line-prep",
        choices=("pad_square", "stretch", "chunk"),
        default=None,
        help="How TrOCR line crops are prepared. Default: TROCR_LINE_PREP or pad_square (unvalidated options)",
    )
    transcribe.add_argument(
        "--trocr-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional local TrOCR directory. Default remains "
            "microsoft/trocr-base-handwritten via TROCR_CHECKPOINT_DIR. "
            "Does not change the Qwen checkpoint."
        ),
    )
    _add_preprocess_flags(transcribe)
    _add_model_flags(transcribe)

    evaluate = sub.add_parser("evaluate", help="Score a predictions JSON file against the dataset")
    evaluate.add_argument("--dataset", type=Path, default=Path("data/eval"))
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)

    build = sub.add_parser("build-dataset", help="Render the synthetic evaluation images")
    build.add_argument("--out", type=Path, default=Path("data/eval"))

    real = sub.add_parser("run-images", help="Transcribe images already stored under data/images")
    real.add_argument("--input", type=Path, default=Path("data/images"))
    real.add_argument("--out", type=Path, default=Path("outputs/real_images"))
    real.add_argument("--variant", default="original", choices=list(VARIANT_NAMES))
    _add_preprocess_flags(real)
    _add_model_flags(real)

    serve = sub.add_parser("serve", help="Local page for upload, variants, and OCR")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--out", type=Path, default=Path("outputs/sessions"))

    prepare_trocr = sub.add_parser(
        "prepare-trocr-dataset",
        help="Validate RxHandBD labels, splits, and missing images (no training)",
    )
    prepare_trocr.add_argument("--config", type=Path, default=Path("configs/trocr_finetune.json"))
    prepare_trocr.add_argument("--out", type=Path, default=Path("outputs/trocr_finetune/prepare"))
    prepare_trocr.add_argument("--source", choices=("ml", "raw"), default=None)
    prepare_trocr.add_argument("--drop-uncertain", action="store_true")

    finetune = sub.add_parser(
        "finetune-trocr",
        help="Fine-tune local microsoft/trocr-base-handwritten (not Qwen)",
    )
    finetune.add_argument("--config", type=Path, default=Path("configs/trocr_finetune.json"))
    finetune.add_argument("--output-dir", type=Path, default=None)
    finetune.add_argument("--source-checkpoint", type=Path, default=None)
    finetune.add_argument("--resume-from-checkpoint", type=Path, default=None)
    finetune.add_argument("--device", default=None)
    finetune.add_argument("--epochs", type=int, default=None)
    finetune.add_argument("--train-batch-size", type=int, default=None)
    finetune.add_argument("--eval-batch-size", type=int, default=None)
    finetune.add_argument("--learning-rate", type=float, default=None)
    finetune.add_argument("--max-train-samples", type=int, default=None)
    finetune.add_argument("--max-eval-samples", type=int, default=None)
    finetune.add_argument("--drop-uncertain", action="store_true")
    finetune.add_argument("--source", choices=("ml", "raw"), default=None)

    eval_trocr = sub.add_parser(
        "evaluate-trocr",
        help="CER/WER original vs fine-tuned TrOCR on the held-out test split",
    )
    eval_trocr.add_argument("--config", type=Path, default=Path("configs/trocr_finetune.json"))
    eval_trocr.add_argument("--split-manifest", type=Path, default=None)
    eval_trocr.add_argument("--original-checkpoint", type=Path, required=True)
    eval_trocr.add_argument("--finetuned-checkpoint", type=Path, required=True)
    eval_trocr.add_argument("--out", type=Path, required=True)
    eval_trocr.add_argument("--device", default="cpu")
    eval_trocr.add_argument("--max-new-tokens", type=int, default=64)
    eval_trocr.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "build-dataset":
        path = write_dataset(args.out)
        print(path)
        return 0
    if args.command == "preprocess":
        variants, _metadata = prepare(
            args.image,
            config=_config_from_args(args),
            variants=_names(args.variants),
            output_dir=args.out,
        )
        print(f"Wrote {len(variants)} variants to {args.out}")
        return 0
    if args.command == "transcribe":
        args.out.mkdir(parents=True, exist_ok=True)
        if args.engine == "hybrid":
            from ocr.hybrid import HybridConfig, run_hybrid
            from ocr.routing import RoutingConfig

            hybrid = run_hybrid(
                args.image,
                args.out,
                HandwrittenOcrClient(_qwen_config(args)),
                _trocr_client_from_args(args),
                HybridConfig(
                    routing=RoutingConfig(policy=args.hybrid_policy),
                    dual_compare=args.hybrid_dual,
                    preprocess=_config_from_args(args),
                    ocr_source=args.ocr_source,
                    reading_direction=args.reading_direction,
                    review_policy=args.review_policy,
                    min_confidence=args.min_confidence,
                ),
            )
            print(
                json.dumps(
                    {
                        "mode": "hybrid",
                        "status": hybrid.status,
                        "review_required": True,
                        "lines_needing_review": hybrid.lines_needing_review,
                        "mean_confidence": hybrid.mean_confidence,
                        "reading_direction": hybrid.reading_direction,
                        "line_count": hybrid.line_count,
                        "engine_counts": hybrid.engine_counts,
                        "fallback_count": hybrid.fallback_count,
                        "detect_ms": hybrid.detect_ms,
                        "ocr_ms": hybrid.ocr_ms,
                        "error": hybrid.error,
                        "output_dir": hybrid.output_dir,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if hybrid.status != "error" else 1
        if args.engine == "qwen":
            config = _qwen_config(args)
            result = compare_image(
                args.image,
                HandwrittenOcrClient(config),
                variants=_names(args.variants),
                config=_config_from_args(args),
                output_dir=args.out,
            )
            for item in result.transcriptions:
                if item.status == "success" and item.text is not None:
                    (args.out / f"{item.preprocessing_method}.txt").write_text(item.text, encoding="utf-8")
            report = write_report(result, args.out)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            print(f"Report: {report}")
            return 0 if all(item.status == "success" for item in result.transcriptions) else 1

        qwen_client = None
        if args.engine in {"qwen", "compare"}:
            qwen_client = HandwrittenOcrClient(_qwen_config(args))
        trocr_client = _trocr_client_from_args(args)
        compared = recognize_image(
            args.image,
            mode=args.engine,
            qwen_client=qwen_client,
            trocr_client=trocr_client,
            variants=_names(args.variants),
            config=_config_from_args(args),
            output_dir=args.out,
            crops_path=args.crops,
        )
        if compared.qwen is not None:
            write_report(compared.qwen, args.out)
        report = write_engine_report(compared, args.out)
        print(json.dumps(compared.to_dict(), ensure_ascii=False, indent=2))
        print(f"Report: {report}")
        return 0 if _engine_succeeded(compared) else 1
    if args.command == "evaluate":
        samples = load_dataset(args.dataset)
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        report = score_predictions(samples, predictions)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"variants": [item.to_dict() for item in report.variants]}, indent=2))
        return 0
    if args.command == "run-images":
        config = _qwen_config(args)
        report = run_real_images(
            args.input,
            args.out,
            HandwrittenOcrClient(config),
            variant=args.variant,
            preprocess_config=_config_from_args(args),
        )
        print(
            json.dumps(
                {
                    "discovered": report.discovered,
                    "succeeded": report.succeeded,
                    "failed": report.failed,
                    "not_run": report.not_run,
                    "model_ran_locally": report.model_ran_locally,
                    "inference": report.inference,
                    "elapsed_ms": report.elapsed_ms,
                    "summary_csv": report.summary_csv,
                    "blocker": report.blocker,
                },
                indent=2,
            )
        )
        if report.blocker:
            return 2
        return 0 if report.failed == 0 else 1
    if args.command == "serve":
        from ocr.server import serve

        serve(args.host, args.port, args.out)
        return 0
    if args.command == "prepare-trocr-dataset":
        from ocr.trocr_finetune.train import config_from_args, prepare_only

        config = config_from_args(args)
        config.output_dir = str(args.out)
        report = prepare_only(config)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "finetune-trocr":
        from ocr.trocr_finetune.train import config_from_args, train

        summary = train(config_from_args(args))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary.get("status") == "completed" else 1
    if args.command == "evaluate-trocr":
        from ocr.trocr_finetune.evaluate import main as evaluate_main

        argv = [
            "--config",
            str(args.config),
            "--original-checkpoint",
            str(args.original_checkpoint),
            "--finetuned-checkpoint",
            str(args.finetuned_checkpoint),
            "--out",
            str(args.out),
            "--device",
            args.device,
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if args.split_manifest is not None:
            argv.extend(["--split-manifest", str(args.split_manifest)])
        if args.max_samples is not None:
            argv.extend(["--max-samples", str(args.max_samples)])
        return evaluate_main(argv)
    parser.error(f"Unknown command {args.command}")
    return 2


def _names(value):
    if not value or value.strip().lower() == "all":
        return list(VARIANT_NAMES)

    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in VARIANT_NAMES]

    if unknown:
        raise ValueError(
            f"Unknown variant(s): {', '.join(unknown)}. "
            f"Available: {', '.join(VARIANT_NAMES)}"
        )

    return names


def _add_preprocess_flags(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--no-upscale", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--no-orientation", action="store_true", help="Do not detect sideways pages")
    parser.add_argument("--no-page-crop", action="store_true", help="Do not cut the paper out of a photo")
    parser.add_argument("--deskew-max-degrees", type=float, default=25.0)
    parser.add_argument(
        "--check-upside-down",
        action="store_true",
        help="Compare 0 and 180 degrees using recognizer confidence (two extra short passes per page)",
    )
    parser.add_argument(
        "--denoise",
        choices=["median", "nlmeans"],
        default="median",
    )

    # Enabled by default; disable explicitly when needed.
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Disable sharpening",
    )
    parser.add_argument(
        "--no-illumination",
        action="store_true",
        help="Disable illumination correction",
    )


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt-mode",
        choices=("general", "prescription"),
        default=None,
        help="Prompt sent to the model. Default: HANDWRITTEN_OCR_PROMPT_MODE or general",
    )
    parser.add_argument("--max-image-side", type=int, default=None, help="Longest side sent to the model")
    parser.add_argument("--no-tiling", action="store_true", help="Send large pages as one image instead of bands")


def _qwen_config(args: argparse.Namespace) -> HandwrittenOcrConfig:
    config = HandwrittenOcrConfig.from_env()
    if getattr(args, "prompt_mode", None):
        config.prompt_mode = args.prompt_mode
    if getattr(args, "max_image_side", None):
        config.max_image_side = args.max_image_side
    if getattr(args, "no_tiling", False):
        config.tile_pages = False
    return config


def _trocr_client_from_args(args: argparse.Namespace) -> TrocrClient:
    config = TrocrConfig.from_env()
    checkpoint = getattr(args, "trocr_checkpoint", None)
    if checkpoint is not None:
        config.checkpoint_dir = Path(checkpoint)
        config.configured = True
    if getattr(args, "trocr_line_prep", None):
        config.line_prep = args.trocr_line_prep
    return TrocrClient(config)


def _config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        enable_upscale=not args.no_upscale,
        enable_denoise=not args.no_denoise,
        enable_clahe=not args.no_clahe,
        enable_deskew=not args.no_deskew,
        deskew_max_degrees=getattr(args, "deskew_max_degrees", 25.0),
        enable_orientation=not getattr(args, "no_orientation", False),
        enable_page_crop=not getattr(args, "no_page_crop", False),
        check_upside_down=getattr(args, "check_upside_down", False),
        denoise_method=args.denoise,
        enable_sharpen=not args.no_sharpen,
        enable_illumination=not args.no_illumination,
    )


def _engine_succeeded(result) -> bool:
    if result.mode == "qwen":
        transcriptions = result.qwen.transcriptions if result.qwen else []
        return bool(transcriptions) and all(item.status == "success" for item in transcriptions)
    if result.mode == "trocr":
        return result.trocr is not None and result.trocr.status == STATUS_SUCCESS
    if result.mode == "trocr-full-page":
        return (
            result.trocr_full_page is not None
            and result.trocr_full_page.status == STATUS_SUCCESS
        )
    if result.mode == "hybrid":
        return False
    qwen_ok = (
        result.qwen is not None
        and bool(result.qwen.transcriptions)
        and all(item.status == "success" for item in result.qwen.transcriptions)
    )
    trocr_ok = result.trocr is not None and result.trocr.status == STATUS_SUCCESS
    return bool(qwen_ok and trocr_ok)


if __name__ == "__main__":
    sys.exit(main())
