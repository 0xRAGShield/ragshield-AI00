"""Batch OCR for images already stored under ``data/images``.

The selected variant is read locally and passed to the handwritten OCR checkpoint.
A missing checkpoint records every image as not run.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ocr.evaluation import character_error_rate, word_error_rate
from ocr.handwritten_ocr_client import HandwrittenOcrClient, HandwrittenOcrConfig, LocalModelError
from ocr.pipeline import prepare
from ocr.preprocessing import SUPPORTED_SUFFIXES, ImageValidationError, PreprocessConfig


@dataclass
class ImageResult:
    source: str
    status: str
    model: str
    preprocessing_method: str
    latency_ms: float | None = None
    error: str | None = None
    text_file: str | None = None
    result_file: str | None = None
    cer: float | None = None
    wer: float | None = None
    quality_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.cer is None:
            payload.pop("cer")
            payload.pop("wer")
        if not self.quality_warnings:
            payload.pop("quality_warnings", None)
        return payload


@dataclass
class BatchReport:
    model: str
    base_url: str
    model_ran_locally: bool
    inference: str
    blocker: str | None
    discovered: int
    succeeded: int
    failed: int
    not_run: int
    elapsed_ms: float
    summary_csv: str
    images: list[ImageResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "model_ran_locally": self.model_ran_locally,
            "inference": self.inference,
            "blocker": self.blocker,
            "discovered": self.discovered,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "not_run": self.not_run,
            "elapsed_ms": self.elapsed_ms,
            "summary_csv": self.summary_csv,
            "images": [item.to_dict() for item in self.images],
        }


def discover_images(root: str | Path) -> list[Path]:
    """Return supported images under ``root``, sorted by relative path."""
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"Image directory not found: {base}")
    found = [
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(found, key=lambda path: path.relative_to(base).as_posix())


def run_real_images(
    image_root: str | Path,
    output_dir: str | Path,
    client: HandwrittenOcrClient,
    variant: str = "original",
    preprocess_config: PreprocessConfig | None = None,
    require_probe: bool = True,
) -> BatchReport:
    """Transcribe each discovered image. One failure does not stop the rest."""
    started = time.perf_counter()
    root = Path(image_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    images = discover_images(root)
    config = client.config
    originals = {path: path.read_bytes() for path in images}

    if require_probe:
        try:
            client.ensure_ready()
        except LocalModelError as exc:
            blocked = exc
        else:
            blocked = None
    else:
        blocked = None
    if blocked is not None:
        report = _empty_report(
            images,
            root,
            destination,
            config,
            variant,
            blocker=str(blocked),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        _write_summaries(report, destination)
        _assert_unchanged(originals)
        return report

    results: list[ImageResult] = []
    for path in images:
        results.append(
            _process_one(
                path,
                root,
                destination,
                client,
                variant,
                preprocess_config or PreprocessConfig(),
            )
        )
    elapsed = (time.perf_counter() - started) * 1000
    report = _summarize(results, destination, config, elapsed, blocker=None)
    _write_summaries(report, destination)
    _assert_unchanged(originals)
    return report


def _process_one(
    path: Path,
    root: Path,
    destination: Path,
    client: HandwrittenOcrClient,
    variant: str,
    preprocess_config: PreprocessConfig,
) -> ImageResult:
    relative = path.relative_to(root).as_posix()
    folder_name = relative.replace("/", "__")
    folder = destination / folder_name
    model = client.config.model
    try:
        prepared, _metadata = prepare(
            path,
            config=preprocess_config,
            variants=[variant],
            output_dir=folder,
        )
        chosen = next(item for item in prepared if item.name == variant)
    except (ImageValidationError, OSError, KeyError, ValueError) as exc:
        return _write_result(
            folder,
            ImageResult(
                source=relative,
                status="error",
                model=model,
                preprocessing_method=variant,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )

    transcription = client.transcribe(chosen.image, preprocessing_method=variant)
    text_file = None
    if transcription.status == "success" and transcription.text is not None:
        text_path = folder / "transcription.txt"
        text_path.write_text(transcription.text, encoding="utf-8")
        text_file = text_path.relative_to(destination).as_posix()
    result = ImageResult(
        source=relative,
        status=transcription.status,
        model=transcription.model,
        preprocessing_method=variant,
        latency_ms=transcription.latency_ms,
        error=transcription.error,
        text_file=text_file,
        quality_warnings=list(transcription.quality_warnings or []),
    )
    reference = _ground_truth(path)
    if reference is not None and transcription.status == "success" and transcription.text is not None:
        result.cer = round(character_error_rate(reference, transcription.text), 6)
        result.wer = round(word_error_rate(reference, transcription.text), 6)
    return _write_result(folder, result)


def _ground_truth(path: Path) -> str | None:
    candidates = [
        path.with_suffix(".txt"),
        path.parent / "ground_truth" / f"{path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return None


def _write_result(folder: Path, result: ImageResult) -> ImageResult:
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    result.result_file = result_path.relative_to(folder.parent).as_posix()
    result_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _empty_report(
    images: list[Path],
    root: Path,
    destination: Path,
    config: HandwrittenOcrConfig,
    variant: str,
    blocker: str,
    elapsed_ms: float,
) -> BatchReport:
    rows = [
        ImageResult(
            source=path.relative_to(root).as_posix(),
            status="not_run",
            model=config.model,
            preprocessing_method=variant,
            error=blocker,
        )
        for path in images
    ]
    return _summarize(rows, destination, config, elapsed_ms, blocker=blocker)


def _summarize(
    rows: list[ImageResult],
    destination: Path,
    config: HandwrittenOcrConfig,
    elapsed_ms: float,
    blocker: str | None,
) -> BatchReport:
    return BatchReport(
        model=config.model,
        base_url=config.base_url,
        model_ran_locally=blocker is None,
        inference="local-cpu",
        blocker=blocker,
        discovered=len(rows),
        succeeded=sum(1 for row in rows if row.status == "success"),
        failed=sum(1 for row in rows if row.status == "error"),
        not_run=sum(1 for row in rows if row.status == "not_run"),
        elapsed_ms=round(elapsed_ms, 2),
        summary_csv=str(destination / "summary.csv"),
        images=rows,
    )


def _write_summaries(report: BatchReport, destination: Path) -> None:
    csv_path = destination / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "status", "preprocessing_method", "latency_ms", "text_file"],
        )
        writer.writeheader()
        for item in report.images:
            writer.writerow(
                {
                    "image_path": item.source,
                    "status": item.status,
                    "preprocessing_method": item.preprocessing_method,
                    "latency_ms": "" if item.latency_ms is None else item.latency_ms,
                    "text_file": item.text_file or "",
                }
            )
    (destination / "summary.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report.summary_csv = str(csv_path)


def _assert_unchanged(originals: dict[Path, bytes]) -> None:
    for path, content in originals.items():
        if path.read_bytes() != content:
            raise RuntimeError(f"Original image was modified: {path}")

