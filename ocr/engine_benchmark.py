"""Run several OCR engines over the same folder of pages and compare them.

Put a ``<image>.txt`` with the correct text next to an image (or in a ``ground_truth/``
folder) to get error rates. Failed pages count as fully wrong in ``cer_all`` so an
engine cannot look good by refusing the hard pages.

Pages are given to every engine identically. ``preprocess="original"`` hands over the
EXIF-oriented photo untouched (isolates the engines); ``"aligned"`` hands over this
project's cropped, turned and deskewed colour page.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from ocr.batch import _ground_truth, discover_images
from ocr.engines.base import STATUS_SUCCESS, Engine
from ocr.evaluation import score_predictions
from ocr.preprocessing import PreprocessConfig, prepare_hybrid_page, validate_image

PREPROCESS_CHOICES = ("original", "aligned")


@dataclass
class BenchmarkRow:
    image: str
    engine: str
    status: str
    latency_ms: float
    chars: int
    confidence: float | None
    cer: float | None = None
    wer: float | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    rows: list[BenchmarkRow]
    engines: list[str]
    scored_images: int
    summary: list[dict]
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {
            "engines": self.engines,
            "scored_images": self.scored_images,
            "summary": self.summary,
            "elapsed_ms": self.elapsed_ms,
            "rows": [row.__dict__ for row in self.rows],
        }


def load_page(path: Path, preprocess: str, config: PreprocessConfig | None = None) -> Image.Image:
    page, info = validate_image(path)
    if preprocess == "original":
        return ImageOps.exif_transpose(page).convert("RGB")
    prepared = prepare_hybrid_page(page, info, config or PreprocessConfig())
    return (prepared.aligned or prepared.processed).convert("RGB")


def run_engine_benchmark(
    input_dir: str | Path,
    out_dir: str | Path,
    engines: list[Engine],
    preprocess: str = "original",
    preprocess_config: PreprocessConfig | None = None,
) -> BenchmarkReport:
    if preprocess not in PREPROCESS_CHOICES:
        raise ValueError(f"preprocess must be one of {PREPROCESS_CHOICES}")
    started = time.perf_counter()
    root, destination = Path(input_dir), Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkRow] = []
    samples: list[dict] = []
    predictions: list[dict] = []

    for path in discover_images(root):
        relative = path.relative_to(root).as_posix()
        reference = _ground_truth(path)
        if reference is not None:
            samples.append({"id": relative, "text": reference})
        try:
            page = load_page(path, preprocess, preprocess_config)
        except Exception as exc:  # noqa: BLE001 - an unreadable file must not end the run
            for engine in engines:
                rows.append(BenchmarkRow(relative, engine.name, "error", 0.0, 0, None, error=f"unreadable image: {exc}"))
                if reference is not None:
                    predictions.append({"image_id": relative, "variant": engine.name, "status": "error", "text": None})
            continue
        folder = destination / "pages" / relative.replace("/", "__").rsplit(".", 1)[0]
        folder.mkdir(parents=True, exist_ok=True)
        for engine in engines:
            result = engine.recognize_page(page)
            text = result.text if result.status == STATUS_SUCCESS else None
            if text is not None:
                (folder / f"{engine.name}.txt").write_text(text, encoding="utf-8")
            (folder / f"{engine.name}.json").write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            rows.append(BenchmarkRow(relative, engine.name, result.status, round(result.latency_ms, 1), len(text or ""),
                                     result.confidence, error=result.error, warnings=list(result.warnings)))
            if reference is not None:
                predictions.append({"image_id": relative, "variant": engine.name,
                                    "status": "success" if text is not None else "error", "text": text,
                                    "latency_ms": result.latency_ms, "error": result.error})

    summary = _summarize(rows, engines, samples, predictions)
    _attach_item_errors(rows, samples, predictions)
    report = BenchmarkReport(rows=rows, engines=[e.name for e in engines], scored_images=len(samples),
                             summary=summary, elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
    _write(report, destination)
    return report


def _summarize(rows, engines, samples, predictions) -> list[dict]:
    scores = {}
    if samples and predictions:
        scores = {item.variant: item for item in score_predictions(samples, predictions).variants}
    out = []
    for engine in engines:
        mine = [row for row in rows if row.engine == engine.name]
        ok = [row for row in mine if row.status == STATUS_SUCCESS]
        latencies = [row.latency_ms for row in ok]
        confidences = [row.confidence for row in ok if row.confidence is not None]
        entry = {
            "engine": engine.name,
            "images": len(mine),
            "failed": len(mine) - len(ok),
            "failure_rate": round((len(mine) - len(ok)) / len(mine), 4) if mine else 0.0,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "license": engine.license_note,
        }
        score = scores.get(engine.name)
        if score is not None:
            entry.update(scored=score.count, cer_all=score.cer_all, wer_all=score.wer_all,
                         cer_micro=score.cer_micro, wer_micro=score.wer_micro)
        out.append(entry)
    return out


def _attach_item_errors(rows: list[BenchmarkRow], samples: list[dict], predictions: list[dict]) -> None:
    if not samples or not predictions:
        return
    for item in score_predictions(samples, predictions).items:
        for row in rows:
            if row.image == item.image_id and row.engine == item.variant:
                row.cer, row.wer = item.cer, item.wer


def _write(report: BenchmarkReport, destination: Path) -> None:
    (destination / "engine_benchmark.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    with (destination / "engine_benchmark.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "engine", "status", "latency_ms", "chars", "confidence", "cer", "wer", "error"])
        for row in report.rows:
            writer.writerow([row.image, row.engine, row.status, row.latency_ms, row.chars, row.confidence,
                             row.cer, row.wer, row.error or ""])
    (destination / "engine_summary.md").write_text(_markdown(report), encoding="utf-8")


def _markdown(report: BenchmarkReport) -> str:
    def cell(value, spec="{}"):
        return "n/a" if value is None else spec.format(value)

    lines = [
        "# OCR engine comparison", "",
        f"{len(report.rows) // max(1, len(report.engines))} pages, {report.scored_images} with ground truth, "
        f"{report.elapsed_ms / 1000:.1f}s.", "",
        "`cer_all` counts a failed page as completely wrong, so it is the number to compare. "
        "Confidence means different things per engine: compare it within an engine only.", "",
        "| engine | images | failed | cer_all | wer_all | cer (successes) | mean latency | mean confidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in report.summary:
        lines.append(
            f"| {entry['engine']} | {entry['images']} | {entry['failed']} | {cell(entry.get('cer_all'), '{:.3f}')} "
            f"| {cell(entry.get('wer_all'), '{:.3f}')} | {cell(entry.get('cer_micro'), '{:.3f}')} "
            f"| {cell(entry['mean_latency_ms'], '{:.0f} ms')} | {cell(entry['mean_confidence'], '{:.2f}')} |")
    lines += ["", "## Licences", ""]
    lines += [f"- **{entry['engine']}**: {entry['license'] or 'not stated'}" for entry in report.summary]
    return "\n".join(lines) + "\n"
