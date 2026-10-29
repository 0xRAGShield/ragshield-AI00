"""Find out why pages fail: input quality, line detection, or recognition.

Run it on a folder of papers the OCR handles badly. Without a model it measures the
input (blur, contrast, tilt, size), runs the preprocessing and line detection, and
saves an overlay image with the detected boxes. With a model (``runner``) it also
records confidence and, when a ``<image>.txt`` ground truth sits next to the
image, the character error rate.

Every threshold below is a rule of thumb, not a calibrated value. Use the buckets
to decide where to look first, then open the overlay and judge for yourself.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ocr.batch import _ground_truth, discover_images
from ocr.evaluation import character_error_rate
from ocr.line_detection import LineDetectConfig, detect_lines
from ocr.preprocessing import ImageValidationError, PreprocessConfig, prepare_hybrid_page, validate_image

# ---- rules of thumb ---------------------------------------------------------
BLUR_LAPLACIAN_VAR = 60.0      # below this the page is probably out of focus
LOW_CONTRAST_SPREAD = 60.0     # grey levels between the typical paper tone and the darkest 1% (ink)
DARK_MEAN = 90.0               # mean grey level
MIN_SHORT_SIDE = 600           # pixels; smaller scans lose handwriting detail
HIGH_SKEW_DEGREES = 8.0
MIN_LINE_HEIGHT_PX = 14        # median detected line height after preprocessing
MAX_REASONABLE_LINES = 150
HIGH_CER = 0.30
LOW_MEAN_CONFIDENCE = 0.75

BUCKETS = ("input_quality", "detection", "recognition", "ok", "unreadable_file")


@dataclass
class TriageRow:
    image: str
    status: str = "ok"
    bucket: str = "ok"
    notes: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    dpi: str = ""
    blur: float | None = None
    contrast: float | None = None
    brightness: float | None = None
    skew_degrees: float | None = None
    rotation_deg: int = 0
    page_cropped: bool = False
    scale: float | None = None
    lines: int = 0
    rows: int = 0
    median_line_height: float | None = None
    detection_warnings: list[str] = field(default_factory=list)
    overlay: str = ""
    # filled only when a recognizer ran
    ocr_status: str = ""
    mean_confidence: float | None = None
    lines_needing_review: int | None = None
    empty_lines: int | None = None
    cer: float | None = None
    ocr_ms: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TriageReport:
    rows: list[TriageRow]
    elapsed_ms: float

    def counts(self) -> dict[str, int]:
        return {bucket: sum(1 for row in self.rows if row.bucket == bucket) for bucket in BUCKETS}

    def to_dict(self) -> dict:
        return {"counts": self.counts(), "elapsed_ms": self.elapsed_ms, "images": [row.to_dict() for row in self.rows]}


def triage_image(
    path: Path,
    root: Path,
    out_dir: Path,
    preprocess: PreprocessConfig,
    detect: LineDetectConfig,
    runner: Callable[[Path], object] | None = None,
) -> TriageRow:
    relative = path.relative_to(root).as_posix()
    row = TriageRow(image=relative)
    try:
        page, info = validate_image(path)
    except ImageValidationError as exc:
        row.status, row.bucket = "error", "unreadable_file"
        row.notes.append(str(exc))
        return row

    row.width, row.height = info.width, info.height
    row.dpi = "" if info.dpi is None else f"{info.dpi[0]:.0f}x{info.dpi[1]:.0f}"
    gray = _grey(page)
    row.blur = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
    row.contrast = round(float(np.median(gray) - np.percentile(gray, 1)), 1)
    row.brightness = round(float(gray.mean()), 1)

    try:
        prepared = prepare_hybrid_page(page, info, preprocess)
        geometry = prepared.geometry
        row.skew_degrees = round(float(geometry.angle_degrees), 2)
        row.rotation_deg = int(geometry.rotation_k90 % 4) * 90
        row.page_cropped = geometry.page_homography is not None
        row.scale = round(float(geometry.scale), 2)
        detection = detect_lines(prepared.processed, detect)
    except Exception as exc:  # noqa: BLE001 - triage must keep going on a bad page
        row.status, row.bucket = "error", "detection"
        row.notes.append(f"preprocessing/detection failed: {type(exc).__name__}: {exc}")
        return row

    row.lines = len(detection.lines)
    row.rows = len({line.row_id for line in detection.lines})
    heights = [line.bbox[3] - line.bbox[1] for line in detection.lines]
    row.median_line_height = round(float(statistics.median(heights)), 1) if heights else None
    row.detection_warnings = list(detection.warnings)
    row.overlay = _write_overlay(prepared.processed, detection, out_dir / "overlays", relative)

    if runner is not None:
        _run_recognizer(row, path, runner)
    _assign_bucket(row)
    return row


def _grey(page: Image.Image) -> np.ndarray:
    gray = np.asarray(page.convert("L"))
    longest = max(gray.shape)
    if longest > 1600:
        factor = 1600 / longest
        gray = cv2.resize(gray, (int(gray.shape[1] * factor), int(gray.shape[0] * factor)), interpolation=cv2.INTER_AREA)
    return gray


def _write_overlay(image: Image.Image, detection, directory: Path, relative: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width = max(2, canvas.width // 400)
    for line in detection.lines:
        draw.rectangle(line.bbox, outline=(220, 30, 30), width=width)
        draw.text((line.bbox[0] + 4, max(0, line.bbox[1] - 14)), str(line.reading_order), fill=(220, 30, 30))
    longest = max(canvas.size)
    if longest > 1400:
        factor = 1400 / longest
        canvas = canvas.resize((int(canvas.width * factor), int(canvas.height * factor)), Image.Resampling.LANCZOS)
    target = directory / (relative.replace("/", "__").rsplit(".", 1)[0] + ".jpg")
    canvas.save(target, quality=85)
    return target.relative_to(directory.parent).as_posix()


def _run_recognizer(row: TriageRow, path: Path, runner) -> None:
    started = time.perf_counter()
    try:
        result = runner(path)
    except Exception as exc:  # noqa: BLE001
        row.ocr_status = "error"
        row.notes.append(f"recognizer failed: {type(exc).__name__}: {exc}")
        return
    row.ocr_ms = round((time.perf_counter() - started) * 1000, 1)
    row.ocr_status = getattr(result, "status", "")
    row.mean_confidence = getattr(result, "mean_confidence", None)
    row.lines_needing_review = getattr(result, "lines_needing_review", None)
    lines = getattr(result, "lines", [])
    row.empty_lines = sum(1 for line in lines if not getattr(line, "raw_text", None))
    reference = _ground_truth(path)
    if reference is not None:
        row.cer = round(character_error_rate(reference, getattr(result, "merged_transcription", "") or ""), 4)


def _assign_bucket(row: TriageRow) -> None:
    """First matching stage wins: a page that never reached the recognizer cleanly is not a recognition problem."""
    detection: list[str] = []
    quality: list[str] = []
    recognition: list[str] = []

    if row.lines == 0:
        detection.append("no text lines found")
    if row.lines > MAX_REASONABLE_LINES:
        detection.append(f"{row.lines} boxes: probably over-segmented (texture, noise or ruling lines)")
    if row.median_line_height is not None and row.median_line_height < MIN_LINE_HEIGHT_PX:
        detection.append(f"median line height {row.median_line_height}px is very small")
    for warning in row.detection_warnings:
        if "dropped tall region" in warning or "multi-column" in warning.lower():
            detection.append(warning)

    if row.blur is not None and row.blur < BLUR_LAPLACIAN_VAR:
        quality.append(f"blurry (sharpness {row.blur} < {BLUR_LAPLACIAN_VAR})")
    if row.contrast is not None and row.contrast < LOW_CONTRAST_SPREAD:
        quality.append(f"low contrast (paper-to-ink spread {row.contrast} grey levels)")
    if row.brightness is not None and row.brightness < DARK_MEAN:
        quality.append(f"dark image (mean {row.brightness})")
    if min(row.width, row.height) and min(row.width, row.height) < MIN_SHORT_SIDE:
        quality.append(f"small image ({row.width}x{row.height})")
    if row.skew_degrees is not None and abs(row.skew_degrees) > HIGH_SKEW_DEGREES:
        quality.append(f"tilted {row.skew_degrees} degrees (corrected, but check the overlay)")
    if row.rotation_deg:
        quality.append(f"page looked sideways and was turned {row.rotation_deg} degrees")

    if row.ocr_status in {"error"}:
        recognition.append("recognizer returned an error")
    if row.cer is not None and row.cer > HIGH_CER:
        recognition.append(f"CER {row.cer} against ground truth")
    if row.mean_confidence is not None and row.mean_confidence < LOW_MEAN_CONFIDENCE:
        recognition.append(f"mean confidence {row.mean_confidence}")
    if row.lines and row.empty_lines is not None and row.empty_lines / row.lines > 0.3:
        recognition.append(f"{row.empty_lines} of {row.lines} lines came back empty")

    row.notes.extend(detection + quality + recognition)
    if detection:
        row.bucket = "detection"
    elif quality and (recognition or row.ocr_status == ""):
        row.bucket = "input_quality"
    elif recognition:
        row.bucket = "recognition"
    elif quality:
        row.bucket = "input_quality"
    else:
        row.bucket = "ok"


def run_triage(
    input_dir: str | Path,
    out_dir: str | Path,
    preprocess: PreprocessConfig | None = None,
    detect: LineDetectConfig | None = None,
    runner: Callable[[Path], object] | None = None,
) -> TriageReport:
    started = time.perf_counter()
    root, destination = Path(input_dir), Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [
        triage_image(path, root, destination, preprocess or PreprocessConfig(), detect or LineDetectConfig(), runner)
        for path in discover_images(root)
    ]
    report = TriageReport(rows=rows, elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
    write_triage(report, destination)
    return report


_CSV_COLUMNS = (
    "image", "bucket", "status", "notes", "width", "height", "dpi", "blur", "contrast", "brightness",
    "skew_degrees", "rotation_deg", "page_cropped", "scale", "lines", "rows", "median_line_height",
    "ocr_status", "mean_confidence", "lines_needing_review", "empty_lines", "cer", "ocr_ms", "overlay",
)


def write_triage(report: TriageReport, destination: Path) -> None:
    (destination / "triage.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    with (destination / "triage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in report.rows:
            record = row.to_dict()
            record["notes"] = " | ".join(row.notes)
            writer.writerow(record)
    (destination / "triage_summary.md").write_text(_summary(report), encoding="utf-8")


def _summary(report: TriageReport) -> str:
    counts = report.counts()
    total = len(report.rows) or 1
    lines = [
        "# OCR triage",
        "",
        f"{len(report.rows)} images in {report.elapsed_ms / 1000:.1f}s. "
        "Buckets say where to look first; they are rules of thumb, not verdicts.",
        "",
        "| bucket | images | share |",
        "| --- | ---: | ---: |",
    ]
    lines += [f"| {bucket} | {counts[bucket]} | {counts[bucket] / total:.0%} |" for bucket in BUCKETS if counts[bucket]]
    for bucket in ("detection", "input_quality", "recognition", "unreadable_file"):
        members = [row for row in report.rows if row.bucket == bucket]
        if not members:
            continue
        lines += ["", f"## {bucket}", ""]
        lines += [f"- `{row.image}`: {'; '.join(row.notes) or 'see overlay'}" for row in members]
    return "\n".join(lines) + "\n"
