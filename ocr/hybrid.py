"""Hybrid line OCR: detect lines, route to Qwen or TrOCR, merge in reading order."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from ocr.handwritten_ocr_client import HandwrittenOcrClient, Transcription
from ocr.line_detection import (
    DetectedLine,
    LineDetectConfig,
    LineDetectionResult,
    detect_lines,
    save_line_crops,
)
from ocr.preprocessing import (
    ImageValidationError,
    crop_region,
    PreprocessConfig,
    PreparedPage,
    prepare_hybrid_page,
    validate_image,
)
from ocr.routing import (
    ENGINE_MANUAL_REVIEW,
    ENGINE_QWEN,
    ENGINE_QWEN_FALLBACK,
    ENGINE_TROCR,
    RoutingConfig,
    RoutingDecision,
    classify_recognized_script,
    dominant_direction,
    route_line,
    should_fallback_to_qwen,
)
from ocr.trocr_client import (
    STATUS_SUCCESS,
    TrocrClient,
    TrocrCropResult,
)

OVERALL_REVIEW_NOTE = (
    "Medical prescription OCR requires a person to check every line against the "
    "image. Successful execution is not transcription accuracy and is not a "
    "clinical verification."
)


@dataclass
class HybridConfig:
    detect: LineDetectConfig = field(default_factory=LineDetectConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    dual_compare: bool = False
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    # "aligned": recognize original-colour crops (scaled and deskewed like the page).
    # "processed": recognize crops of the grayscale CLAHE/denoise/sharpen image.
    ocr_source: str = "aligned"
    # Order of boxes that share a row: "auto" follows the recognized script (right to
    # left when most letters are Arabic), or force "ltr" / "rtl".
    reading_direction: str = "auto"
    # Per-line review flags. "risk": flag lines with a concrete reason (failed, low
    # confidence, numbers, [illegible], repetition). "all": flag every line.
    # min_confidence is an UNCALIBRATED starting point; tune it on a labelled set.
    review_policy: str = "risk"
    min_confidence: float = 0.75
    # When a page looks sideways, let the recognizer's confidence pick the direction.
    orientation_by_recognizer: bool = True

    def __post_init__(self) -> None:
        if self.review_policy not in {"risk", "all"}:
            raise ValueError("review_policy must be 'risk' or 'all'")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.ocr_source not in {"aligned", "processed"}:
            raise ValueError("ocr_source must be 'aligned' or 'processed'")
        if self.reading_direction not in {"auto", "ltr", "rtl"}:
            raise ValueError("reading_direction must be 'auto', 'ltr' or 'rtl'")


@dataclass
class LineOcrResult:
    line_id: str
    reading_order: int
    bbox: list[int]
    bbox_oriented: list[int]
    crop_path: str | None
    selected_engine: str
    routing_reason: str
    raw_text: str | None
    status: str
    latency_ms: float
    review_required: bool
    warnings: list[str] = field(default_factory=list)
    fallback_text: str | None = None
    fallback_reason: str | None = None
    script_after_ocr: str | None = None
    row_id: int = 0
    # Geometric-mean token probability of the chosen engine. Ranks lines for review; not an accuracy.
    confidence: float | None = None
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HybridResult:
    source: str
    width: int
    height: int
    status: str
    review_required: bool
    merged_transcription: str
    line_count: int
    detection_warnings: list[str]
    lines: list[LineOcrResult]
    engine_counts: dict[str, int]
    fallback_count: int
    detect_ms: float
    ocr_ms: float
    warnings: list[str]
    output_dir: str | None = None
    error: str | None = None
    config: dict = field(default_factory=dict)
    processed_width: int = 0
    processed_height: int = 0
    preprocess_variant: str | None = None
    preprocess_steps: list[str] = field(default_factory=list)
    reading_direction: str = "ltr"
    lines_needing_review: int = 0
    mean_confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "reading_direction": self.reading_direction,
            "lines_needing_review": self.lines_needing_review,
            "mean_confidence": self.mean_confidence,
            "width": self.width,
            "height": self.height,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
            "status": self.status,
            "review_required": True,
            "review_note": OVERALL_REVIEW_NOTE,
            "merged_transcription": self.merged_transcription,
            "line_count": self.line_count,
            "detection_warnings": list(self.detection_warnings),
            "lines": [item.to_dict() for item in self.lines],
            "engine_counts": dict(self.engine_counts),
            "fallback_count": self.fallback_count,
            "detect_ms": self.detect_ms,
            "ocr_ms": self.ocr_ms,
            "warnings": list(self.warnings),
            "output_dir": self.output_dir,
            "error": self.error,
            "config": dict(self.config),
            "preprocess_variant": self.preprocess_variant,
            "preprocess_steps": list(self.preprocess_steps),
        }


def run_hybrid(
    path: str | Path,
    output_dir: str | Path | None,
    qwen_client: HandwrittenOcrClient,
    trocr_client: TrocrClient,
    config: HybridConfig | None = None,
) -> HybridResult:
    config = config or HybridConfig()
    source = Path(path)
    destination = Path(output_dir) if output_dir is not None else None
    warnings = [OVERALL_REVIEW_NOTE]

    try:
        page, info = validate_image(source)
    except (ImageValidationError, OSError) as exc:
        result = _error_result(source, destination, warnings, config, exc)
        if destination is not None:
            _write_hybrid_outputs(result, destination, prepared=None)
        return result

    try:
        scorer = make_orientation_scorer(qwen_client) if config.orientation_by_recognizer else None
        prepared = prepare_hybrid_page(page, info, config.preprocess, scorer)
    except Exception as exc:
        warnings.append(f"preprocessing failed: {type(exc).__name__}: {exc}")
        result = _error_result(source, destination, warnings, config, exc)
        if destination is not None:
            _write_hybrid_outputs(result, destination, prepared=None)
        return result

    detection = detect_lines(prepared.processed, config.detect)
    if config.ocr_source == "aligned" and prepared.aligned is not None:
        for line in detection.lines:
            line.ocr_image = crop_region(prepared.aligned, line.bbox)
    warnings.extend(info.warnings)
    warnings.extend(detection.warnings)
    if config.ocr_source == "aligned" and prepared.aligned is not None:
        warnings.append(
            "hybrid line detection uses combined_gray; OCR crops are cut from the "
            "original-colour page aligned to it (same scale and deskew)"
        )
    else:
        warnings.append(
            "hybrid line detection and OCR crops use combined_gray "
            "(grayscale stack without Otsu/adaptive binarization)"
        )
    if destination is not None:
        save_line_crops(detection.lines, destination / "lines")

    line_results: list[LineOcrResult] = []
    ocr_ms = 0.0
    fallback_count = 0
    counts: dict[str, int] = {
        ENGINE_QWEN: 0,
        ENGINE_TROCR: 0,
        ENGINE_QWEN_FALLBACK: 0,
        ENGINE_MANUAL_REVIEW: 0,
    }

    for line in detection.lines:
        oriented_box = prepared.geometry.map_processed_box_to_oriented(line.bbox)
        item, used_fallback, elapsed = _recognize_line(
            line,
            str(source),
            qwen_client,
            trocr_client,
            config,
            bbox_oriented=list(oriented_box),
        )
        ocr_ms += elapsed
        fallback_count += int(used_fallback)
        counts[item.selected_engine] = counts.get(item.selected_engine, 0) + 1
        line_results.append(item)

    direction = (
        dominant_direction(item.raw_text for item in line_results)
        if config.reading_direction == "auto"
        else config.reading_direction
    )
    line_results = order_lines(line_results, direction)
    merged = merge_text(line_results)
    usable = [item for item in line_results if item.status == "success" and item.raw_text]
    if not detection.lines:
        status = "no_lines"
        warnings.append("hybrid produced no line transcriptions")
    elif not usable:
        status = "error"
        warnings.append("lines were detected but none could be transcribed")
    elif len(usable) < len(line_results):
        status = "partial"
        warnings.append(
            f"{len(line_results) - len(usable)} of {len(line_results)} detected lines "
            "failed or were empty; the merged text is incomplete"
        )
    else:
        status = "success"

    result = HybridResult(
        source=str(source),
        width=prepared.geometry.oriented_width,
        height=prepared.geometry.oriented_height,
        status=status,
        review_required=True,
        merged_transcription=merged,
        line_count=len(line_results),
        detection_warnings=list(detection.warnings),
        lines=line_results,
        engine_counts=counts,
        fallback_count=fallback_count,
        detect_ms=detection.detect_ms,
        ocr_ms=ocr_ms,
        warnings=warnings,
        output_dir=str(destination) if destination else None,
        config=_config_dict(config, prepared),
        processed_width=prepared.geometry.processed_width,
        processed_height=prepared.geometry.processed_height,
        preprocess_variant=prepared.variant.name,
        preprocess_steps=list(prepared.variant.steps),
        reading_direction=direction,
        lines_needing_review=sum(1 for item in line_results if item.review_required),
        mean_confidence=_mean([item.confidence for item in line_results if item.confidence is not None]),
    )
    if destination is not None:
        _write_hybrid_outputs(result, destination, prepared=prepared)
    return result


def order_lines(lines: list[LineOcrResult], direction: str) -> list[LineOcrResult]:
    """Rows top to bottom; inside a row left to right, or right to left for Arabic.

    Renumbers ``reading_order`` so the JSON, the text file and the report agree.
    """
    ordered = sorted(
        lines,
        key=lambda item: (item.row_id, -item.bbox[0] if direction == "rtl" else item.bbox[0]),
    )
    for index, item in enumerate(ordered, start=1):
        item.reading_order = index
    return ordered


def merge_text(lines: list[LineOcrResult]) -> str:
    """One text line per visual row. Boxes on the same row are joined with a space."""
    rows: list[list[str]] = []
    last_row = None
    for item in lines:
        if not item.raw_text:
            continue
        if item.row_id == last_row and rows:
            rows[-1].append(item.raw_text)
        else:
            rows.append([item.raw_text])
            last_row = item.row_id
    return "\n".join(" ".join(parts) for parts in rows)


def _recognize_line(
    line: DetectedLine,
    source: str,
    qwen_client: HandwrittenOcrClient,
    trocr_client: TrocrClient,
    config: HybridConfig,
    bbox_oriented: list[int],
) -> tuple[LineOcrResult, bool, float]:
    decision = route_line(line.image, config.routing)
    warnings = list(line.warnings) + list(decision.warnings)
    fallback_text = None
    fallback_reason = None
    used_fallback = False
    elapsed = 0.0
    confidence: float | None = None
    quality: list[str] = []

    if decision.engine == ENGINE_TROCR:
        trocr = _run_trocr(trocr_client, line, source)
        elapsed += trocr.latency_ms
        raw = trocr.text
        status = trocr.status
        engine = ENGINE_TROCR
        confidence = trocr.confidence
        if should_fallback_to_qwen(raw, status, config.routing):
            qwen = _run_qwen(qwen_client, line)
            elapsed += qwen.latency_ms
            used_fallback = True
            fallback_text = qwen.text
            fallback_reason = (
                f"TrOCR status={status} script={classify_recognized_script(raw)}; "
                "retried with Qwen. Heuristic fallback, not an accuracy score."
            )
            warnings.append(fallback_reason)
            if qwen.status == "success" and qwen.text:
                raw = qwen.text
                status = qwen.status
                engine = ENGINE_QWEN_FALLBACK
                confidence = qwen.confidence
                quality = list(qwen.quality_warnings)
        elif config.dual_compare:
            qwen = _run_qwen(qwen_client, line)
            elapsed += qwen.latency_ms
            fallback_text = qwen.text
            fallback_reason = "dual_compare: Qwen also ran; Qwen did not replace TrOCR"
            warnings.append(fallback_reason)
    else:
        qwen = _run_qwen(qwen_client, line)
        elapsed += qwen.latency_ms
        raw = qwen.text
        status = qwen.status
        engine = ENGINE_QWEN
        confidence = qwen.confidence
        quality = list(qwen.quality_warnings)
        if qwen.error:
            warnings.append(qwen.error)

    script = classify_recognized_script(raw)
    reasons = review_reasons(
        raw=raw,
        status=status,
        confidence=confidence,
        quality_warnings=quality,
        line_warnings=line.warnings,
        used_fallback=used_fallback,
        config=config,
    )
    if reasons:
        warnings.append("review: " + ", ".join(reasons))
    review = bool(reasons)

    return (
        LineOcrResult(
            line_id=line.line_id,
            reading_order=line.reading_order,
            bbox=list(line.bbox),
            bbox_oriented=list(bbox_oriented),
            crop_path=line.crop_path,
            selected_engine=engine,
            routing_reason=decision.reason,
            raw_text=raw,
            status=status if status else "error",
            latency_ms=elapsed,
            review_required=review,
            warnings=warnings,
            fallback_text=fallback_text,
            fallback_reason=fallback_reason,
            script_after_ocr=script,
            row_id=line.row_id,
            confidence=confidence,
            review_reasons=reasons,
        ),
        used_fallback,
        elapsed,
    )


_NUMBER = re.compile(r"[0-9\u0660-\u0669\u06f0-\u06f9]")
_REPETITION_HINTS = ("repeat", "duplicate", "loop")


def make_orientation_scorer(client, probe_side: int = 800):
    """Score a candidate page orientation by how confidently the recognizer reads it.

    Runs only when the page looks sideways (or check_upside_down is on). The probe is
    a downsized copy, so each candidate costs one short recognition pass.
    """

    def score(array):
        image = Image.fromarray(array).convert("RGB")
        longest = max(image.size)
        if longest > probe_side:
            factor = probe_side / longest
            image = image.resize((max(1, int(image.width * factor)), max(1, int(image.height * factor))), Image.Resampling.LANCZOS)
        result = client.transcribe(image, preprocessing_method="orientation_probe")
        if result.status != "success" or not result.text:
            return None
        return getattr(result, "confidence", None)

    return score


def review_reasons(
    *,
    raw: str | None,
    status: str | None,
    confidence: float | None,
    quality_warnings: list[str],
    line_warnings: list[str],
    used_fallback: bool,
    config: HybridConfig,
) -> list[str]:
    """Concrete reasons a person should look at this line. Empty means no specific concern.

    A page as a whole still needs human review; this ranks which lines to look at first.
    """
    reasons: list[str] = []
    if config.review_policy == "all":
        reasons.append("policy_all")
    if not raw or status != "success":
        reasons.append("failed_or_empty")
        return reasons
    if confidence is not None and confidence < config.min_confidence:
        reasons.append(f"low_confidence({confidence:.2f}<{config.min_confidence:.2f})")
    if _NUMBER.search(raw) or needs_safety_review_numbers(raw):
        reasons.append("contains_numbers_or_dosage")
    if "[illegible]" in raw.lower():
        reasons.append("illegible_marker")
    if any(hint in warning.lower() for warning in quality_warnings for hint in _REPETITION_HINTS):
        reasons.append("possible_repetition")
    if line_warnings:
        reasons.append("line_detection_warning")
    if used_fallback:
        reasons.append("engine_fallback")
    return reasons


def needs_safety_review_numbers(raw: str) -> bool:
    """Dosage-like text: a number next to a unit such as mg, ml, مجم, مل."""
    return bool(re.search(r"\b(mg|mcg|ml|g|iu|tab|tabs|cap|caps)\b|مجم|ملجم|مل|جم|قرص|كبسولة", raw, re.IGNORECASE))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _run_qwen(client: HandwrittenOcrClient, line: DetectedLine) -> Transcription:
    return client.transcribe(line.ocr_image or line.image, preprocessing_method="hybrid_line")


def _run_trocr(
    client: TrocrClient,
    line: DetectedLine,
    source: str,
) -> TrocrCropResult:
    return client.transcribe_line(
        line.ocr_image or line.image,
        crop_id=line.line_id,
        source=source,
        box=line.bbox,
    )


def _config_dict(config: HybridConfig, prepared: PreparedPage | None = None) -> dict:
    payload = {
        "policy": config.routing.policy,
        "enable_qwen_fallback": config.routing.enable_qwen_fallback,
        "dual_compare": config.dual_compare,
        "pad_px": config.detect.pad_px,
        "preprocess": asdict(config.preprocess),
        "preprocess_variant": "combined_gray",
        "ocr_source": config.ocr_source,
        "reading_direction": config.reading_direction,
    }
    if prepared is not None:
        payload["preprocess_steps"] = list(prepared.variant.steps)
        payload["preprocess_notes"] = list(prepared.variant.notes)
        payload["geometry"] = prepared.geometry.to_dict()
    return payload


def _error_result(
    source: Path,
    destination: Path | None,
    warnings: list[str],
    config: HybridConfig,
    exc: BaseException,
) -> HybridResult:
    return HybridResult(
        source=str(source),
        width=0,
        height=0,
        status="error",
        review_required=True,
        merged_transcription="",
        line_count=0,
        detection_warnings=[],
        lines=[],
        engine_counts={},
        fallback_count=0,
        detect_ms=0.0,
        ocr_ms=0.0,
        warnings=warnings,
        output_dir=str(destination) if destination else None,
        error=f"{type(exc).__name__}: {exc}",
        config=_config_dict(config),
        preprocess_variant="combined_gray",
    )


def _write_hybrid_outputs(
    result: HybridResult,
    directory: Path,
    prepared: PreparedPage | None,
) -> None:
    from ocr.report import write_hybrid_report

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hybrid_result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (directory / "hybrid_transcription.txt").write_text(
        result.merged_transcription,
        encoding="utf-8",
    )
    for item in result.lines:
        if item.raw_text is None:
            continue
        (directory / "lines").mkdir(parents=True, exist_ok=True)
        (directory / "lines" / f"{item.line_id}.txt").write_text(
            item.raw_text, encoding="utf-8"
        )
    if prepared is not None:
        prepared.oriented.convert("RGB").save(directory / "original.png", format="PNG")
        prepared.processed.convert("RGB").save(
            directory / "preprocessed.png",
            format="PNG",
        )
    write_hybrid_report(result, directory)
