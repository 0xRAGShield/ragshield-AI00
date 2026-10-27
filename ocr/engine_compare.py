"""Run Qwen and/or TrOCR without replacing either model's text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ocr.config import MODE_TROCR_FULL_PAGE, RECOGNITION_MODES
from ocr.crops import CropLoadError, CropSpec, load_crops_for_image
from ocr.handwritten_ocr_client import HandwrittenOcrClient, Transcription
from ocr.pipeline import CompareResult, compare_image
from ocr.preprocessing import ImageValidationError, PreprocessConfig, validate_image
from ocr.trocr_client import (
    STATUS_CROPS_MISSING,
    STATUS_INFERENCE_FAILURE,
    STATUS_SUCCESS,
    TrocrBatchResult,
    TrocrClient,
    TrocrFullPageResult,
)

# Operational flag only. Qwen is full-page; TrOCR is crop-level. A person
# should inspect both original outputs. This is not an accuracy metric and
# does not mean either transcription was verified or disproven.
REVIEW_REQUIRED_MEANING = (
    "review_required means keep both original outputs for human inspection. "
    "It is not an accuracy score. Qwen transcribes the full page; TrOCR "
    "transcribes caller-supplied crops only. Matching or differing strings "
    "do not prove either model is correct, and this is not region-level alignment."
)
COMPARISON_KIND_PAGE_VS_CROPS = "page_vs_crops"


@dataclass
class EngineCompareResult:
    source: str
    mode: str
    output_dir: str | None
    qwen: CompareResult | None = None
    trocr: TrocrBatchResult | None = None
    trocr_full_page: TrocrFullPageResult | None = None
    review_required: bool = False
    notes: list[str] = field(default_factory=list)
    crop_metadata: list[dict] = field(default_factory=list)
    comparison_kind: str | None = None
    normalized_strings_equal: bool | None = None
    write_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "source": self.source,
            "mode": self.mode,
            "output_dir": self.output_dir,
            "review_required": self.review_required,
            "review_required_meaning": REVIEW_REQUIRED_MEANING,
            "comparison_kind": self.comparison_kind,
            "normalized_strings_equal": self.normalized_strings_equal,
            "notes": list(self.notes),
            "crop_metadata": list(self.crop_metadata),
            "qwen": self.qwen.to_dict() if self.qwen else None,
            "trocr": self.trocr.to_dict() if self.trocr else None,
            "trocr_full_page": (
                self.trocr_full_page.to_dict() if self.trocr_full_page else None
            ),
        }
        if self.write_errors:
            payload["write_errors"] = list(self.write_errors)
        return payload


def recognize_image(
    path: str | Path,
    mode: str = "qwen",
    qwen_client: HandwrittenOcrClient | None = None,
    trocr_client: TrocrClient | None = None,
    variants: Sequence[str] | None = None,
    config: PreprocessConfig | None = None,
    output_dir: str | Path | None = None,
    crops_path: str | Path | None = None,
) -> EngineCompareResult:
    if mode not in RECOGNITION_MODES:
        raise ValueError(f"Unknown recognition mode {mode}. Choose from {RECOGNITION_MODES}")

    source = Path(path)
    destination = Path(output_dir) if output_dir is not None else None
    notes: list[str] = []
    crops: list[CropSpec] = []
    crop_error: CropLoadError | None = None
    page_result: TrocrFullPageResult | None = None

    if mode == MODE_TROCR_FULL_PAGE:
        client = trocr_client or TrocrClient()
        try:
            image, info = validate_image(source)
        except (ImageValidationError, OSError) as exc:
            page_result = TrocrFullPageResult(
                source=str(source),
                model=client.config.model,
                checkpoint_dir=str(client.config.checkpoint_dir),
                status=STATUS_INFERENCE_FAILURE,
                error=f"{type(exc).__name__}: {exc}",
            )
            notes.append(page_result.error or "image validation failed")
        else:
            page_result = client.transcribe_full_page(image, source=str(source))
            notes.extend(page_result.warnings)
            if info.warnings:
                notes.extend(info.warnings)
            if page_result.status != STATUS_SUCCESS:
                notes.append(page_result.error or page_result.status)
        result = EngineCompareResult(
            source=str(source),
            mode=mode,
            output_dir=str(destination) if destination else None,
            trocr_full_page=page_result,
            notes=notes,
        )
        if destination is not None:
            _write_outputs(result, destination)
        return result

    if mode in {"trocr", "compare"}:
        try:
            crops = load_crops_for_image(source, crops_path)
        except CropLoadError as exc:
            crop_error = exc
            notes.append(f"{exc.code}: {exc}")
        else:
            if not crops:
                notes.append(
                    f"{STATUS_CROPS_MISSING}: no sidecar and no --crops path. "
                    "TrOCR was not given a full page. Add a .crops.json sidecar "
                    "or pass --crops."
                )

    qwen_result = None
    if mode in {"qwen", "compare"}:
        if qwen_client is None:
            raise ValueError("Qwen client is required for qwen and compare modes")
        qwen_result = compare_image(
            source,
            qwen_client,
            variants=variants,
            config=config,
            output_dir=destination,
        )

    trocr_result = None
    if mode in {"trocr", "compare"}:
        client = trocr_client or TrocrClient()
        if crop_error is not None:
            trocr_result = _trocr_result_from_crop_error(client, crop_error)
        else:
            trocr_result = client.transcribe_crops(crops)
            if trocr_result.status != STATUS_SUCCESS:
                notes.append(trocr_result.error or trocr_result.status)

    review_required, extra_notes, comparison_kind, strings_equal = _review_flags(
        mode, qwen_result, trocr_result, crops
    )
    notes.extend(extra_notes)

    result = EngineCompareResult(
        source=str(source),
        mode=mode,
        output_dir=str(destination) if destination else None,
        qwen=qwen_result,
        trocr=trocr_result,
        review_required=review_required,
        notes=notes,
        crop_metadata=[crop.to_dict() for crop in crops],
        comparison_kind=comparison_kind,
        normalized_strings_equal=strings_equal,
    )
    if destination is not None:
        _write_outputs(result, destination)
    return result


def _trocr_result_from_crop_error(
    client: TrocrClient,
    exc: CropLoadError,
) -> TrocrBatchResult:
    inspected = client.inspect()
    return TrocrBatchResult(
        model=inspected.model,
        checkpoint_dir=inspected.checkpoint_dir,
        status=exc.code,
        error=f"{exc.code}: {exc}",
        crops=[],
        joined_text=None,
    )


def _review_flags(
    mode: str,
    qwen_result: CompareResult | None,
    trocr_result: TrocrBatchResult | None,
    crops: list[CropSpec],
) -> tuple[bool, list[str], str | None, bool | None]:
    notes: list[str] = []
    if mode != "compare":
        return False, notes, None, None

    notes.append(REVIEW_REQUIRED_MEANING)
    boxed = sum(1 for crop in crops if crop.box is not None)
    notes.append(
        f"Crop traceability: {len(crops)} supplied crop(s), {boxed} with a bounding box. "
        "Qwen has no per-region boxes, so crop boxes are not aligned to Qwen spans."
    )

    if qwen_result is None or trocr_result is None:
        notes.append("compare mode is incomplete; both original outputs that exist are kept.")
        return True, notes, COMPARISON_KIND_PAGE_VS_CROPS, None

    qwen_texts = [
        item.text
        for item in qwen_result.transcriptions
        if item.status == "success" and item.text
    ]
    trocr_ok = trocr_result.status == STATUS_SUCCESS
    if not qwen_texts:
        notes.append("Qwen produced no successful text. TrOCR was not used as a replacement.")
        return True, notes, COMPARISON_KIND_PAGE_VS_CROPS, None
    if not trocr_ok:
        notes.append(
            "TrOCR did not complete successfully. Qwen's original transcription is kept. "
            "This is not a successful TrOCR run."
        )
        return True, notes, COMPARISON_KIND_PAGE_VS_CROPS, None

    qwen_joined = _normalize("\n".join(qwen_texts))
    trocr_joined = _normalize(trocr_result.joined_text or "")
    strings_equal = qwen_joined == trocr_joined
    if strings_equal:
        notes.append(
            "After whitespace normalize, the full-page Qwen string equals the joined "
            "TrOCR crop strings. That is not verification, not an accuracy score, and "
            "not evidence that the same regions were read."
        )
    else:
        notes.append(
            "Full-page Qwen text and joined TrOCR crop text differ. Both candidates "
            "are kept. The difference is a review signal only; it does not prove "
            "either model is wrong, because the inputs are not the same region."
        )
    return True, notes, COMPARISON_KIND_PAGE_VS_CROPS, strings_equal


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _write_outputs(result: EngineCompareResult, directory: Path) -> None:
    used_names: set[str] = set()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result.write_errors.append(f"Could not create output directory {directory}: {exc}")
        result.notes.append(result.write_errors[-1])
        return

    _write_text(
        directory / "engine_compare.json",
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        result,
    )

    if result.mode == MODE_TROCR_FULL_PAGE and result.trocr_full_page is not None:
        _write_text(
            directory / "trocr_full_page.json",
            json.dumps(result.trocr_full_page.to_dict(), ensure_ascii=False, indent=2),
            result,
        )
        if result.trocr_full_page.text is not None:
            _write_text(
                directory / "trocr_full_page.txt",
                result.trocr_full_page.text,
                result,
            )
        if result.write_errors:
            _write_text(
                directory / "write_errors.txt",
                "\n".join(result.write_errors),
                result,
            )
            _write_text(
                directory / "engine_compare.json",
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                result,
            )
        return

    _write_text(
        directory / "crop_metadata.json",
        json.dumps(result.crop_metadata, ensure_ascii=False, indent=2),
        result,
    )

    if result.trocr is not None:
        _write_text(
            directory / "trocr.json",
            json.dumps(result.trocr.to_dict(), ensure_ascii=False, indent=2),
            result,
        )
        if result.trocr.joined_text:
            _write_text(directory / "trocr.txt", result.trocr.joined_text, result)
        for index, crop in enumerate(result.trocr.crops):
            if crop.text is None:
                continue
            preferred = f"trocr_{_safe_filename(crop.crop_id)}.txt"
            path = _unique_path(directory, used_names, preferred)
            if path.name != preferred:
                result.notes.append(
                    f"TrOCR crop {crop.crop_id!r} text written to {path.name} "
                    "to avoid overwriting another crop file."
                )
            _write_text(path, crop.text, result)

    if result.qwen is not None:
        for index, item in enumerate(result.qwen.transcriptions):
            if item.status != "success" or item.text is None:
                continue
            method = item.preprocessing_method or f"variant_{index}"
            preferred = f"qwen_{method}.txt"
            path = _unique_path(directory, used_names, preferred)
            if path.name != preferred:
                result.notes.append(
                    f"Qwen variant {method!r} text written to {path.name} "
                    "because that preprocessing method appeared more than once."
                )
            _write_text(path, item.text, result)

    if result.write_errors:
        _write_text(
            directory / "write_errors.txt",
            "\n".join(result.write_errors),
            result,
        )
        _write_text(
            directory / "engine_compare.json",
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            result,
        )


def _write_text(path: Path, content: str, result: EngineCompareResult) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        message = f"Failed to write {path}: {exc}"
        result.write_errors.append(message)
        result.notes.append(message)


def _unique_path(directory: Path, used: set[str], preferred: str) -> Path:
    name = preferred
    stem = Path(preferred).stem
    suffix = Path(preferred).suffix
    index = 2
    while name in used:
        name = f"{stem}_{index}{suffix}"
        index += 1
    used.add(name)
    return directory / name


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "crop"


def qwen_transcriptions(result: EngineCompareResult) -> list[Transcription]:
    if result.qwen is None:
        return []
    return result.qwen.transcriptions
