"""Classical OpenCV line segmentation. Does not modify the source image."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ocr.preprocessing import (
    ImageValidationError,
    crop_region,
    to_gray,
    validate_array,
)


@dataclass
class LineDetectConfig:
    pad_px: int = 6
    pad_ratio: float = 0.15
    min_line_height: int = 8
    min_line_width: int = 12
    min_area: int = 64
    max_height_frac: float = 0.5
    merge_y_overlap: float = 0.45
    merge_x_gap_frac: float = 0.05
    column_gap_frac: float = 0.2
    detect_max_side: int = 1600
    morph_width_frac: float = 0.035
    morph_height: int = 3
    # Remove thin ruling lines (notebook rules, table borders) before finding text.
    remove_rules: bool = True
    rule_min_frac: float = 0.25
    # Size the closing kernel from the text itself instead of the page width.
    adaptive_kernel: bool = True
    kernel_height_factor: float = 4.0
    # Cut regions that swallowed several lines apart at their blank rows.
    split_tall: bool = True
    split_height_factor: float = 3.0
    # Order of boxes that share a row: "ltr" or "rtl" (Arabic).
    direction: str = "ltr"

    def __post_init__(self) -> None:
        if self.direction not in {"ltr", "rtl"}:
            raise ValueError("direction must be 'ltr' or 'rtl'")


@dataclass
class DetectedLine:
    line_id: str
    bbox: tuple[int, int, int, int]
    reading_order: int
    image: Image.Image
    crop_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    # What the recognizer should read when it differs from ``image`` (for example the
    # original-colour crop). ``image`` stays the crop the routing heuristics were built for.
    ocr_image: Image.Image | None = None
    row_id: int = 0

    def to_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "bbox": list(self.bbox),
            "reading_order": self.reading_order,
            "row_id": self.row_id,
            "crop_path": self.crop_path,
            "warnings": list(self.warnings),
        }


@dataclass
class LineDetectionResult:
    lines: list[DetectedLine]
    warnings: list[str]
    width: int
    height: int
    column_layout_suspected: bool = False
    detect_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "line_count": len(self.lines),
            "warnings": list(self.warnings),
            "width": self.width,
            "height": self.height,
            "column_layout_suspected": self.column_layout_suspected,
            "detect_ms": self.detect_ms,
            "lines": [line.to_dict() for line in self.lines],
        }


def detect_lines(
    image: Image.Image,
    config: LineDetectConfig | None = None,
) -> LineDetectionResult:
    """Detect text-line boxes on a copy. Coordinates are in the original image."""
    import time

    started = time.perf_counter()
    config = config or LineDetectConfig()
    try:
        page, _info = validate_array(image.copy())
    except ImageValidationError as exc:
        size = getattr(image, "size", (0, 0))
        return LineDetectionResult(
            lines=[],
            warnings=[str(exc)],
            width=size[0],
            height=size[1],
            detect_ms=(time.perf_counter() - started) * 1000,
        )
    width, height = page.size
    warnings: list[str] = []

    if width < 8 or height < 8:
        warnings.append("image too small for line detection")
        return LineDetectionResult(
            lines=[],
            warnings=warnings,
            width=width,
            height=height,
            detect_ms=(time.perf_counter() - started) * 1000,
        )

    array = np.asarray(page)
    gray = to_gray(array)
    scale = 1.0
    longest = max(width, height)
    if longest > config.detect_max_side:
        scale = config.detect_max_side / float(longest)
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    boxes, detect_notes = _boxes_from_gray(gray, config)
    warnings.extend(detect_notes)
    mapped = [_scale_box(box, scale, width, height) for box in boxes]
    padded = [_pad_box(box, width, height, config) for box in mapped]
    filtered = _filter_boxes(padded, width, height, config, warnings)
    merged = _merge_boxes(filtered, width, config)
    merged = _dedupe_boxes(merged)
    column_layout = _column_layout_suspected(merged, width, config)
    if column_layout:
        warnings.append(
            "Possible multi-column layout. Lines are ordered top-to-bottom then "
            "left-to-right; columns may be interleaved. Human review required."
        )
    ordered, row_ids = _reading_order(merged, config.direction)

    if not ordered:
        warnings.append(
            "No text lines detected. Bounding boxes were not invented. "
            "Manual crops or review are required."
        )

    lines: list[DetectedLine] = []
    for index, (box, row_id) in enumerate(zip(ordered, row_ids), start=1):
        line_warnings: list[str] = []
        x1, y1, x2, y2 = box
        if (x2 - x1) < config.min_line_width or (y2 - y1) < config.min_line_height:
            line_warnings.append("crop is very small")
        try:
            crop = crop_region(page, box)
        except ValueError as exc:
            warnings.append(f"skipped invalid box {box}: {exc}")
            continue
        if _is_blank(crop):
            line_warnings.append("crop appears empty or nearly blank")
        lines.append(
            DetectedLine(
                line_id=f"line_{index:04d}",
                bbox=box,
                reading_order=index,
                image=crop,
                warnings=line_warnings,
                row_id=row_id,
            )
        )

    return LineDetectionResult(
        lines=lines,
        warnings=warnings,
        width=width,
        height=height,
        column_layout_suspected=column_layout,
        detect_ms=(time.perf_counter() - started) * 1000,
    )


def save_line_crops(
    lines: list[DetectedLine],
    directory: str | Path,
) -> list[DetectedLine]:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    for line in lines:
        name = f"{line.line_id}.png"
        path = folder / name
        (line.ocr_image or line.image).save(path, format="PNG")
        line.crop_path = f"lines/{name}"
    return lines


def _boxes_from_gray(
    gray: np.ndarray,
    config: LineDetectConfig,
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    notes: list[str] = []
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    height, width = blur.shape[:2]
    block = _odd(max(31, min(101, int(round(0.02 * max(height, width))))))
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        11,
    )
    if config.remove_rules:
        binary, removed = _remove_rules(binary, config)
        if removed:
            notes.append(f"removed {removed} thin ruling lines before finding text")

    kernel_w = max(12, int(round(width * config.morph_width_frac)))
    text_height = _median_component_height(binary) if config.adaptive_kernel else None
    if text_height is not None:
        kernel_w = int(min(max(12, round(config.kernel_height_factor * text_height)), 0.2 * width))
    kernel_h = max(1, config.morph_height)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _hierarchy = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < config.min_area:
            continue
        boxes.append((x, y, x + w, y + h))
    if config.split_tall:
        boxes, split_count = _split_tall_boxes(boxes, binary, height, config, text_height)
        if split_count:
            notes.append(f"split {split_count} region(s) that covered several lines")
    if not boxes:
        notes.append("adaptive morphology found no candidate regions")
    return boxes, notes


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _remove_rules(binary: np.ndarray, config: LineDetectConfig) -> tuple[np.ndarray, int]:
    """Erase thin horizontal and vertical lines. Thick strokes and text are kept."""
    height, width = binary.shape[:2]
    thickness = max(3, int(round(0.003 * max(height, width))))
    mask = np.zeros_like(binary)
    removed = 0
    for horizontal in (True, False):
        length = max(25, int(round((width if horizontal else height) * config.rule_min_frac)))
        size = (length, 1) if horizontal else (1, length)
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
        for index in range(1, count):
            extent = stats[index, cv2.CC_STAT_HEIGHT] if horizontal else stats[index, cv2.CC_STAT_WIDTH]
            if extent <= thickness:
                mask[labels == index] = 255
                removed += 1
    if removed:
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(mask))
    return binary, removed


def _median_component_height(binary: np.ndarray) -> float | None:
    """Median height of ink blobs: a robust size for letters, whatever the page scale."""
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    page_height = binary.shape[0]
    heights = [
        int(stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= 12 and 3 <= stats[i, cv2.CC_STAT_HEIGHT] <= 0.15 * page_height
    ]
    if len(heights) < 8:
        return None
    return float(np.median(heights))


def _split_tall_boxes(
    boxes: list[tuple[int, int, int, int]],
    binary: np.ndarray,
    page_height: int,
    config: LineDetectConfig,
    text_height: float | None = None,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Cut boxes that are several lines tall apart at rows with little ink.

    A line is about 2.5 letter heights tall. That estimate works even when the
    whole page merged into one box; otherwise fall back to the median box height.
    """
    if not boxes:
        return boxes, 0
    heights = [box[3] - box[1] for box in boxes]
    if text_height is not None:
        reference: float | None = 2.5 * text_height
    else:
        reference = float(np.median(heights)) if len(heights) >= 3 else None
    limit = config.max_height_frac * page_height
    result: list[tuple[int, int, int, int]] = []
    split_count = 0
    for box in boxes:
        x1, y1, x2, y2 = box
        h = y2 - y1
        too_tall = h > limit or (reference is not None and h > config.split_height_factor * reference and h > 4 * config.min_line_height)
        if not too_tall:
            result.append(box)
            continue
        parts = _split_by_blank_rows(binary[y1:y2, x1:x2], config)
        if len(parts) > 1:
            split_count += 1
            for top, bottom in parts:
                result.append((x1, y1 + top, x2, y1 + bottom))
        else:
            result.append(box)
    return result, split_count


def _split_by_blank_rows(region: np.ndarray, config: LineDetectConfig) -> list[tuple[int, int]]:
    """Row ranges separated by valleys in the ink profile. One range means no valley.

    Each text line shows up as a peak in the row-wise ink count. Cutting at the
    lowest point between two clear peaks separates lines even when descenders and
    ascenders of neighbouring lines touch and no row is completely blank.
    """
    rows = (region > 0).sum(axis=1).astype(np.float64)
    total = len(rows)
    if total < 2 * config.min_line_height or rows.max() <= 0:
        return [(0, total)]
    window = max(3, total // 60)
    smooth = np.convolve(rows, np.ones(window) / window, mode="same")
    floor = 0.35 * smooth.max()
    peaks: list[int] = []
    for index in range(1, total - 1):
        if smooth[index] >= floor and smooth[index] >= smooth[index - 1] and smooth[index] > smooth[index + 1]:
            if peaks and index - peaks[-1] < config.min_line_height:
                if smooth[index] > smooth[peaks[-1]]:
                    peaks[-1] = index  # keep the stronger of two close peaks
            else:
                peaks.append(index)
    if len(peaks) < 2:
        return [(0, total)]
    cuts: list[int] = []
    for left, right in zip(peaks, peaks[1:]):
        valley = left + int(np.argmin(smooth[left : right + 1]))
        if smooth[valley] <= 0.65 * min(smooth[left], smooth[right]):
            cuts.append(valley)
    if not cuts:
        return [(0, total)]
    edges = [0] + cuts + [total]
    return [(top, bottom) for top, bottom in zip(edges, edges[1:]) if bottom - top >= config.min_line_height] or [(0, total)]


def _scale_box(
    box: tuple[int, int, int, int],
    scale: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if scale >= 0.999:
        return box
    inv = 1.0 / scale
    x1, y1, x2, y2 = box
    mapped = (
        int(round(x1 * inv)),
        int(round(y1 * inv)),
        int(round(x2 * inv)),
        int(round(y2 * inv)),
    )
    return _clip_box(mapped, width, height)


def _pad_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    config: LineDetectConfig,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad = config.pad_px + int(round((y2 - y1) * config.pad_ratio))
    return _clip_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height)


def _clip_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def _filter_boxes(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    config: LineDetectConfig,
    warnings: list[str],
) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    max_h = int(height * config.max_height_frac)
    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if h > max_h:
            warnings.append(f"dropped tall region {box}; likely not a single line")
            continue
        if w < config.min_line_width or h < config.min_line_height:
            continue
        kept.append(box)
    return kept


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    config: LineDetectConfig,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    remaining = sorted(boxes, key=lambda box: (box[1], box[0]))
    merged: list[tuple[int, int, int, int]] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            nxt: list[tuple[int, int, int, int]] = []
            for other in remaining:
                if _should_merge(current, other, width, config):
                    current = _union(current, other)
                    changed = True
                else:
                    nxt.append(other)
            remaining = nxt
        merged.append(current)
    return merged


def _should_merge(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    width: int,
    config: LineDetectConfig,
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    min_h = max(1, min(ay2 - ay1, by2 - by1))
    if overlap / min_h < config.merge_y_overlap:
        return False
    gap = max(0, max(ax1, bx1) - min(ax2, bx2))
    return gap <= width * config.merge_x_gap_frac


def _union(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _dedupe_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        unique.append(box)
    return unique


def _column_layout_suspected(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    config: LineDetectConfig,
) -> bool:
    for index, a in enumerate(boxes):
        for b in boxes[index + 1 :]:
            overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            min_h = max(1, min(a[3] - a[1], b[3] - b[1]))
            if overlap / min_h < 0.4:
                continue
            gap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
            if gap >= width * config.column_gap_frac:
                return True
    return False


def group_rows(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Group boxes that sit on the same visual row, top to bottom.

    Two boxes share a row when they overlap vertically by at least half of the
    shorter one. Ordering by exact top pixel instead would let a 1-2 px
    difference decide which of two fields on one row is read first.
    """
    rows: list[list[tuple[int, int, int, int]]] = []
    spans: list[tuple[int, int]] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        for index, (top, bottom) in enumerate(spans):
            overlap = min(bottom, box[3]) - max(top, box[1])
            shorter = max(1, min(bottom - top, box[3] - box[1]))
            if overlap / shorter >= 0.5:
                rows[index].append(box)
                spans[index] = (min(top, box[1]), max(bottom, box[3]))
                break
        else:
            rows.append([box])
            spans.append((box[1], box[3]))
    order = sorted(range(len(rows)), key=lambda i: spans[i][0])
    return [rows[i] for i in order]


def _reading_order(
    boxes: list[tuple[int, int, int, int]],
    direction: str = "ltr",
) -> tuple[list[tuple[int, int, int, int]], list[int]]:
    """Rows top to bottom; within a row left to right, or right to left for Arabic."""
    ordered: list[tuple[int, int, int, int]] = []
    row_ids: list[int] = []
    for row_id, row in enumerate(group_rows(boxes), start=1):
        for box in sorted(row, key=lambda b: b[0], reverse=direction == "rtl"):
            ordered.append(box)
            row_ids.append(row_id)
    return ordered, row_ids


def _is_blank(image: Image.Image) -> bool:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    return float(array.std()) < 2.0
