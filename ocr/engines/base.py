"""Common shape for page-level OCR engines.

An engine takes one page image and returns text, optional per-line boxes and an
optional confidence. Keeping the shape small makes it cheap to add another engine
and to compare engines on the same pages.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.util import find_spec

from PIL import Image

from ocr.routing import dominant_direction

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_UNAVAILABLE = "unavailable"


@dataclass
class EngineLine:
    text: str
    bbox: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2 in the input image
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "bbox": list(self.bbox) if self.bbox else None, "confidence": self.confidence}


@dataclass
class EngineResult:
    engine: str
    status: str
    text: str | None = None
    lines: list[EngineLine] = field(default_factory=list)
    # Meaning differs per engine (token probability, recognition score). Compare it within an
    # engine, not across engines, and never read it as an accuracy.
    confidence: float | None = None
    latency_ms: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    model: str | None = None

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "status": self.status,
            "text": self.text,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
            "warnings": list(self.warnings),
            "model": self.model,
            "lines": [line.to_dict() for line in self.lines],
        }


class Engine(ABC):
    name: str = "engine"
    description: str = ""
    license_note: str = ""
    # Importable module names that must exist for the engine to run.
    requires: tuple[str, ...] = ()

    def availability(self) -> tuple[bool, str]:
        """(usable, detail). Cheap: checks installed packages, never loads a model."""
        missing = [module for module in self.requires if find_spec(module) is None]
        if missing:
            return False, "missing Python package(s): " + ", ".join(missing)
        return True, "ready"

    @abstractmethod
    def _recognize(self, image: Image.Image) -> EngineResult:
        """Do the work. May raise; ``recognize_page`` turns that into an error result."""

    def recognize_page(self, image: Image.Image) -> EngineResult:
        usable, detail = self.availability()
        if not usable:
            return EngineResult(engine=self.name, status=STATUS_UNAVAILABLE, error=detail)
        started = time.perf_counter()
        try:
            result = self._recognize(image.convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop a comparison run
            return EngineResult(
                engine=self.name,
                status=STATUS_ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    def close(self) -> None:
        """Release servers or models. Called once when a run ends."""


def order_lines(lines: list[EngineLine], direction: str = "auto") -> list[EngineLine]:
    """Rows top to bottom; inside a row left to right, or right to left for Arabic.

    Lines without a box keep their given order and come after boxed ones.
    """
    if direction == "auto":
        direction = dominant_direction(line.text for line in lines)
    boxed = [line for line in lines if line.bbox is not None]
    unboxed = [line for line in lines if line.bbox is None]
    ordered: list[EngineLine] = []
    for row in _group_rows(boxed):
        ordered.extend(sorted(row, key=lambda line: line.bbox[0], reverse=direction == "rtl"))
    return ordered + unboxed


def _group_rows(lines: list[EngineLine]) -> list[list[EngineLine]]:
    """Lines that overlap vertically by half of the shorter one share a row."""
    rows: list[list[EngineLine]] = []
    spans: list[tuple[int, int]] = []
    for line in sorted(lines, key=lambda item: (item.bbox[1] + item.bbox[3]) / 2):
        top, bottom = line.bbox[1], line.bbox[3]
        for index, (row_top, row_bottom) in enumerate(spans):
            overlap = min(row_bottom, bottom) - max(row_top, top)
            if overlap / max(1, min(row_bottom - row_top, bottom - top)) >= 0.5:
                rows[index].append(line)
                spans[index] = (min(row_top, top), max(row_bottom, bottom))
                break
        else:
            rows.append([line])
            spans.append((top, bottom))
    return [rows[i] for i in sorted(range(len(rows)), key=lambda i: spans[i][0])]


def join_rows(lines: list[EngineLine]) -> str:
    """One text line per visual row; boxes on the same row are joined with a space."""
    rows: list[list[EngineLine]] = []
    for line in lines:
        if not line.text:
            continue
        if line.bbox is not None and rows and rows[-1][-1].bbox is not None and _same_row(rows[-1][-1].bbox, line.bbox):
            rows[-1].append(line)
        else:
            rows.append([line])
    return "\n".join(" ".join(item.text for item in row) for row in rows)


def _same_row(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap / max(1, min(a[3] - a[1], b[3] - b[1])) >= 0.5


def mean_or_none(values) -> float | None:
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None
