"""PaddleOCR 3.x as an engine.

PaddleOCR detects text lines and recognizes each one. It is built mostly for printed
and scene text, so expect it to do better on typed documents, forms and labels than
on handwriting. Verify on your own pages with ``python -m ocr compare-engines``.

Result keys used (PaddleX ``OCRResult``): ``rec_texts``, ``rec_scores``, ``rec_polys``.
Input images are passed as BGR arrays, the OpenCV convention.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ocr.engines.base import (
    STATUS_SUCCESS,
    Engine,
    EngineLine,
    EngineResult,
    join_rows,
    mean_or_none,
    order_lines,
)


class PaddleEngine(Engine):
    name = "paddleocr"
    description = "PaddleOCR 3.x line detection + recognition (best on printed text)."
    license_note = "Apache 2.0 code. Check the licence of the downloaded models."
    requires = ("paddleocr", "paddle")

    def __init__(self, lang: str = "ar", ocr_version: str | None = None, device: str | None = None,
                 direction: str = "auto", factory=None, **options) -> None:
        self.lang = lang
        self.ocr_version = ocr_version
        self.device = device
        self.direction = direction
        self.options = options
        self._factory = factory
        self._ocr = None

    def availability(self) -> tuple[bool, str]:
        if self._factory is not None:
            return True, "ready (injected)"
        return super().availability()

    def _engine(self):
        if self._ocr is None:
            factory = self._factory
            if factory is None:
                from paddleocr import PaddleOCR as factory  # noqa: N813
            kwargs = {
                "lang": self.lang,
                # Page orientation and unwarping are handled by this project's own preprocessing.
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            if self.ocr_version:
                kwargs["ocr_version"] = self.ocr_version
            if self.device:
                kwargs["device"] = self.device
            kwargs.update(self.options)
            self._ocr = factory(**kwargs)
        return self._ocr

    def _recognize(self, image: Image.Image) -> EngineResult:
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        results = self._engine().predict(bgr)
        lines: list[EngineLine] = []
        for page in results or []:
            texts = list(page["rec_texts"])
            scores = list(page["rec_scores"])
            polys = list(page["rec_polys"])
            for text, score, poly in zip(texts, scores, polys):
                if not str(text).strip():
                    continue
                points = np.asarray(poly, dtype=float).reshape(-1, 2)
                bbox = (int(points[:, 0].min()), int(points[:, 1].min()), int(np.ceil(points[:, 0].max())), int(np.ceil(points[:, 1].max())))
                lines.append(EngineLine(text=str(text).strip(), bbox=bbox, confidence=float(score)))
        ordered = order_lines(lines, self.direction)
        warnings = [] if ordered else ["no text found"]
        return EngineResult(
            engine=self.name,
            status=STATUS_SUCCESS,
            text=join_rows(ordered),
            lines=ordered,
            confidence=mean_or_none(line.confidence for line in ordered),
            warnings=warnings,
            model=f"PaddleOCR lang={self.lang}" + (f" {self.ocr_version}" if self.ocr_version else ""),
        )
