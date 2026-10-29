"""Surya OCR (2.x) as an engine.

Surya 2.x runs layout and recognition through one vision-language model served by a
local backend: vLLM (NVIDIA GPU, via Docker) or llama.cpp (CPU / Apple Silicon). The
manager starts that server on first use, or attaches to one at SURYA_INFERENCE_URL.

LICENCE: the code is Apache 2.0, but the model weights use a modified AI Pubs Open
Rail-M licence: free for research, personal use and startups under a funding/revenue
limit; broader commercial use needs a licence from the vendor. Read it before using
Surya for client work.

Blocks come back in reading order with HTML content and a mean token probability.
"""

from __future__ import annotations

from PIL import Image

from ocr.engines.base import STATUS_SUCCESS, Engine, EngineLine, EngineResult, mean_or_none
from ocr.engines.text import html_to_text


class SuryaEngine(Engine):
    name = "surya"
    description = "Surya 2.x layout-aware VLM OCR, 90+ languages. Needs a vLLM or llama.cpp backend."
    license_note = "Code Apache 2.0; weights: modified AI Pubs Open Rail-M (commercial limits apply)."
    requires = ("surya",)

    def __init__(self, predictor=None, manager=None) -> None:
        self._predictor = predictor
        self._manager = manager
        self._owns_manager = False

    def availability(self) -> tuple[bool, str]:
        if self._predictor is not None:
            return True, "ready (injected)"
        return super().availability()

    def _get_predictor(self):
        if self._predictor is None:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor

            if self._manager is None:
                self._manager = SuryaInferenceManager()  # starts or attaches to a backend lazily
                self._owns_manager = True
            self._predictor = RecognitionPredictor(self._manager)
        return self._predictor

    def _recognize(self, image: Image.Image) -> EngineResult:
        page = self._get_predictor()([image])[0]
        warnings: list[str] = []
        lines: list[EngineLine] = []
        for block in sorted(page.blocks, key=lambda item: item.reading_order):
            if getattr(block, "skipped", False):
                continue
            if getattr(block, "error", False):
                warnings.append(f"block {block.reading_order} ({block.label}) failed")
                continue
            text = html_to_text(block.html)
            if not text:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in block.bbox)
            lines.append(EngineLine(text=text, bbox=(x1, y1, x2, y2), confidence=getattr(block, "confidence", None)))
        return EngineResult(
            engine=self.name,
            status=STATUS_SUCCESS,
            text="\n".join(line.text for line in lines),
            lines=lines,  # blocks (paragraphs, tables), not single text lines
            confidence=mean_or_none(line.confidence for line in lines),
            warnings=warnings + ([] if lines else ["no text found"]),
            model="surya",
        )

    def close(self) -> None:
        if self._owns_manager and self._manager is not None:
            try:
                self._manager.stop()
            finally:
                self._manager = None
                self._predictor = None
