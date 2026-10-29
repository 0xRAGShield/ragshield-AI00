"""The existing Qwen-VL client, exposed as an engine."""

from __future__ import annotations

from PIL import Image

from ocr.engines.base import STATUS_ERROR, STATUS_SUCCESS, Engine, EngineLine, EngineResult


class QwenEngine(Engine):
    name = "qwen"
    description = "Local Qwen-VL checkpoint (default: sherif1313/Arabic-English-handwritten-OCR-v3). Handwriting, Arabic and English."
    license_note = "Check the licence of the checkpoint you configure."
    requires = ()  # the client reports a missing checkpoint itself

    def __init__(self, client=None) -> None:
        self._client = client

    def _get_client(self):
        if self._client is None:
            from ocr.handwritten_ocr_client import HandwrittenOcrClient, HandwrittenOcrConfig

            self._client = HandwrittenOcrClient(HandwrittenOcrConfig.from_env())
        return self._client

    def availability(self) -> tuple[bool, str]:
        client = self._client
        if client is None:
            return True, "loads the local checkpoint on first use"
        return True, "ready"

    def _recognize(self, image: Image.Image) -> EngineResult:
        client = self._get_client()
        result = client.transcribe(image, preprocessing_method="engine")
        if result.status != "success" or result.text is None:
            return EngineResult(engine=self.name, status=STATUS_ERROR, error=result.error or "recognition failed",
                                model=result.model)
        lines = [EngineLine(text=part) for part in result.text.split("\n") if part.strip()]
        return EngineResult(
            engine=self.name,
            status=STATUS_SUCCESS,
            text=result.text,
            lines=lines,
            confidence=result.confidence,
            warnings=list(result.quality_warnings),
            model=result.model,
        )
