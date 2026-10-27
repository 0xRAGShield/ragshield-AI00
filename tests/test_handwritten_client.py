import os
from pathlib import Path

import numpy as np
from PIL import Image

from ocr.handwritten_ocr_client import (
    MODEL_ID,
    REQUIRED_FILES,
    HandwrittenOcrClient,
    HandwrittenOcrConfig,
    LocalModelError,
)
from ocr.pipeline import compare_image
from ocr.preprocessing import PreprocessConfig


def _touch_checkpoint(directory: Path) -> None:
    directory.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (directory / name).write_bytes(b"x")


def _image() -> Image.Image:
    image = Image.new("RGB", (24, 24), "white")
    image.putpixel((2, 2), (0, 0, 0))
    return image


def test_missing_checkpoint_does_not_invent_text(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    client = HandwrittenOcrClient(HandwrittenOcrConfig(cache_dir=tmp_path / "empty"))

    def loader(_path):
        raise AssertionError("loader ran without local files")

    client._loader = loader
    result = client.transcribe(_image(), "original")
    assert result.status == "error"
    assert result.text is None
    assert result.model == MODEL_ID
    assert "incomplete" in (result.error or "")
    assert "external service" in (result.error or "")


def test_loader_is_offline_and_local(tmp_path: Path) -> None:
    directory = tmp_path / "cache" / "sherif1313__Arabic-English-handwritten-OCR-v3"
    _touch_checkpoint(directory)
    seen = {}

    def loader(path: Path):
        seen["path"] = path
        seen["offline"] = os.environ.get("HF_HUB_OFFLINE")
        seen["transformers_offline"] = os.environ.get("TRANSFORMERS_OFFLINE")

        class _Model:
            def generate(self, **kwargs):
                raise RuntimeError("stopped before tokens")

        class _Tokenizer:
            eos_token_id = 0

        class _Processor:
            tokenizer = _Tokenizer()

            def apply_chat_template(self, *args, **kwargs):
                return "prompt"

            def __call__(self, *args, **kwargs):
                import torch

                return {"input_ids": torch.zeros((1, 2), dtype=torch.long)}

            def batch_decode(self, *args, **kwargs):
                return ["unused"]

        return _Model(), _Processor()

    client = HandwrittenOcrClient(HandwrittenOcrConfig(cache_dir=tmp_path / "cache"), loader=loader)
    result = client.transcribe(np.zeros((16, 16), dtype=np.uint8), "grayscale")
    assert seen["path"] == directory
    assert seen["offline"] == "1"
    assert seen["transformers_offline"] == "1"
    assert result.status == "error"
    assert result.text is None
    assert "RuntimeError" in (result.error or "")
    assert "api.mistral.ai" not in (result.error or "")


def test_refuses_a_different_model_id() -> None:
    os.environ["HANDWRITTEN_OCR_MODEL"] = "some/other-model"
    try:
        try:
            HandwrittenOcrConfig.from_env()
        except LocalModelError as exc:
            assert MODEL_ID in str(exc)
        else:
            raise AssertionError("other model ids must be refused")
    finally:
        os.environ.pop("HANDWRITTEN_OCR_MODEL", None)


def test_pipeline_saves_the_client_text_without_loading_weights(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    _image().save(image_path)

    class _Client:
        def transcribe(self, image, preprocessing_method: str):
            from ocr.handwritten_ocr_client import Transcription

            return Transcription(
                text="نص تجريبي 500 mg",
                model=MODEL_ID,
                status="success",
                preprocessing_method=preprocessing_method,
                latency_ms=3.0,
            )

    result = compare_image(
        image_path,
        _Client(),
        variants=["original"],
        config=PreprocessConfig(enable_upscale=False, enable_deskew=False),
        output_dir=tmp_path / "out",
    )
    assert result.transcriptions[0].text == "نص تجريبي 500 mg"
    assert result.transcriptions[0].model == MODEL_ID
    saved = (tmp_path / "out" / "transcriptions.json").read_text(encoding="utf-8")
    assert "نص تجريبي 500 mg" in saved
    assert "api.mistral.ai" not in saved
