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


class _FakeTensor:
    """Minimal stand-in so these tests do not need torch."""

    def __init__(self, shape):
        self.shape = shape

    def to(self, *_args, **_kwargs):
        return self


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
                return {"input_ids": _FakeTensor((1, 2))}

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


def test_other_model_ids_are_accepted_and_reported(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OCR_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("HANDWRITTEN_OCR_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    config = HandwrittenOcrConfig.from_env()
    assert config.model == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert config.resolved_checkpoint_dir() == tmp_path / "Qwen__Qwen2.5-VL-7B-Instruct"
    assert config.base_url.endswith("Qwen__Qwen2.5-VL-7B-Instruct")


def test_explicit_checkpoint_dir_and_loading_options_come_from_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HANDWRITTEN_OCR_CHECKPOINT_DIR", str(tmp_path / "mine"))
    monkeypatch.setenv("HANDWRITTEN_OCR_ATTN", "sdpa")
    monkeypatch.setenv("HANDWRITTEN_OCR_MAX_PIXELS", "1003520")
    config = HandwrittenOcrConfig.from_env()
    assert config.resolved_checkpoint_dir() == tmp_path / "mine"
    assert (config.attn_implementation, config.max_pixels) == ("sdpa", 1003520)


def test_generic_checkpoint_check_names_what_is_missing(tmp_path: Path) -> None:
    from ocr.handwritten_ocr_client import missing_checkpoint_files

    folder = tmp_path / "other"
    folder.mkdir()
    assert len(missing_checkpoint_files(folder, "org/other")) == 4
    (folder / "config.json").write_text("{}")
    (folder / "tokenizer_config.json").write_text("{}")
    (folder / "preprocessor_config.json").write_text("{}")
    assert missing_checkpoint_files(folder, "org/other") == ["model weights (*.safetensors or pytorch_model*.bin)"]
    (folder / "model-00001-of-00004.safetensors").write_bytes(b"x")
    assert missing_checkpoint_files(folder, "org/other") == []
    assert missing_checkpoint_files(folder, MODEL_ID)  # the default model still needs its exact file set


def test_a_non_default_model_loads_through_the_generic_loader(monkeypatch, tmp_path: Path) -> None:
    from ocr import handwritten_ocr_client as module

    folder = tmp_path / "cache" / "org__vl-model"
    folder.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors"):
        (folder / name).write_bytes(b"{}")
    seen = {}

    class _Model:
        generation_config = None

    monkeypatch.setattr(module, "_load_from_disk", lambda path, device, dtype, **kw: seen.update(kw, path=path) or (_Model(), object()))
    client = HandwrittenOcrClient(HandwrittenOcrConfig(model="org/vl-model", cache_dir=tmp_path / "cache", attn_implementation="sdpa", max_pixels=500000))
    client.ensure_ready()
    assert seen["path"] == folder
    assert (seen["model_id"], seen["attn_implementation"], seen["max_pixels"]) == ("org/vl-model", "sdpa", 500000)

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
