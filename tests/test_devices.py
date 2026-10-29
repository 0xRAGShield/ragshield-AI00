import sys
from pathlib import Path

import pytest
from PIL import Image

from ocr import handwritten_ocr_client as qwen_module
from ocr.config import resolve_device, resolve_dtype
from ocr.handwritten_ocr_client import REQUIRED_FILES, HandwrittenOcrClient, HandwrittenOcrConfig
from ocr.trocr_client import TrocrClient, TrocrConfig


@pytest.fixture
def no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError


def test_explicit_device_is_kept_and_lowercased() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("CUDA:1") == "cuda:1"


def test_auto_device_falls_back_to_cpu_without_torch(no_torch) -> None:
    assert resolve_device("auto") == "cpu"
    assert resolve_device(None) == "cpu"


def test_dtype_resolution(no_torch) -> None:
    assert resolve_dtype("auto", "cpu") == "float32"
    assert resolve_dtype("auto", "cuda") == "float16"  # cannot probe bf16 without torch
    assert resolve_dtype("bfloat16", "cpu") == "bfloat16"
    with pytest.raises(ValueError):
        resolve_dtype("int4", "cpu")


def test_qwen_config_reads_device_and_dtype_from_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OCR_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("HANDWRITTEN_OCR_DEVICE", "cuda:0")
    monkeypatch.setenv("HANDWRITTEN_OCR_DTYPE", "bfloat16")
    config = HandwrittenOcrConfig.from_env()
    assert (config.device, config.dtype, config.cache_dir) == ("cuda:0", "bfloat16", tmp_path)


def test_qwen_default_loader_receives_resolved_device_and_dtype(monkeypatch, tmp_path: Path, no_torch) -> None:
    checkpoint = tmp_path / "cache" / "sherif1313__Arabic-English-handwritten-OCR-v3"
    checkpoint.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (checkpoint / name).write_text("{}", encoding="utf-8")
    seen = {}

    class _Model:
        def generate(self, **_):
            raise RuntimeError("stop")

    monkeypatch.setattr(qwen_module, "_load_from_disk", lambda path, device, dtype: seen.update(device=device, dtype=dtype) or (_Model(), object()))
    client = HandwrittenOcrClient(HandwrittenOcrConfig(cache_dir=tmp_path / "cache", device="cuda", dtype="auto"))
    client.ensure_ready()
    assert seen == {"device": "cuda", "dtype": "float16"}
    assert client.device == "cuda"


class _Tensor:
    def __init__(self) -> None:
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


def test_trocr_moves_pixels_to_the_configured_device_not_a_hardcoded_cpu() -> None:
    tensor = _Tensor()

    class _Processor:
        def __call__(self, images, return_tensors):
            return {"pixel_values": tensor}

        def batch_decode(self, generated, skip_special_tokens):
            return [" hello "]

    class _Model:
        def generate(self, pixel_values, max_new_tokens):
            return [[1]]

    client = TrocrClient(TrocrConfig(device="cuda"))
    client._model, client._processor, client.device = _Model(), _Processor(), "cuda"
    assert client._generate(Image.new("RGB", (40, 40), "white")) == "hello"
    assert tensor.moved_to == "cuda"
