import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocr.handwritten_ocr_client import (
    GENERAL_PROMPT,
    TRANSCRIPTION_PROMPT,
    HandwrittenOcrClient,
    HandwrittenOcrConfig,
    prompt_for,
)
from ocr.preprocessing import PreprocessConfig, upscale_if_needed
from ocr.tiling import needs_tiling, plan_bands


def _page(width=3000, height=4000, rows=24) -> np.ndarray:
    """White page with a text-like bar every 150 px and blank space between."""
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    step = height // rows
    for index in range(rows):
        top = index * step + step // 3
        draw.rectangle((150, top, width - 150, top + step // 3), fill=0)
    return np.asarray(image)


def test_small_pages_are_never_split() -> None:
    assert not needs_tiling(1200, 1600, 1792)
    assert plan_bands(_page(1200, 1600), 1792) == [(0, 1600)]


def test_a_phone_sized_page_needs_bands() -> None:
    assert needs_tiling(3000, 4000, 1792)


def test_bands_cover_the_page_without_overlap_and_cut_on_blank_rows() -> None:
    gray = _page()
    bands = plan_bands(gray, 1792)
    assert len(bands) >= 2
    assert bands[0][0] == 0 and bands[-1][1] == gray.shape[0]
    for (_, bottom), (top, _) in zip(bands, bands[1:]):
        assert bottom == top  # contiguous, no duplicated rows
        assert gray[top].min() == 255, "cut went through a line of text"
    scale = 1792 / gray.shape[1]
    for top, bottom in bands:
        assert (bottom - top) * scale <= 1792 * 1.25


def test_large_page_is_transcribed_in_bands_and_joined(tmp_path) -> None:
    client = HandwrittenOcrClient(HandwrittenOcrConfig(cache_dir=tmp_path))
    client._model = object()  # skip checkpoint loading
    sizes = []

    def fake_generate(image):
        sizes.append(image.size)
        return f"band{len(sizes)}"

    client._generate = fake_generate
    page = Image.fromarray(_page()).convert("RGB")
    result = client.transcribe(page, "original")
    assert result.status == "success"
    assert result.text == "\n".join(f"band{i + 1}" for i in range(len(sizes)))
    assert len(sizes) >= 2 and all(width == 3000 for width, _ in sizes)
    assert sum(height for _, height in sizes) == 4000
    assert any("split into" in warning for warning in result.quality_warnings)


def test_tiling_can_be_disabled(tmp_path) -> None:
    client = HandwrittenOcrClient(HandwrittenOcrConfig(cache_dir=tmp_path, tile_pages=False))
    client._model = object()
    calls = []
    client._generate = lambda image: calls.append(image.size) or "whole"
    client.transcribe(Image.fromarray(_page()).convert("RGB"), "original")
    assert calls == [(3000, 4000)]


def test_line_crops_are_not_tiled_or_analysed(tmp_path) -> None:
    client = HandwrittenOcrClient(HandwrittenOcrConfig(cache_dir=tmp_path))
    client._model = object()
    calls = []
    client._generate = lambda image: calls.append(image.size) or "line"
    client.transcribe(Image.new("RGB", (900, 80), "white"), "hybrid_line")
    assert calls == [(900, 80)]


def test_prompt_modes() -> None:
    assert prompt_for("general") == GENERAL_PROMPT
    assert prompt_for("prescription") == TRANSCRIPTION_PROMPT
    assert "prescription" not in GENERAL_PROMPT.lower()
    assert prompt_for("general", custom="  Read it.  ") == "Read it."
    with pytest.raises(ValueError):
        prompt_for("poetry")


def test_prompt_mode_comes_from_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OCR_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("HANDWRITTEN_OCR_PROMPT_MODE", "prescription")
    monkeypatch.setenv("HANDWRITTEN_OCR_TILE_PAGES", "0")
    config = HandwrittenOcrConfig.from_env()
    assert config.prompt_mode == "prescription" and config.tile_pages is False
    monkeypatch.setenv("HANDWRITTEN_OCR_PROMPT_MODE", "typo")
    with pytest.raises(ValueError):
        HandwrittenOcrConfig.from_env()


def test_large_image_with_a_low_dpi_tag_is_not_upscaled() -> None:
    image = np.zeros((2000, 3000, 3), dtype=np.uint8)
    scaled, scale, notes = upscale_if_needed(image, (72.0, 72.0), PreprocessConfig())
    assert scale == 1.0 and scaled.shape == image.shape
    assert any("upscaling skipped" in note for note in notes)


def test_upscale_never_exceeds_the_output_side_cap() -> None:
    image = np.zeros((2000, 1100, 3), dtype=np.uint8)  # short side below the skip limit
    scaled, scale, _ = upscale_if_needed(image, (72.0, 72.0), PreprocessConfig(max_output_side=3000))
    assert max(scaled.shape[:2]) <= 3000
    assert scale == pytest.approx(1.5)
