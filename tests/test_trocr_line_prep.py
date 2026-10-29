import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocr.tiling import plan_segments
from ocr.trocr_client import STATUS_SUCCESS, TrocrClient, TrocrConfig, TrocrCropResult


def _line(width=3000, height=80, words=14) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    step = width // words
    for index in range(words):  # "words" with clear gaps between them
        draw.rectangle((index * step + 20, 20, index * step + step - 60, 60), fill="black")
    return image


def _client(mode: str, record: list) -> TrocrClient:
    client = TrocrClient(TrocrConfig(line_prep=mode))
    client.inspect = lambda: type("S", (), {"status": "not_loaded", "error": None})()
    client.ensure_ready = lambda: None

    def fake_one(spec):
        record.append(spec.image.size)
        return TrocrCropResult(crop_id=spec.crop_id, status=STATUS_SUCCESS, text=f"t{len(record)}", latency_ms=2.0)

    client._transcribe_one = fake_one
    return client


def test_pad_square_is_the_default_and_keeps_the_old_behaviour():
    assert TrocrConfig().line_prep == "pad_square"
    seen: list = []
    result = _client("pad_square", seen).transcribe_line(_line(), "l1", "src")
    assert seen == [(3000, 3000)] and result.text == "t1"


def test_stretch_hands_the_crop_over_untouched():
    seen: list = []
    _client("stretch", seen).transcribe_line(_line(), "l1", "src")
    assert seen == [(3000, 80)]


def test_chunk_splits_a_long_line_at_gaps_and_joins_the_pieces():
    seen: list = []
    result = _client("chunk", seen).transcribe_line(_line(), "l1", "src")
    assert len(seen) >= 3
    assert all(width / height <= 5.5 for width, height in seen)
    assert sum(width for width, _ in seen) == 3000  # nothing lost, nothing duplicated
    assert result.status == STATUS_SUCCESS
    assert result.text == " ".join(f"t{i + 1}" for i in range(len(seen)))
    assert result.latency_ms == pytest.approx(2.0 * len(seen))


def test_chunk_leaves_short_lines_alone():
    seen: list = []
    _client("chunk", seen).transcribe_line(_line(width=300, words=2), "l1", "src")
    assert len(seen) == 1


def test_a_failing_chunk_fails_the_line_instead_of_returning_partial_text():
    client = _client("chunk", [])
    client._transcribe_one = lambda spec: TrocrCropResult(crop_id=spec.crop_id, status="inference_failure", error="boom")
    result = client.transcribe_line(_line(), "l1", "src")
    assert result.status == "inference_failure" and result.text is None


def test_segments_are_cut_in_the_gaps_between_words():
    gray = np.asarray(_line().convert("L"))
    for left, right in plan_segments(gray, 5.0)[:-1]:
        assert gray[:, right].min() == 255, "cut went through a word"


def test_invalid_line_prep_is_rejected(monkeypatch):
    with pytest.raises(ValueError):
        TrocrConfig(line_prep="zoom")
    monkeypatch.setenv("TROCR_LINE_PREP", "stretch")
    assert TrocrConfig.from_env().line_prep == "stretch"
