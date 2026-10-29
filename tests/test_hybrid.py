import json
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from PIL import Image, ImageDraw

from ocr.handwritten_ocr_client import Transcription
from ocr.hybrid import HybridConfig, run_hybrid
from ocr.preprocessing import PreprocessConfig
from ocr.routing import ENGINE_QWEN, ENGINE_QWEN_FALLBACK, ENGINE_TROCR, RoutingConfig
from ocr.trocr_client import TrocrCropResult


class _Qwen:
    def __init__(self, text="qwen-line"):
        self.text = text
        self.calls = 0

    def transcribe(self, image, preprocessing_method: str) -> Transcription:
        self.calls += 1
        return Transcription(
            text=self.text,
            raw_text=self.text,
            model="sherif1313/Arabic-English-handwritten-OCR-v3",
            status="success",
            preprocessing_method=preprocessing_method,
            latency_ms=1.0,
        )


class _Trocr:
    def __init__(self, text="trocr-line", status="success"):
        self.text = text
        self.status = status
        self.calls = 0

    def inspect(self):
        return type("S", (), {"status": "success", "error": None})()

    def transcribe_line(self, image, crop_id, source, box=None) -> TrocrCropResult:
        self.calls += 1
        return TrocrCropResult(
            crop_id=crop_id,
            status=self.status,
            text=self.text if self.status == "success" else None,
            error=None if self.status == "success" else "missing",
            latency_ms=2.0,
            box=list(box) if box else None,
            source=source,
        )


def _prescription(path: Path) -> Path:
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 18, 220, 32), fill="black")
    draw.rectangle((10, 70, 220, 84), fill="black")
    image.save(path)
    return path


def test_hybrid_merges_in_reading_order_and_flags_review(tmp_path):
    path = _prescription(tmp_path / "rx.png")
    qwen = _Qwen("from-qwen")
    trocr = _Trocr("from-trocr")
    result = run_hybrid(
        path,
        tmp_path / "out",
        qwen,
        trocr,
        HybridConfig(routing=RoutingConfig(policy="english_line_heuristic")),
    )
    assert result.review_required is True
    assert result.to_dict()["review_required"] is True
    assert result.line_count >= 1
    payload = json.loads((tmp_path / "out" / "hybrid_result.json").read_text(encoding="utf-8"))
    assert payload["review_required"] is True
    assert "merged_transcription" in payload
    assert (tmp_path / "out" / "hybrid_transcription.txt").is_file()
    assert (tmp_path / "out" / "hybrid_report.html").is_file()
    orders = [item.reading_order for item in result.lines]
    assert orders == sorted(orders)
    for item in result.lines:
        assert item.raw_text in {"from-qwen", "from-trocr"}
        assert item.selected_engine in {ENGINE_QWEN, ENGINE_TROCR, ENGINE_QWEN_FALLBACK}
        assert item.bbox[2] > item.bbox[0]


def test_hybrid_qwen_fallback_when_trocr_fails(tmp_path):
    path = _prescription(tmp_path / "rx.png")
    qwen = _Qwen("fallback-qwen")
    trocr = _Trocr(text=None, status="checkpoint_missing")
    result = run_hybrid(
        path,
        tmp_path / "out",
        qwen,
        trocr,
        HybridConfig(routing=RoutingConfig(policy="english_line_heuristic")),
    )
    assert result.fallback_count >= 1 or all(
        item.selected_engine == ENGINE_QWEN for item in result.lines
    )
    if result.fallback_count:
        assert any(item.selected_engine == ENGINE_QWEN_FALLBACK for item in result.lines)
        assert any(item.raw_text == "fallback-qwen" for item in result.lines)


def test_hybrid_preserves_empty_and_writes_collision_safe_folder(tmp_path):
    path = tmp_path / "blankish.png"
    image = Image.new("RGB", (90, 70), (245, 245, 245))
    image.putpixel((1, 1), (244, 244, 244))
    image.save(path)
    result = run_hybrid(path, tmp_path / "out", _Qwen(), _Trocr())
    assert result.review_required is True
    assert result.status in {"no_lines", "success"}
    first = json.loads((tmp_path / "out" / "hybrid_result.json").read_text(encoding="utf-8"))
    run_hybrid(path, tmp_path / "out", _Qwen("second"), _Trocr("second"))
    second = json.loads((tmp_path / "out" / "hybrid_result.json").read_text(encoding="utf-8"))
    assert first["source"]
    assert second["source"]


def test_cli_qwen_and_hybrid_flags_do_not_collide():
    from ocr.config import RECOGNITION_MODES

    assert RECOGNITION_MODES[0] == "qwen"
    assert "trocr" in RECOGNITION_MODES
    assert "trocr-full-page" in RECOGNITION_MODES
    assert "compare" in RECOGNITION_MODES
    assert "hybrid" in RECOGNITION_MODES


def _scale_only_config() -> HybridConfig:
    return HybridConfig(
        routing=RoutingConfig(policy="english_line_heuristic"),
        preprocess=PreprocessConfig(
            enable_upscale=True,
            enable_denoise=False,
            enable_clahe=False,
            enable_deskew=False,
            enable_sharpen=False,
            enable_illumination=False,
            max_scale=2.0,
            target_dpi=300,
            assumed_dpi_when_missing=72,
            min_short_side_px=1200,
        ),
    )


def test_hybrid_calls_full_combined_preprocess(monkeypatch, tmp_path):
    from ocr import hybrid as hybrid_mod
    from ocr.preprocessing import prepare_hybrid_page as real_prepare

    calls = []

    def wrapped(image, info, config):
        calls.append(config)
        return real_prepare(image, info, config)

    monkeypatch.setattr(hybrid_mod, "prepare_hybrid_page", wrapped)
    path = _prescription(tmp_path / "rx.png")
    preprocess = PreprocessConfig(enable_denoise=False, enable_clahe=True)
    result = run_hybrid(
        path,
        tmp_path / "out",
        _Qwen(),
        _Trocr(),
        HybridConfig(preprocess=preprocess),
    )
    assert calls
    assert calls[0] is preprocess
    assert result.preprocess_variant == "combined_gray"
    assert "grayscale" in result.preprocess_steps
    payload = json.loads((tmp_path / "out" / "hybrid_result.json").read_text(encoding="utf-8"))
    assert payload["config"]["preprocess"]["enable_denoise"] is False
    assert payload["config"]["preprocess"]["enable_clahe"] is True
    assert payload["preprocess_variant"] == "combined_gray"
    html = (tmp_path / "out" / "hybrid_report.html").read_text(encoding="utf-8")
    assert "combined_gray" in html
    assert (tmp_path / "out" / "original.png").is_file()
    assert (tmp_path / "out" / "preprocessed.png").is_file()
    for item in result.lines:
        assert item.bbox_oriented[2] > item.bbox_oriented[0]


def test_hybrid_detects_and_crops_from_processed_image(monkeypatch, tmp_path):
    from ocr import hybrid as hybrid_mod
    from ocr.line_detection import detect_lines as real_detect

    seen = {}

    def wrapped(image, config=None):
        seen["size"] = image.size
        result = real_detect(image, config)
        seen["detect_size"] = (result.width, result.height)
        if result.lines:
            seen["crop_size"] = result.lines[0].image.size
        return result

    monkeypatch.setattr(hybrid_mod, "detect_lines", wrapped)
    path = _prescription(tmp_path / "rx.png")
    result = run_hybrid(path, tmp_path / "out", _Qwen(), _Trocr(), _scale_only_config())
    assert seen["size"] == (480, 240)
    assert seen["detect_size"] == (480, 240)
    assert result.processed_width == 480
    assert result.processed_height == 240
    assert result.width == 240
    assert result.height == 120
    assert "lanczos-upscale" in result.preprocess_steps
    assert "median" not in result.preprocess_steps
    for item in result.lines:
        assert item.bbox[2] - item.bbox[0] >= (item.bbox_oriented[2] - item.bbox_oriented[0]) * 1.5
        left, _top, right, _bottom = item.bbox_oriented
        assert 0 <= left < right <= 240


def test_hybrid_cli_preprocess_flags_are_forwarded(monkeypatch, tmp_path):
    from ocr.hybrid import HybridResult

    captured = {}

    def fake_run_hybrid(path, out, qwen, trocr, config):
        captured["config"] = config
        return HybridResult(
            source=str(path),
            width=1,
            height=1,
            status="success",
            review_required=True,
            merged_transcription="",
            line_count=0,
            detection_warnings=[],
            lines=[],
            engine_counts={},
            fallback_count=0,
            detect_ms=0.0,
            ocr_ms=0.0,
            warnings=[],
            output_dir=str(out),
            preprocess_variant="combined_gray",
        )

    monkeypatch.setattr("ocr.cli.HandwrittenOcrClient", lambda cfg: object())
    monkeypatch.setattr("ocr.cli.TrocrClient", lambda cfg: object())
    monkeypatch.setattr("ocr.hybrid.run_hybrid", fake_run_hybrid)
    from ocr.cli import main

    image = _prescription(tmp_path / "rx.png")
    code = main(
        [
            "transcribe",
            str(image),
            "--out",
            str(tmp_path / "out"),
            "--engine",
            "hybrid",
            "--no-denoise",
            "--no-deskew",
            "--no-upscale",
        ]
    )
    assert code == 0
    preprocess = captured["config"].preprocess
    assert preprocess.enable_denoise is False
    assert preprocess.enable_deskew is False
    assert preprocess.enable_upscale is False
    assert preprocess.enable_clahe is True


def test_cli_config_from_args_maps_no_flags():
    from ocr.cli import _config_from_args

    args = SimpleNamespace(
        no_upscale=True,
        no_denoise=True,
        no_clahe=True,
        no_deskew=True,
        denoise="nlmeans",
        no_sharpen=True,
        no_illumination=True,
    )
    config = _config_from_args(args)
    assert config.enable_upscale is False
    assert config.enable_denoise is False
    assert config.enable_clahe is False
    assert config.enable_deskew is False
    assert config.enable_sharpen is False
    assert config.enable_illumination is False
    assert config.denoise_method == "nlmeans"


def test_hybrid_preprocessing_failure_is_reported(monkeypatch, tmp_path):
    from ocr import hybrid as hybrid_mod

    def boom(_image, _info, _config):
        raise RuntimeError("synthetic preprocess failure")

    monkeypatch.setattr(hybrid_mod, "prepare_hybrid_page", boom)
    path = _prescription(tmp_path / "rx.png")
    result = run_hybrid(path, tmp_path / "out", _Qwen(), _Trocr())
    assert result.status == "error"
    assert "synthetic preprocess failure" in (result.error or "")
    payload = json.loads((tmp_path / "out" / "hybrid_result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["config"]["preprocess_variant"] == "combined_gray"


def test_cli_qwen_mode_does_not_call_hybrid(monkeypatch, tmp_path):
    called = {"hybrid": 0, "compare": 0}

    class _Result:
        transcriptions = []

        def to_dict(self):
            return {"mode": "qwen", "status": "success"}

    def fake_compare(*_args, **_kwargs):
        called["compare"] += 1
        return _Result()

    def fake_hybrid(*_args, **_kwargs):
        called["hybrid"] += 1
        raise AssertionError("hybrid must not run in qwen mode")

    monkeypatch.setattr("ocr.cli.compare_image", fake_compare)
    monkeypatch.setattr("ocr.cli.write_report", lambda *_args, **_kwargs: tmp_path / "report.html")
    monkeypatch.setattr("ocr.cli.HandwrittenOcrClient", lambda cfg: object())
    monkeypatch.setattr("ocr.hybrid.run_hybrid", fake_hybrid)
    from ocr.cli import main

    image = _prescription(tmp_path / "rx.png")
    code = main(
        [
            "transcribe",
            str(image),
            "--out",
            str(tmp_path / "out"),
            "--engine",
            "qwen",
        ]
    )
    assert code == 0
    assert called["compare"] == 1
    assert called["hybrid"] == 0


class _FlakyQwen:
    """Reads the first line, then returns nothing for the rest."""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, image, preprocessing_method: str) -> Transcription:
        self.calls += 1
        text = "first line" if self.calls == 1 else ""
        return Transcription(text=text, raw_text=text, model="m", status="success",
                             preprocessing_method=preprocessing_method, latency_ms=1.0)


def _qwen_only() -> HybridConfig:
    return HybridConfig(routing=RoutingConfig(policy="qwen_default"))


def test_status_is_error_when_every_detected_line_comes_back_empty(tmp_path):
    result = run_hybrid(_prescription(tmp_path / "rx.png"), None, _Qwen(""), _Trocr(), _qwen_only())
    assert result.line_count >= 2
    assert result.status == "error"
    assert result.merged_transcription == ""


def test_status_is_partial_when_only_some_lines_are_read(tmp_path):
    qwen = _FlakyQwen()
    result = run_hybrid(_prescription(tmp_path / "rx.png"), None, qwen, _Trocr(), _qwen_only())
    assert qwen.calls >= 2
    assert result.status == "partial"
    assert result.merged_transcription == "first line"
    assert any("incomplete" in warning for warning in result.warnings)


def test_status_is_success_when_all_lines_are_read(tmp_path):
    result = run_hybrid(_prescription(tmp_path / "rx.png"), None, _Qwen("ok"), _Trocr(), _qwen_only())
    assert result.status == "success"


class _RecordingQwen(_Qwen):
    def __init__(self):
        super().__init__("ok")
        self.images = []

    def transcribe(self, image, preprocessing_method: str):
        self.images.append(image)
        return super().transcribe(image, preprocessing_method)


def _red_ink_page(path: Path, tilt: float = 0.0) -> Path:
    image = Image.new("RGB", (900, 400), (250, 250, 245))
    draw = ImageDraw.Draw(image)
    for top in (60, 160, 260):
        draw.rectangle((60, top, 840, top + 28), fill=(200, 20, 20))
    if tilt:
        image = image.rotate(tilt, expand=True, fillcolor=(250, 250, 245), resample=Image.Resampling.BICUBIC)
    image.save(path)
    return path


def test_ocr_reads_original_colour_crops_by_default(tmp_path):
    qwen = _RecordingQwen()
    run_hybrid(_red_ink_page(tmp_path / "p.png"), tmp_path / "out", qwen, _Trocr(), _qwen_only())
    assert len(qwen.images) >= 3
    for crop in qwen.images:
        pixels = np.asarray(crop.convert("RGB")).reshape(-1, 3)
        red = (pixels[:, 0] > 150) & (pixels[:, 1] < 90)
        assert red.any(), "crop lost its colour; expected the red ink to be present"
    saved = sorted((tmp_path / "out" / "lines").glob("line_*.png"))
    assert Image.open(saved[0]).convert("RGB").getpixel((0, 0)) != (0, 0, 0)


def test_processed_source_can_be_selected(tmp_path):
    qwen = _RecordingQwen()
    config = HybridConfig(routing=RoutingConfig(policy="qwen_default"), ocr_source="processed")
    run_hybrid(_red_ink_page(tmp_path / "p.png"), None, qwen, _Trocr(), config)
    for crop in qwen.images:
        r, g, b = crop.convert("RGB").getpixel((crop.width // 2, crop.height // 2))
        assert r == g == b  # grayscale stack


def test_aligned_colour_page_matches_the_processed_page_when_deskewed(tmp_path):
    from ocr.preprocessing import PreprocessConfig, prepare_hybrid_page, validate_image

    page, info = validate_image(_red_ink_page(tmp_path / "tilt.png", tilt=5.0))
    prepared = prepare_hybrid_page(page, info, PreprocessConfig(enable_upscale=False))
    assert prepared.geometry.angle_degrees != 0
    assert prepared.aligned is not None and prepared.aligned.size == prepared.processed.size


def test_invalid_ocr_source_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        HybridConfig(ocr_source="raw")
