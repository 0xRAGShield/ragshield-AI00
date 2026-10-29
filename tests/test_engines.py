import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ocr.engines import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    STATUS_UNAVAILABLE,
    Engine,
    EngineResult,
    create_engine,
    engine_names,
    engine_status,
    register_engine,
)
from ocr.engines.base import EngineLine, join_rows, order_lines
from ocr.engines.paddle import PaddleEngine
from ocr.engines.qwen import QwenEngine
from ocr.engines.surya import SuryaEngine
from ocr.engines.text import html_to_text
from ocr.handwritten_ocr_client import Transcription


# ----------------------------------------------------------------- html_to_text
@pytest.mark.parametrize(
    "markup, expected",
    [
        ("<p>Hello</p><p>World</p>", "Hello\nWorld"),
        ("line one<br>line two", "line one\nline two"),
        ("<table><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>", "1\t2\n3\t4"),
        ("<math>x &amp; y</math>", "x & y"),
        ("<p>أموكسيسيلين   500 مجم</p>", "أموكسيسيلين 500 مجم"),
        ("", ""),
        (None, ""),
    ],
)
def test_html_to_text(markup, expected):
    assert html_to_text(markup) == expected


# ----------------------------------------------------------------- base class
class _Boom(Engine):
    name = "boom"

    def _recognize(self, image):
        raise RuntimeError("out of memory")


def test_a_crashing_engine_returns_an_error_result_instead_of_raising():
    result = _Boom().recognize_page(Image.new("RGB", (10, 10)))
    assert result.status == STATUS_ERROR and "out of memory" in result.error and result.latency_ms >= 0


def test_missing_packages_make_an_engine_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr("ocr.engines.base.find_spec", lambda name: None)
    result = PaddleEngine().recognize_page(Image.new("RGB", (10, 10)))
    assert result.status == STATUS_UNAVAILABLE and "paddleocr" in result.error


def test_order_lines_reads_rows_top_down_and_arabic_right_to_left():
    left = EngineLine("يسار", (0, 100, 100, 140))
    right = EngineLine("يمين", (500, 102, 600, 141))
    below = EngineLine("تحت", (0, 300, 100, 340))
    ordered = order_lines([below, left, right], "auto")
    assert [line.text for line in ordered] == ["يمين", "يسار", "تحت"]
    assert join_rows(ordered) == "يمين يسار\nتحت"
    latin = [EngineLine("left", (0, 100, 100, 140)), EngineLine("right", (500, 98, 600, 139))]
    assert join_rows(order_lines(latin[::-1], "auto")) == "left right"


# ----------------------------------------------------------------- PaddleOCR
class _FakePaddle:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs, self.seen, self.page = kwargs, [], None
        _FakePaddle.instances.append(self)

    def predict(self, array):
        self.seen.append(array)
        return [self.page] if self.page is not None else []


def _poly(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)


def _paddle(page, **options):
    _FakePaddle.instances.clear()
    engine = PaddleEngine(factory=_FakePaddle, **options)
    engine._engine().page = page
    return engine


def test_paddle_receives_a_bgr_array_and_project_friendly_options():
    engine = _paddle({"rec_texts": [], "rec_scores": [], "rec_polys": []}, lang="ar", ocr_version="PP-OCRv5", device="gpu:0")
    image = Image.new("RGB", (4, 3), (0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0))  # pure red
    engine.recognize_page(image)
    fake = _FakePaddle.instances[0]
    assert tuple(fake.seen[0][0, 0]) == (0, 0, 255)  # red arrives as BGR
    assert fake.kwargs["lang"] == "ar" and fake.kwargs["ocr_version"] == "PP-OCRv5" and fake.kwargs["device"] == "gpu:0"
    assert fake.kwargs["use_doc_orientation_classify"] is False and fake.kwargs["use_doc_unwarping"] is False


def test_paddle_lines_are_ordered_filtered_and_scored():
    page = {
        "rec_texts": ["تحت", "يسار", "   ", "يمين"],
        "rec_scores": [0.9, 0.8, 0.99, 0.6],
        "rec_polys": [_poly(0, 300, 100, 340), _poly(0, 100, 100, 140), _poly(0, 200, 50, 220), _poly(500, 102, 600, 141)],
    }
    result = _paddle(page).recognize_page(Image.new("RGB", (700, 400)))
    assert result.status == STATUS_SUCCESS
    assert result.text == "يمين يسار\nتحت"  # blank text dropped, Arabic row read right to left
    assert result.confidence == pytest.approx((0.9 + 0.8 + 0.6) / 3, abs=1e-4)
    assert result.lines[0].bbox == (500, 102, 600, 141)


def test_paddle_empty_page_is_a_success_with_a_warning():
    result = _paddle(None).recognize_page(Image.new("RGB", (50, 50)))
    assert result.status == STATUS_SUCCESS and result.text == "" and "no text found" in result.warnings


def test_paddle_failures_are_contained():
    engine = _paddle({"rec_texts": ["x"], "rec_scores": [1.0], "rec_polys": [_poly(0, 0, 10, 10)]})
    engine._engine().predict = lambda array: (_ for _ in ()).throw(RuntimeError("cuda error"))
    assert engine.recognize_page(Image.new("RGB", (50, 50))).status == STATUS_ERROR


# ----------------------------------------------------------------- Surya
def _block(order, html, bbox=(0, 0, 10, 10), confidence=0.9, label="Text", skipped=False, error=False):
    return SimpleNamespace(reading_order=order, html=html, bbox=list(bbox), confidence=confidence, label=label, skipped=skipped, error=error)


class _FakeSurya:
    def __init__(self, blocks):
        self.blocks, self.images = blocks, []

    def __call__(self, images):
        self.images.extend(images)
        return [SimpleNamespace(blocks=self.blocks, image_bbox=[0, 0, 100, 100])]


def test_surya_blocks_follow_reading_order_and_skip_pictures_and_errors():
    blocks = [
        _block(2, "<p>second</p>", (0, 50, 90, 80), 0.6),
        _block(0, "<h1>Title</h1>", (0, 0, 90, 20), 0.9),
        _block(1, "", label="Picture", skipped=True),
        _block(3, "<p>broken</p>", error=True),
        _block(4, "<table><tr><td>a</td><td>b</td></tr></table>", (0, 90, 90, 99.6), 0.8),
    ]
    result = SuryaEngine(predictor=_FakeSurya(blocks)).recognize_page(Image.new("RGB", (100, 100)))
    assert result.status == STATUS_SUCCESS
    assert result.text == "Title\nsecond\na\tb"
    assert result.confidence == pytest.approx(0.7667, abs=1e-3)
    assert result.lines[-1].bbox == (0, 90, 90, 100)
    assert any("block 3" in warning for warning in result.warnings)


def test_surya_starts_its_own_backend_lazily_and_stops_only_what_it_started(monkeypatch):
    monkeypatch.setattr("ocr.engines.base.find_spec", lambda name: object())
    events = []

    class _Manager:
        def __init__(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    class _Recognition(_FakeSurya):
        def __init__(self, manager):
            super().__init__([_block(0, "<p>hi</p>")])

    inference, recognition = ModuleType("surya.inference"), ModuleType("surya.recognition")
    inference.SuryaInferenceManager, recognition.RecognitionPredictor = _Manager, _Recognition
    monkeypatch.setitem(sys.modules, "surya", ModuleType("surya"))
    monkeypatch.setitem(sys.modules, "surya.inference", inference)
    monkeypatch.setitem(sys.modules, "surya.recognition", recognition)

    engine = SuryaEngine()
    assert events == []  # constructing costs nothing
    assert engine.recognize_page(Image.new("RGB", (20, 20))).text == "hi"
    engine.recognize_page(Image.new("RGB", (20, 20)))
    engine.close()
    assert events == ["start", "stop"]  # started once, stopped once

    events.clear()
    external = SuryaEngine(manager=_Manager(), predictor=_FakeSurya([_block(0, "<p>x</p>")]))
    events.clear()
    external.close()
    assert events == []  # a manager we did not start is left running


def test_surya_declares_its_licence_and_backend_needs():
    engine = SuryaEngine()
    assert "Rail-M" in engine.license_note and "vLLM" in engine.description


# ----------------------------------------------------------------- Qwen + registry
class _Client:
    def __init__(self, ok=True):
        self.ok = ok

    def transcribe(self, image, preprocessing_method):
        if not self.ok:
            return Transcription(text=None, model="m", status="error", preprocessing_method=preprocessing_method, latency_ms=1, error="no checkpoint")
        return Transcription(text="line a\n\nline b", model="m", status="success", preprocessing_method=preprocessing_method,
                             latency_ms=1, confidence=0.88, quality_warnings=["note"])


def test_qwen_engine_wraps_the_client():
    result = QwenEngine(client=_Client()).recognize_page(Image.new("RGB", (20, 20)))
    assert result.status == STATUS_SUCCESS and [l.text for l in result.lines] == ["line a", "line b"]
    assert result.confidence == 0.88 and result.warnings == ["note"] and result.model == "m"
    failed = QwenEngine(client=_Client(ok=False)).recognize_page(Image.new("RGB", (20, 20)))
    assert failed.status == STATUS_ERROR and failed.error == "no checkpoint"


def test_registry_creates_known_engines_and_rejects_unknown_ones():
    assert {"qwen", "paddleocr", "surya"} <= set(engine_names())
    assert isinstance(create_engine("paddleocr", lang="en"), PaddleEngine)
    with pytest.raises(ValueError, match="Unknown engine"):
        create_engine("tesseract")


def test_a_new_engine_can_be_registered_and_is_listed(monkeypatch):
    from ocr.engines import registry

    monkeypatch.setattr(registry, "_FACTORIES", dict(registry._FACTORIES))  # leave no global state behind

    class _Echo(Engine):
        name, description = "echo", "test engine"

        def _recognize(self, image):
            return EngineResult(engine="echo", status=STATUS_SUCCESS, text="echo")

    register_engine("echo", _Echo)
    assert create_engine("echo").recognize_page(Image.new("RGB", (5, 5))).text == "echo"
    assert "echo" in {row["name"] for row in engine_status()}
