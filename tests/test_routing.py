from PIL import Image, ImageDraw

from ocr.routing import (
    ENGINE_QWEN,
    ENGINE_TROCR,
    RoutingConfig,
    classify_recognized_script,
    looks_implausible,
    route_line,
    should_fallback_to_qwen,
)


def _wide_line():
    image = Image.new("RGB", (180, 24), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 10, 176, 14), fill="black")
    return image


def test_arabic_mixed_english_numeric_and_ambiguous_text():
    assert classify_recognized_script("الاسم محمد") == "arabic"
    assert classify_recognized_script("Amoxil 500 mg مع ماء") == "mixed"
    assert classify_recognized_script("Amoxicillin") == "latin"
    assert classify_recognized_script("500") == "numeric"
    assert classify_recognized_script("   ") == "empty"


def test_qwen_default_and_english_geometry_heuristic():
    wide = _wide_line()
    qwen = route_line(wide, RoutingConfig(policy="qwen_default"))
    assert qwen.engine == ENGINE_QWEN
    assert qwen.review_required is True

    heuristic = route_line(wide, RoutingConfig(policy="english_line_heuristic"))
    assert heuristic.engine == ENGINE_TROCR
    assert "not a validated" in heuristic.reason.lower() or "heuristic" in heuristic.reason.lower()

    tiny = Image.new("RGB", (10, 10), "white")
    empty = route_line(tiny)
    assert empty.engine == ENGINE_QWEN
    assert empty.review_required is True


def test_fallback_after_trocr_failure_and_implausible():
    assert should_fallback_to_qwen(None, "checkpoint_missing") is True
    assert should_fallback_to_qwen("الاسم", "success") is True
    assert should_fallback_to_qwen("Amoxil", "success") is False
    assert looks_implausible("aaaaaaa") is True
    assert looks_implausible("Amoxil 500 mg") is False


def test_numeric_like_short_line_goes_to_qwen():
    image = Image.new("RGB", (80, 18), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 5, 70, 14), fill="black")
    decision = route_line(image)
    assert decision.engine == ENGINE_QWEN
    assert decision.review_required is True
