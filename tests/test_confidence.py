import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocr.confidence import LOW_TOKEN_LOGPROB, TokenStats, stats_from_logprobs
from ocr.handwritten_ocr_client import HandwrittenOcrClient, HandwrittenOcrConfig, Transcription
from ocr.hybrid import HybridConfig, make_orientation_scorer, review_reasons, run_hybrid
from ocr.preprocessing import ImageInfo, PreprocessConfig, apply_combined_preprocess
from ocr.routing import RoutingConfig
from ocr.trocr_client import TrocrCropResult

from test_geometry import text_page


# ------------------------------------------------------------------ token stats
def test_confidence_is_the_geometric_mean_of_token_probabilities():
    stats = stats_from_logprobs([math.log(0.9), math.log(0.4)])
    assert stats.confidence == pytest.approx(math.sqrt(0.9 * 0.4))
    assert stats.min_token_prob == pytest.approx(0.4)
    assert stats.low_fraction == 0.5  # 0.4 < 0.5


def test_nothing_recorded_means_no_confidence_not_zero():
    stats = TokenStats()
    assert stats.confidence is None and stats.min_token_prob is None and stats.low_fraction is None


def test_merging_bands_weights_by_token_count():
    a = stats_from_logprobs([math.log(1.0)] * 9)
    b = stats_from_logprobs([math.log(0.1)])
    assert a.merge(b).confidence == pytest.approx(0.1 ** (1 / 10))
    assert TokenStats().merge(TokenStats()).confidence is None


def test_low_threshold_is_half():
    assert LOW_TOKEN_LOGPROB == pytest.approx(math.log(0.5))


# ------------------------------------------------------------- client aggregation
def _client(tmp_path, per_call):
    client = HandwrittenOcrClient(HandwrittenOcrConfig(cache_dir=tmp_path))
    client._model = object()
    calls = iter(per_call)

    def fake_generate(image):
        text, logprobs = next(calls)
        client._last_stats = stats_from_logprobs(logprobs)
        return text

    client._generate = fake_generate
    return client


def test_transcription_carries_measured_confidence(tmp_path):
    client = _client(tmp_path, [("hello", [math.log(0.9)] * 3)])
    result = client.transcribe(Image.new("RGB", (400, 60), "white"), "x")
    assert result.confidence == pytest.approx(0.9)
    payload = result.to_dict()
    assert payload["confidence"] == 0.9 and "min_token_prob" in payload


def test_confidence_is_combined_across_bands(tmp_path):
    client = _client(tmp_path, [("top", [math.log(1.0)] * 9), ("bottom", [math.log(0.1)])])
    page = Image.new("RGB", (3000, 4000), "white")
    ImageDraw.Draw(page).rectangle((100, 100, 2900, 200), fill="black")
    ImageDraw.Draw(page).rectangle((100, 3000, 2900, 3100), fill="black")
    result = client.transcribe(page, "x")
    assert result.text == "top\nbottom"
    assert result.confidence == pytest.approx(0.1 ** (1 / 10))


def test_no_invented_confidence_when_the_model_reports_none(tmp_path):
    client = _client(tmp_path, [("hello", [])])
    result = client.transcribe(Image.new("RGB", (400, 60), "white"), "x")
    assert result.confidence is None and "confidence" not in result.to_dict()


# --------------------------------------------------------------- review reasons
def _reasons(raw="Amoxicillin", status="success", confidence=0.95, quality=(), line=(), fallback=False, **cfg):
    return review_reasons(raw=raw, status=status, confidence=confidence, quality_warnings=list(quality),
                          line_warnings=list(line), used_fallback=fallback, config=HybridConfig(**cfg))


def test_a_clean_confident_line_has_no_review_reason():
    assert _reasons() == []


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"raw": ""}, "failed_or_empty"),
        ({"status": "error"}, "failed_or_empty"),
        ({"confidence": 0.5}, "low_confidence"),
        ({"raw": "take 2 tablets"}, "contains_numbers_or_dosage"),
        ({"raw": "٢ حبة"}, "contains_numbers_or_dosage"),
        ({"raw": "Paracetamol mg"}, "contains_numbers_or_dosage"),
        ({"raw": "take [illegible] daily"}, "illegible_marker"),
        ({"quality": ["repeated phrase detected"]}, "possible_repetition"),
        ({"line": ["crop is tiny"]}, "line_detection_warning"),
        ({"fallback": True}, "engine_fallback"),
    ],
)
def test_each_concern_is_named(kwargs, expected):
    assert any(reason.startswith(expected) for reason in _reasons(**kwargs))


def test_missing_confidence_is_not_treated_as_low():
    assert _reasons(confidence=None) == []


def test_policy_all_flags_every_line():
    assert _reasons(review_policy="all")[0] == "policy_all"


def test_threshold_is_configurable_and_validated():
    assert _reasons(confidence=0.6, min_confidence=0.5) == []
    with pytest.raises(ValueError):
        HybridConfig(min_confidence=1.5)
    with pytest.raises(ValueError):
        HybridConfig(review_policy="some")


# ---------------------------------------------------------------- hybrid end to end
class _ScriptedQwen:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def transcribe(self, image, preprocessing_method):
        text, confidence = self.outputs.pop(0)
        return Transcription(text=text, raw_text=text, model="m", status="success",
                             preprocessing_method=preprocessing_method, latency_ms=1.0, confidence=confidence)


class _NoTrocr:
    def inspect(self):
        return type("S", (), {"status": "success", "error": None})()

    def transcribe_line(self, *a, **k):  # pragma: no cover - never routed here
        return TrocrCropResult(crop_id="x", status="error")


def _three_lines(path):
    image = Image.new("RGB", (900, 400), "white")
    draw = ImageDraw.Draw(image)
    for top in (60, 160, 260):
        draw.rectangle((60, top, 840, top + 28), fill="black")
    image.save(path)
    return path


def test_lines_are_flagged_individually_and_summarised(tmp_path):
    qwen = _ScriptedQwen([("Amoxicillin", 0.95), ("take 2 tablets", 0.92), ("unclear scrawl", 0.40)])
    config = HybridConfig(routing=RoutingConfig(policy="qwen_default"))
    result = run_hybrid(_three_lines(tmp_path / "p.png"), tmp_path / "out", qwen, _NoTrocr(), config)
    by_text = {line.raw_text: line for line in result.lines}
    assert by_text["Amoxicillin"].review_required is False and by_text["Amoxicillin"].review_reasons == []
    assert by_text["take 2 tablets"].review_reasons == ["contains_numbers_or_dosage"]
    assert any(r.startswith("low_confidence") for r in by_text["unclear scrawl"].review_reasons)
    assert result.lines_needing_review == 2
    assert result.mean_confidence == pytest.approx((0.95 + 0.92 + 0.40) / 3, abs=1e-3)
    assert result.review_required is True  # the page as a whole still needs a person
    assert "confidence=" in (tmp_path / "out" / "hybrid_report.html").read_text(encoding="utf-8")


# ------------------------------------------------------- orientation by recognizer
INFO = ImageInfo(path=None, width=0, height=0, mode="RGB", dpi=None)


def _banner_page() -> np.ndarray:
    page = text_page().copy()
    page[0:60, :, :] = 15  # a dark banner marks the top of the page
    return page


def _banner_on_top_scorer(calls):
    def score(array):
        gray = array.mean(axis=2)
        h = gray.shape[0]
        calls.append(array.shape)
        return 0.9 if gray[: h // 20].mean() < gray[-h // 20 :].mean() - 50 else 0.2
    return score


@pytest.mark.parametrize("k", [1, 3])
def test_sideways_page_is_turned_the_right_way_by_the_scorer(k):
    sideways = np.ascontiguousarray(np.rot90(_banner_page(), k))
    calls = []
    variant, geometry = apply_combined_preprocess(
        sideways, INFO, PreprocessConfig(enable_page_crop=False, enable_upscale=False, enable_deskew=False),
        _banner_on_top_scorer(calls))
    assert len(calls) == 2  # both quarter turns were tried
    assert geometry.rotation_k90 == (4 - k) % 4
    processed = variant.image
    assert processed[: processed.shape[0] // 20].mean() < processed[-processed.shape[0] // 20 :].mean() - 20


def test_upside_down_check_is_off_by_default_and_costs_nothing():
    flipped = np.ascontiguousarray(np.rot90(_banner_page(), 2))
    calls = []
    _, geometry = apply_combined_preprocess(
        flipped, INFO, PreprocessConfig(enable_page_crop=False, enable_upscale=False, enable_deskew=False),
        _banner_on_top_scorer(calls))
    assert calls == [] and geometry.rotation_k90 == 0


def test_upside_down_page_is_fixed_when_requested():
    flipped = np.ascontiguousarray(np.rot90(_banner_page(), 2))
    _, geometry = apply_combined_preprocess(
        flipped, INFO,
        PreprocessConfig(enable_page_crop=False, enable_upscale=False, enable_deskew=False, check_upside_down=True),
        _banner_on_top_scorer([]))
    assert geometry.rotation_k90 == 2


def test_an_upright_page_is_not_probed_even_with_a_scorer():
    calls = []
    apply_combined_preprocess(_banner_page(), INFO,
                              PreprocessConfig(enable_page_crop=False, enable_upscale=False), _banner_on_top_scorer(calls))
    assert calls == []


def test_unratable_candidates_fall_back_to_the_default_turn():
    sideways = np.ascontiguousarray(np.rot90(_banner_page(), 1))
    variant, geometry = apply_combined_preprocess(
        sideways, INFO, PreprocessConfig(enable_page_crop=False, enable_upscale=False, enable_deskew=False),
        lambda array: None)
    assert geometry.rotation_k90 == 1
    assert any("could not rate" in note for note in variant.notes)


def test_scorer_probe_is_downsized_and_uses_the_recognizer_confidence():
    seen = []

    class _Client:
        def transcribe(self, image, preprocessing_method):
            seen.append((image.size, preprocessing_method))
            return Transcription(text="x", model="m", status="success", preprocessing_method=preprocessing_method,
                                 latency_ms=1.0, confidence=0.7)

    score = make_orientation_scorer(_Client(), probe_side=800)
    assert score(np.zeros((2000, 3000, 3), dtype=np.uint8)) == 0.7
    assert max(seen[0][0]) == 800 and seen[0][1] == "orientation_probe"


# ---------------------------------------------------------- torch-only (skipped in CI)
def test_recorder_matches_a_manual_log_softmax():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ocr.confidence import make_recorder

    recorder, processors = make_recorder()
    assert processors is not None
    scores = torch.tensor([[2.0, 0.0, -1.0]])
    assert recorder(None, scores) is scores  # never alters the scores
    expected = torch.log_softmax(scores, dim=-1).max().item()
    assert recorder.stats().total_logprob == pytest.approx(expected)
