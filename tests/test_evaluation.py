import json
from pathlib import Path

from ocr.evaluation import (
    character_error_rate,
    load_dataset,
    normalize_for_metrics,
    score_predictions,
    word_error_rate,
)
from ocr.sample_dataset import write_dataset


def test_normalization_keeps_case_and_punctuation() -> None:
    assert normalize_for_metrics("HbA1c  6.4 %\n") == "HbA1c 6.4 %"
    assert normalize_for_metrics("أموكسيسيلين\u200b") == "أموكسيسيلين"


def test_cer_and_wer_known_values() -> None:
    assert character_error_rate("abc", "abc") == 0
    assert character_error_rate("abc", "axc") == 1 / 3
    assert word_error_rate("10 mg daily", "10 mg daily") == 0
    assert word_error_rate("10 mg daily", "10 mg") == 1 / 3
    assert character_error_rate("", "") == 0
    assert character_error_rate("", "x") == 1


def test_failures_are_excluded_from_cer_and_counted_in_failure_rate() -> None:
    samples = [{"id": "a", "text": "10 mg"}, {"id": "b", "text": "10 mg"}]
    predictions = [
        {"image_id": "a", "variant": "original", "status": "success", "text": "10 mg", "latency_ms": 10},
        {"image_id": "b", "variant": "original", "status": "error", "text": None, "error": "down", "latency_ms": 30},
    ]
    report = score_predictions(samples, predictions)
    summary = report.variants[0]
    assert summary.cer == 0
    assert summary.failure_rate == 0.5
    assert summary.mean_latency_ms == 20
    assert summary.scored == 1


def test_dataset_ground_truth_roundtrip(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    samples = load_dataset(tmp_path)
    by_id = {sample["id"]: sample for sample in samples}
    assert "أموكسيسيلين 500 مجم" in by_id["ar_prescription_clean"]["text"]
    assert "13.2 g/dL" in by_id["en_report_clean"]["text"]
    assert "Rosuvastatin 10 mg" in by_id["mixed_note_clean"]["text"]
    assert Path(by_id["en_report_lowres_noisy"]["image_path"]).is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == 7
