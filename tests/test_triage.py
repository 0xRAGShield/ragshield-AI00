import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from ocr.triage import BUCKETS, run_triage
from test_geometry import rotate_content, text_page


@pytest.fixture(scope="module")
def triaged(tmp_path_factory):
    """Build the sample folder and triage it once; the model-free tests share the result."""
    base = tmp_path_factory.mktemp("triage")
    root = base / "in"
    root.mkdir()
    Image.fromarray(text_page(1600, 2200)).save(root / "clean.png")
    Image.fromarray(text_page(1600, 2200)).filter(ImageFilter.GaussianBlur(7)).save(root / "blurry.png")
    Image.fromarray((text_page(1600, 2200) * 0.25).astype(np.uint8)).save(root / "dark.png")
    Image.fromarray(text_page(300, 400, size=14)).save(root / "tiny.png")
    Image.fromarray(np.ascontiguousarray(np.rot90(text_page(1600, 2200), 1))).save(root / "sideways.png")
    Image.fromarray(rotate_content(text_page(1600, 2200), 14)).save(root / "tilted.png")
    Image.new("RGB", (1600, 2200), (250, 250, 250)).save(root / "blank.png")
    (root / "corrupt.jpg").write_bytes(b"not an image")
    out = base / "out"
    return run_triage(root, out), out


@pytest.fixture
def one_page(tmp_path) -> Path:
    root = tmp_path / "in"
    root.mkdir()
    Image.fromarray(text_page(1000, 1400)).save(root / "clean.png")
    return root


def _rows(report):
    return {row.image: row for row in report.rows}


def test_a_clean_page_is_ok_and_problem_pages_are_explained(triaged):
    rows = _rows(triaged[0])
    assert rows["clean.png"].bucket == "ok" and rows["clean.png"].lines == 30
    assert rows["blurry.png"].bucket == "input_quality" and any("blurry" in n for n in rows["blurry.png"].notes)
    assert rows["dark.png"].bucket == "input_quality" and any("dark" in n for n in rows["dark.png"].notes)
    assert rows["tiny.png"].bucket == "input_quality" and any("small image" in n for n in rows["tiny.png"].notes)
    assert rows["corrupt.jpg"].bucket == "unreadable_file"


def test_tilt_is_corrected_not_mistaken_for_a_sideways_page(triaged):
    row = _rows(triaged[0])["tilted.png"]
    assert row.rotation_deg == 0 and row.skew_degrees == pytest.approx(-14, abs=1.0) and row.lines == 30


def test_a_sideways_page_is_turned_and_still_yields_its_lines(triaged):
    row = _rows(triaged[0])["sideways.png"]
    assert row.rotation_deg == 90 and row.lines == 30
    assert any("sideways" in note for note in row.notes)


def test_outputs_are_written_and_readable(triaged):
    report, out = triaged
    assert set(report.counts()) == set(BUCKETS)
    payload = json.loads((out / "triage.json").read_text(encoding="utf-8"))
    assert len(payload["images"]) == 8 and payload["counts"]["unreadable_file"] >= 1
    rows = list(csv.DictReader((out / "triage.csv").open(encoding="utf-8-sig")))
    assert {row["image"] for row in rows} >= {"clean.png", "corrupt.jpg"}
    assert "| bucket |" in (out / "triage_summary.md").read_text(encoding="utf-8")
    assert (out / "overlays" / "clean.jpg").is_file()
    assert Image.open(out / "overlays" / "clean.jpg").size[0] <= 1400


class _Result:
    def __init__(self, status="success", confidence=0.9, text="hello", lines=3, empty=0):
        self.status, self.mean_confidence, self.merged_transcription = status, confidence, text
        self.lines_needing_review = 1
        self.lines = [type("L", (), {"raw_text": "" if i < empty else "x"})() for i in range(lines)]


def test_recognition_results_and_cer_are_recorded(one_page, tmp_path):
    (one_page / "clean.txt").write_text("hello", encoding="utf-8")
    rows = _rows(run_triage(one_page, tmp_path / "out", runner=lambda path: _Result(text="hello")))
    clean = rows["clean.png"]
    assert clean.cer == 0.0 and clean.mean_confidence == 0.9 and clean.bucket == "ok"


def test_bad_recognition_on_a_clean_page_is_a_recognition_problem(one_page, tmp_path):
    (one_page / "clean.txt").write_text("something completely different", encoding="utf-8")
    rows = _rows(run_triage(one_page, tmp_path / "out", runner=lambda path: _Result(text="hello", confidence=0.5)))
    assert rows["clean.png"].bucket == "recognition"
    assert any("CER" in n for n in rows["clean.png"].notes) and any("confidence" in n for n in rows["clean.png"].notes)


def test_a_crashing_recognizer_does_not_stop_the_run(one_page, tmp_path):
    def boom(path):
        raise RuntimeError("out of memory")

    Image.fromarray(text_page(1000, 1400)).save(one_page / "second.png")
    report = run_triage(one_page, tmp_path / "out", runner=boom)
    assert len(report.rows) == 2
    assert _rows(report)["clean.png"].ocr_status == "error"


def test_pages_that_fail_detection_are_named_as_such(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    noise = np.random.default_rng(0).integers(0, 255, (1200, 1000, 3), dtype=np.uint8)
    Image.fromarray(noise).save(root / "noise.png")
    row = run_triage(root, tmp_path / "out").rows[0]
    assert row.bucket in {"detection", "input_quality"}
