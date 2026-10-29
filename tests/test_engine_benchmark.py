import csv
import json

import pytest
from PIL import Image

from ocr.cli import main
from ocr.engine_benchmark import run_engine_benchmark
from ocr.engines import STATUS_ERROR, STATUS_SUCCESS, Engine, EngineResult, registry
from test_geometry import text_page


class _Fixed(Engine):
    def __init__(self, name, answers, fail_on=(), license_note=""):
        self.name, self.answers, self.fail_on, self.license_note = name, answers, set(fail_on), license_note
        self.closed = False

    def _recognize(self, image):
        key = "wide" if image.width > 900 else "narrow"
        if key in self.fail_on:
            raise RuntimeError("engine crashed")
        return EngineResult(engine=self.name, status=STATUS_SUCCESS, text=self.answers[key], confidence=0.8)

    def close(self):
        self.closed = True


@pytest.fixture
def pages(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    Image.fromarray(text_page(1000, 1400)).save(root / "wide.png")      # 1000 px wide
    Image.fromarray(text_page(600, 800)).save(root / "narrow.png")      # 600 px wide
    Image.fromarray(text_page(600, 800)).save(root / "unlabelled.png")  # no ground truth
    (root / "wide.txt").write_text("hello world", encoding="utf-8")
    (root / "narrow.txt").write_text("good morning", encoding="utf-8")
    return root


def test_engines_are_compared_on_the_same_pages_with_failures_counted(pages, tmp_path):
    exact = _Fixed("exact", {"wide": "hello world", "narrow": "good morning"})
    flaky = _Fixed("flaky", {"wide": "hello world", "narrow": "x"}, fail_on={"narrow"})
    report = run_engine_benchmark(pages, tmp_path / "out", [exact, flaky])
    by_engine = {entry["engine"]: entry for entry in report.summary}
    assert report.scored_images == 2
    assert by_engine["exact"]["cer_all"] == 0 and by_engine["exact"]["failed"] == 0
    assert by_engine["flaky"]["failed"] == 2  # the crash on narrow.png and on the unlabelled narrow page
    # "good morning" is 12 characters; the flaky engine got none of them, so 12 of 23 characters are wrong.
    assert by_engine["flaky"]["cer_all"] == pytest.approx(12 / 23, abs=1e-3)
    assert by_engine["flaky"]["cer_micro"] == 0  # looks perfect if you ignore the failure; that is the trap


def test_outputs_are_written(pages, tmp_path):
    out = tmp_path / "out"
    run_engine_benchmark(pages, out, [_Fixed("exact", {"wide": "hello world", "narrow": "good morning"}, license_note="MIT")])
    rows = list(csv.DictReader((out / "engine_benchmark.csv").open(encoding="utf-8-sig")))
    assert {(r["image"], r["engine"]) for r in rows} >= {("wide.png", "exact"), ("narrow.png", "exact")}
    assert next(r for r in rows if r["image"] == "wide.png")["cer"] == "0.0"
    payload = json.loads((out / "engine_benchmark.json").read_text(encoding="utf-8"))
    assert payload["engines"] == ["exact"]
    assert (out / "pages" / "wide" / "exact.txt").read_text(encoding="utf-8") == "hello world"
    summary = (out / "engine_summary.md").read_text(encoding="utf-8")
    assert "| exact |" in summary and "MIT" in summary and "cer_all" in summary


def test_without_any_ground_truth_it_still_reports_speed_and_failures(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    Image.fromarray(text_page(600, 800)).save(root / "a.png")
    report = run_engine_benchmark(root, tmp_path / "out", [_Fixed("x", {"wide": "a", "narrow": "a"})])
    assert report.scored_images == 0 and "cer_all" not in report.summary[0] and report.summary[0]["images"] == 1


def test_an_unreadable_file_is_recorded_for_every_engine(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    (root / "bad.png").write_bytes(b"nope")
    report = run_engine_benchmark(root, tmp_path / "out", [_Fixed("a", {}), _Fixed("b", {})])
    assert [row.status for row in report.rows] == [STATUS_ERROR, STATUS_ERROR]


def test_aligned_preprocessing_hands_over_the_deskewed_colour_page(tmp_path):
    from test_geometry import rotate_content

    root = tmp_path / "in"
    root.mkdir()
    Image.fromarray(rotate_content(text_page(1000, 1400), 12)).save(root / "tilted.png")
    seen = []

    class _Spy(Engine):
        name = "spy"

        def _recognize(self, image):
            seen.append(image.mode)
            return EngineResult(engine="spy", status=STATUS_SUCCESS, text="x")

    run_engine_benchmark(root, tmp_path / "out", [_Spy()], preprocess="aligned")
    assert seen == ["RGB"]
    with pytest.raises(ValueError):
        run_engine_benchmark(root, tmp_path / "out2", [_Spy()], preprocess="magic")


# ------------------------------------------------------------------------- CLI
def test_cli_lists_engines(capsys):
    assert main(["engines"]) == 0
    out = capsys.readouterr().out
    assert "qwen" in out and "paddleocr" in out and "surya" in out and "Rail-M" in out


def test_cli_compare_engines_skips_unavailable_ones_and_closes_the_rest(pages, tmp_path, monkeypatch, capsys):
    fake = _Fixed("fake", {"wide": "hello world", "narrow": "good morning"})
    monkeypatch.setattr(registry, "_FACTORIES", {**registry._FACTORIES, "fake": lambda **kw: fake})
    monkeypatch.setattr("ocr.engines.base.find_spec", lambda name: None)  # paddleocr and surya look uninstalled
    code = main(["compare-engines", "--input", str(pages), "--out", str(tmp_path / "out"), "--engines", "paddleocr,fake"])
    captured = capsys.readouterr()
    assert code == 0 and "Skipping paddleocr" in captured.err and fake.closed
    assert '"cer_all": 0.0' in captured.out


def test_cli_compare_engines_fails_clearly_when_nothing_can_run(pages, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("ocr.engines.base.find_spec", lambda name: None)
    assert main(["compare-engines", "--input", str(pages), "--out", str(tmp_path / "o"), "--engines", "surya,paddleocr"]) == 2
    assert "No engine is available" in capsys.readouterr().err
