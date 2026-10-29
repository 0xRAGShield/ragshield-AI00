import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("compare_run_images", Path(__file__).resolve().parents[1] / "scripts" / "compare_run_images.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _write(directory: Path, images):
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps({"images": images}), encoding="utf-8")


def test_recognition_and_cer_changes_are_reported(tmp_path, capsys):
    _write(tmp_path / "old", [
        {"source": "a.png", "status": "error"},
        {"source": "b.png", "status": "success", "cer": 0.50},
        {"source": "c.png", "status": "success", "cer": 0.10},
        {"source": "d.png", "status": "success"},
        {"source": "gone.png", "status": "success"},
    ])
    _write(tmp_path / "new", [
        {"source": "a.png", "status": "success", "cer": 0.2},
        {"source": "b.png", "status": "success", "cer": 0.20},
        {"source": "c.png", "status": "success", "cer": 0.30},
        {"source": "d.png", "status": "error"},
        {"source": "fresh.png", "status": "success"},
    ])
    assert module.main([str(tmp_path / "old"), str(tmp_path / "new")]) == 0
    out = capsys.readouterr().out
    assert "Recognized: 3 -> 3" in out
    assert "now recognized:\n  a.png" in out and "no longer recognized:\n  d.png" in out
    assert "2 images that have ground truth" in out and "0.300 -> 0.250" in out  # a.png has no old CER, so it is excluded
    assert "better: 1   worse: 1" in out and "worse: c.png" in out
    assert "gone.png" in out and "fresh.png" in out


def test_without_ground_truth_it_says_how_to_get_cer(tmp_path, capsys):
    _write(tmp_path / "old", [{"source": "a.png", "status": "success"}])
    _write(tmp_path / "new", [{"source": "a.png", "status": "success"}])
    module.main([str(tmp_path / "old"), str(tmp_path / "new")])
    assert "No CER" in capsys.readouterr().out
