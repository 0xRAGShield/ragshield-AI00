import csv
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("compare_triage", Path(__file__).resolve().parents[1] / "scripts" / "compare_triage.py")
compare_triage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_triage)


def _write(directory: Path, rows):
    directory.mkdir()
    with (directory / "triage.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "bucket", "lines", "cer"])
        writer.writeheader()
        writer.writerows(rows)


def test_improvements_regressions_and_cer_are_reported(tmp_path, capsys):
    _write(tmp_path / "old", [
        {"image": "a.png", "bucket": "detection", "lines": "0", "cer": "0.9"},
        {"image": "b.png", "bucket": "ok", "lines": "10", "cer": "0.1"},
        {"image": "c.png", "bucket": "ok", "lines": "5", "cer": ""},
        {"image": "gone.png", "bucket": "ok", "lines": "1", "cer": ""},
    ])
    _write(tmp_path / "new", [
        {"image": "a.png", "bucket": "ok", "lines": "9", "cer": "0.3"},
        {"image": "b.png", "bucket": "detection", "lines": "10", "cer": "0.2"},
        {"image": "c.png", "bucket": "ok", "lines": "5", "cer": ""},
        {"image": "fresh.png", "bucket": "ok", "lines": "2", "cer": ""},
    ])
    assert compare_triage.main([str(tmp_path / "old"), str(tmp_path / "new")]) == 0
    out = capsys.readouterr().out
    assert "improved: 1   worsened: 1   unchanged: 1" in out
    assert "+ a.png: detection -> ok" in out and "- b.png: ok -> detection" in out
    assert "a.png: 0 -> 9" in out
    assert "0.500 -> 0.250" in out
    assert "gone.png" in out and "fresh.png" in out
