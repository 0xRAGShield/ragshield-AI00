import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from ocr.batch import discover_images, run_real_images
from ocr.handwritten_ocr_client import HandwrittenOcrClient, HandwrittenOcrConfig, Transcription
from ocr.preprocessing import PreprocessConfig


class _ScriptedClient:
    def __init__(self, outcomes: list[Transcription], cache_dir: Path) -> None:
        self.config = HandwrittenOcrConfig(cache_dir=cache_dir)
        self._outcomes = list(outcomes)
        self.calls: list[str] = []

    def ensure_ready(self) -> None:
        return None

    def transcribe(self, image, preprocessing_method: str) -> Transcription:
        self.calls.append(preprocessing_method)
        if not self._outcomes:
            raise AssertionError("unexpected extra image")
        return self._outcomes.pop(0)


def _page(path: Path) -> None:
    image = Image.new("RGB", (40, 30), (255, 255, 255))
    image.putpixel((4, 4), (0, 0, 0))
    image.save(path)


def test_discovers_images_sorted_and_ignores_other_files(tmp_path: Path) -> None:
    root = tmp_path / "images"
    (root / "nested").mkdir(parents=True)
    for name in ("b.png", "a.jpg", "nested/c.PNG"):
        Image.new("RGB", (20, 20), "white").save(root / name)
    (root / "notes.txt").write_text("ignore", encoding="utf-8")
    found = [path.relative_to(root).as_posix() for path in discover_images(root)]
    assert found == ["a.jpg", "b.png", "nested/c.PNG"]


def test_discover_images_raises_for_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_images(tmp_path / "does-not-exist")


@pytest.mark.skipif(not Path("data/images").is_dir(), reason="local data/images folder not present")
def test_discovers_local_data_images_in_sorted_order() -> None:
    names = [path.name for path in discover_images(Path("data/images"))]
    assert names == sorted(names)


def test_missing_checkpoint_records_not_run_without_fake_text(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _page(image_root / "page.png")
    before = (image_root / "page.png").read_bytes()
    client = HandwrittenOcrClient(HandwrittenOcrConfig(cache_dir=tmp_path / "missing"))
    report = run_real_images(image_root, tmp_path / "out", client, variant="original")
    assert report.model_ran_locally is False
    assert report.inference == "local-cpu"
    assert report.succeeded == 0
    assert report.not_run == 1
    assert "incomplete" in (report.blocker or "")
    assert "external service" in (report.blocker or "")
    assert not list((tmp_path / "out").rglob("transcription.txt"))
    summary = list(csv.DictReader((tmp_path / "out" / "summary.csv").open(encoding="utf-8")))
    assert summary[0]["image_path"] == "page.png"
    assert summary[0]["status"] == "not_run"
    assert summary[0]["preprocessing_method"] == "original"
    assert summary[0]["text_file"] == ""
    assert (image_root / "page.png").read_bytes() == before


def test_scripted_client_is_not_sent_to_a_remote_host(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _page(image_root / "page.png")

    class _Boom:
        config = HandwrittenOcrConfig(cache_dir=tmp_path / "unused")

        def ensure_ready(self) -> None:
            raise AssertionError("this test disables the probe")

        def transcribe(self, image, preprocessing_method: str) -> Transcription:
            return Transcription(
                text=None,
                model="sherif1313/Arabic-English-handwritten-OCR-v3",
                status="error",
                preprocessing_method=preprocessing_method,
                latency_ms=1.0,
                error="local failure",
            )

    report = run_real_images(image_root, tmp_path / "out", _Boom(), require_probe=False)
    assert report.inference == "local-cpu"
    assert "api.mistral.ai" not in report.base_url


def test_batch_saves_exact_text_and_continues_after_a_failure(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _page(image_root / "a.png")
    _page(image_root / "b.png")
    (image_root / "notes.txt").write_text("ignore me", encoding="utf-8")
    raw = "Amoxicillin 500 mg\n1 tablet every 8 hours"
    client = _ScriptedClient(
        [
            Transcription(
                text=raw,
                model="sherif1313/Arabic-English-handwritten-OCR-v3",
                status="success",
                preprocessing_method="original",
                latency_ms=12.5,
            ),
            Transcription(
                text=None,
                model="sherif1313/Arabic-English-handwritten-OCR-v3",
                status="error",
                preprocessing_method="original",
                latency_ms=4.0,
                error="decoder stopped",
            ),
        ],
        cache_dir=tmp_path / "cache",
    )
    report = run_real_images(
        image_root,
        tmp_path / "out",
        client,
        variant="original",
        preprocess_config=PreprocessConfig(enable_upscale=False, enable_deskew=False),
        require_probe=False,
    )
    assert report.model_ran_locally is True
    assert report.inference == "local-cpu"
    assert "api.mistral.ai" not in report.base_url
    assert report.discovered == 2
    assert report.succeeded == 1
    assert report.failed == 1
    assert client.calls == ["original", "original"]
    first, second = report.images
    saved = (tmp_path / "out" / first.text_file).read_text(encoding="utf-8")
    assert saved == raw
    payload = json.loads((tmp_path / "out" / first.result_file).read_text(encoding="utf-8"))
    assert payload["model"] == "sherif1313/Arabic-English-handwritten-OCR-v3"
    assert payload["preprocessing_method"] == "original"
    assert payload["latency_ms"] == 12.5
    assert "confidence" not in payload
    assert (tmp_path / "out" / "a.png" / "original.png").is_file()
    assert second.status == "error"
    assert second.error == "decoder stopped"
    assert second.text_file is None
    assert (tmp_path / "out" / second.result_file).is_file()
    rows = list(csv.DictReader((tmp_path / "out" / "summary.csv").open(encoding="utf-8")))
    assert [row["image_path"] for row in rows] == ["a.png", "b.png"]
    assert rows[1]["text_file"] == ""
