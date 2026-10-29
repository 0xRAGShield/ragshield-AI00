import io
import re
from pathlib import Path

from PIL import Image, ImageDraw

from ocr.config import VARIANT_NAMES
from ocr.handwritten_ocr_client import Transcription
from ocr.server import DEFAULT_VARIANTS, App, _page


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, image, preprocessing_method: str) -> Transcription:
        self.calls.append(preprocessing_method)
        return Transcription(
            text=f"text from {preprocessing_method}",
            model="m",
            status="success",
            preprocessing_method=preprocessing_method,
            latency_ms=1.0,
        )


def _png_bytes() -> bytes:
    image = Image.new("RGB", (300, 120), "white")
    ImageDraw.Draw(image).rectangle((20, 30, 260, 50), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _multipart(variants: list[str], payload: bytes, boundary: str = "----t") -> tuple[bytes, str]:
    parts = []
    for name in variants:
        parts += [f"--{boundary}\r\n".encode(), b'Content-Disposition: form-data; name="variant"\r\n\r\n', name.encode(), b"\r\n"]
    parts += [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="image"; filename="scan.png"\r\n',
        b"Content-Type: image/png\r\n\r\n",
        payload,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def test_default_variants_are_real_variant_names() -> None:
    assert set(DEFAULT_VARIANTS) <= set(VARIANT_NAMES)
    page = _page("").decode("utf-8")
    assert re.search(r'value="combined_gray" checked', page)
    assert re.search(r'value="original" checked', page)


def test_upload_with_no_variants_uses_defaults_and_does_not_crash(tmp_path: Path) -> None:
    client = _FakeClient()
    app = App(tmp_path / "sessions", client=client)
    body, content_type = _multipart([], _png_bytes())
    html_page = app.handle_upload(body, content_type).decode("utf-8")
    assert "text from original" in html_page
    assert "text from combined_gray" in html_page
    assert sorted(client.calls) == sorted(DEFAULT_VARIANTS)


def test_only_selected_variants_are_sent_to_the_model(tmp_path: Path) -> None:
    client = _FakeClient()
    app = App(tmp_path / "sessions", client=client)
    body, content_type = _multipart(["grayscale"], _png_bytes())
    html_page = app.handle_upload(body, content_type).decode("utf-8")
    assert client.calls == ["grayscale"]
    assert "not sent to the local model" in html_page  # other variants shown as images only


def test_model_client_is_reused_across_requests(tmp_path: Path) -> None:
    client = _FakeClient()
    app = App(tmp_path / "sessions", client=client)
    body, content_type = _multipart(["original"], _png_bytes())
    app.handle_upload(body, content_type)
    app.handle_upload(body, content_type)
    assert client.calls == ["original", "original"]
    assert app._get_client() is client


def test_unreadable_image_returns_an_error_page(tmp_path: Path) -> None:
    app = App(tmp_path / "sessions", client=_FakeClient())
    body, content_type = _multipart(["original"], b"not an image")
    html_page = app.handle_upload(body, content_type).decode("utf-8")
    assert 'class="error"' in html_page
