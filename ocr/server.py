"""Small local upload page. It does not use the RAG API."""

from __future__ import annotations

import html
import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ocr.config import VARIANT_NAMES
from ocr.handwritten_ocr_client import HandwrittenOcrClient, HandwrittenOcrConfig
from ocr.pipeline import CompareResult, prepare, transcribe_variants
from ocr.preprocessing import ImageValidationError, PreprocessConfig
from ocr.report import write_report

DEFAULT_VARIANTS = ("original", "combined_gray")

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Local handwritten OCR</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 32px auto; max-width: 960px; color: #1d1d1d; background: #f7f5f0; }}
    label {{ display: inline-block; margin: 0 12px 8px 0; }}
    img {{ max-width: 100%; height: auto; background: white; }}
    article {{ background: white; border: 1px solid #ddd; padding: 16px; margin-top: 16px; }}
    pre {{ white-space: pre-wrap; font-family: Consolas, monospace; }}
    .error {{ color: #8a1f1f; }}
    button {{ padding: 8px 14px; }}
  </style>
</head>
<body>
  <h1>Local handwritten OCR</h1>
  <p>Inference uses the checkpoint on this machine. Images are not uploaded. The first run loads a 3B model on CPU and can take several minutes.</p>
  <form method="post" enctype="multipart/form-data" action="/run">
    <p><input type="file" name="image" accept="image/png,image/jpeg,image/tiff,.tif,.tiff" required></p>
    <p>
      {checks}
    </p>
    <p><button type="submit">Preprocess and transcribe</button></p>
  </form>
  {body}
</body>
</html>
"""


def _checks(selected: set[str]) -> str:
    boxes = []
    for name in VARIANT_NAMES:
        checked = " checked" if name in selected else ""
        boxes.append(
            f'<label><input type="checkbox" name="variant" value="{name}"{checked}> {name}</label>'
        )
    return "\n".join(boxes)


def _page(body: str, selected: set[str] | None = None) -> bytes:
    chosen = selected if selected is not None else set(DEFAULT_VARIANTS)
    document = _PAGE.format(checks=_checks(chosen), body=body)
    return document.encode("utf-8")


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, dict]]:
    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type)
    if not match:
        raise ValueError("Missing multipart boundary")
    boundary = (match.group(1) or match.group(2)).strip().encode("utf-8")
    fields: dict[str, str] = {}
    files: dict[str, dict] = {}
    for part in body.split(b"--" + boundary):
        if b"Content-Disposition" not in part:
            continue
        header, _, data = part.partition(b"\r\n\r\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        header_text = header.decode("utf-8", errors="replace")
        name_match = re.search(r'name="([^"]+)"', header_text)
        if not name_match:
            continue
        name = name_match.group(1)
        file_match = re.search(r'filename="([^"]*)"', header_text)
        if file_match and file_match.group(1):
            filename = Path(file_match.group(1)).name
            files[name] = {"filename": filename, "data": data}
        else:
            fields.setdefault(name, data.decode("utf-8", errors="replace"))
            fields[name + "_list"] = fields.get(name + "_list", "")
    # Repeated fields are split again below; collect every value.
    return _collect_fields(body, boundary), files


def _collect_fields(body: bytes, boundary: bytes) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    files_ignored = 0
    for part in body.split(b"--" + boundary):
        if b"Content-Disposition" not in part:
            continue
        header, _, data = part.partition(b"\r\n\r\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        header_text = header.decode("utf-8", errors="replace")
        if "filename=" in header_text:
            files_ignored += 1
            continue
        name_match = re.search(r'name="([^"]+)"', header_text)
        if not name_match:
            continue
        fields.setdefault(name_match.group(1), []).append(data.decode("utf-8", errors="replace"))
    return fields


class App:
    """Upload page logic. The model is loaded once, on the first request that needs it."""

    def __init__(self, output_root: Path, client=None) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._client_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                self._client = HandwrittenOcrClient(HandwrittenOcrConfig.from_env())
            return self._client

    def handle_upload(self, body: bytes, content_type: str) -> bytes:
        _fields, files = parse_multipart(body, content_type)
        field_lists = _collect_fields(body, _boundary(content_type))
        variants = [name for name in field_lists.get("variant", []) if name in VARIANT_NAMES]
        if not variants:
            variants = list(DEFAULT_VARIANTS)
        upload = files.get("image")
        if upload is None or not upload["data"]:
            return _page("<p class=\"error\">Choose an image first.</p>", set(variants))
        suffix = Path(upload["filename"]).suffix.lower() or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            return _page("<p class=\"error\">Unsupported file type.</p>", set(variants))
        session = uuid.uuid4().hex
        directory = self.output_root / session
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / f"source{suffix}"
        source.write_bytes(upload["data"])
        try:
            # Preprocess once: every variant is saved so it can be shown, and only
            # the selected ones are sent to the model.
            prepared, metadata = prepare(
                source, config=PreprocessConfig(), variants=list(VARIANT_NAMES), output_dir=directory
            )
            chosen = [variant for variant in prepared if variant.name in variants]
            with self._inference_lock:
                transcriptions = transcribe_variants(chosen, self._get_client())
            (directory / "transcriptions.json").write_text(
                json.dumps([item.to_dict() for item in transcriptions], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result = CompareResult(
                source=str(source),
                output_dir=str(directory),
                transcriptions=transcriptions,
                variant_metadata=metadata,
            )
            write_report(result, directory)
        except (ImageValidationError, KeyError, ValueError) as exc:
            return _page(f"<p class=\"error\">{html.escape(str(exc))}</p>", set(variants))
        cards = []
        by_method = {item.preprocessing_method: item for item in result.transcriptions}
        for name in VARIANT_NAMES:
            image_path = directory / f"{name}.png"
            if not image_path.is_file():
                continue
            transcription = by_method.get(name)
            if transcription is None:
                text = "<p>Image only. This variant was not sent to the local model.</p>"
            elif transcription.status != "success":
                text = f"<pre class=\"error\">{html.escape(transcription.error or 'error')}</pre>"
            else:
                text = f"<pre>{html.escape(transcription.text or '')}</pre>"
            cards.append(
                f"<article><h2>{html.escape(name)}</h2>"
                f"<img src=\"/sessions/{session}/{name}.png\" alt=\"{html.escape(name)}\">"
                f"{text}</article>"
            )
        return _page("".join(cards), set(variants))


def _boundary(content_type: str) -> bytes:
    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type)
    if not match:
        raise ValueError("Missing multipart boundary")
    return (match.group(1) or match.group(2)).strip().encode("utf-8")


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                body = _page("")
                self._send(200, "text/html; charset=utf-8", body)
                return
            if parsed.path.startswith("/sessions/"):
                relative = parsed.path[len("/sessions/") :]
                path = (app.output_root / relative).resolve()
                if app.output_root.resolve() not in path.parents or not path.is_file():
                    self._send(404, "text/plain", b"Not found")
                    return
                kind = "image/png" if path.suffix == ".png" else "application/octet-stream"
                self._send(200, kind, path.read_bytes())
                return
            self._send(404, "text/plain", b"Not found")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/run":
                self._send(404, "text/plain", b"Not found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20 * 1024 * 1024:
                self._send(413, "text/html; charset=utf-8", _page("<p class=\"error\">Upload must be under 20 MB.</p>"))
                return
            body = self.rfile.read(length)
            try:
                page = app.handle_upload(body, self.headers.get("Content-Type", ""))
            except (ValueError, OSError) as exc:
                page = _page(f"<p class=\"error\">{html.escape(str(exc))}</p>")
            except Exception as exc:  # noqa: BLE001 - never drop the connection on a bad upload
                self._send(
                    500,
                    "text/html; charset=utf-8",
                    _page(f"<p class=\"error\">Unexpected error: {html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>"),
                )
                return
            self._send(200, "text/html; charset=utf-8", page)

        def log_message(self, fmt: str, *args) -> None:
            # Avoid writing request paths that may include uploaded filenames.
            return

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(host: str, port: int, output_root: Path) -> None:
    app = App(output_root)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"Local handwritten OCR at http://{host}:{port}")
    server.serve_forever()
