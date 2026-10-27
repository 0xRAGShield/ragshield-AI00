import io

from PIL import Image

from ocr.server import parse_multipart


def test_multipart_upload_fields() -> None:
    image = Image.new("RGB", (24, 24), "white")
    image.putpixel((2, 2), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    boundary = "----ocrtest"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="variant"\r\n\r\noriginal\r\n',
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="variant"\r\n\r\ncombined\r\n',
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="image"; filename="scan.png"\r\n',
            b"Content-Type: image/png\r\n\r\n",
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    fields, files = parse_multipart(body, f"multipart/form-data; boundary={boundary}")
    assert fields["variant"] == ["original", "combined"]
    assert files["image"]["filename"] == "scan.png"
    assert files["image"]["data"] == payload
