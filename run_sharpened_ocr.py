import json
from pathlib import Path
from PIL import Image
from ocr.handwritten_ocr_client import HandwrittenOcrClient, HandwrittenOcrConfig

image_path = Path("outputs/image_enhancement_tests/02_sharpen.png")
out_dir = Path("outputs/ocr_02_sharpen_direct")
out_dir.mkdir(parents=True, exist_ok=True)

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

with Image.open(image_path) as source:
    image = source.convert("RGB")

print(f"Input image: {image_path}")
print(f"Image size: {image.size}")
print("Starting local OCR. Please wait...")

client = HandwrittenOcrClient(HandwrittenOcrConfig.from_env())
result = client.transcribe(
    image=image,
    preprocessing_method="02_sharpen_direct",
)

(out_dir / "result.json").write_text(
    json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

if result.status == "success" and result.text is not None:
    (out_dir / "transcription.txt").write_text(
        result.text,
        encoding="utf-8",
    )
    print("\nOCR RESULT:\n")
    print(result.text)
else:
    print(f"\nOCR FAILED: {result.error}")

print(f"\nStatus: {result.status}")
print(f"Latency: {result.latency_ms / 1000:.1f} seconds")
print(f"Output directory: {out_dir.resolve()}")
