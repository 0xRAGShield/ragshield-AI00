"""Run the local Qwen OCR on one image and write result.json / transcription.txt.

Usage:
    python scripts/ocr_single_image.py path/to/image.png --out outputs/ocr_single
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from ocr.handwritten_ocr_client import HandwrittenOcrClient, HandwrittenOcrConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/ocr_single"))
    parser.add_argument("--label", default="direct", help="preprocessing_method label stored in the result")
    args = parser.parse_args(argv)

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    args.out.mkdir(parents=True, exist_ok=True)

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    print(f"Input image: {args.image}  size={image.size}")
    print("Starting local OCR. Please wait...")

    client = HandwrittenOcrClient(HandwrittenOcrConfig.from_env())
    result = client.transcribe(image=image, preprocessing_method=args.label)

    (args.out / "result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result.status == "success" and result.text is not None:
        (args.out / "transcription.txt").write_text(result.text, encoding="utf-8")
        print("\nOCR RESULT:\n")
        print(result.text)
    else:
        print(f"\nOCR FAILED: {result.error}")

    print(f"\nStatus: {result.status}\nLatency: {result.latency_ms / 1000:.1f} seconds")
    print(f"Output directory: {args.out.resolve()}")
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
