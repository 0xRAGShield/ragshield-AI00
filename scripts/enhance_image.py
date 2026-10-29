"""Write a few simple enhancement variants of one image for manual inspection.

Usage:
    python scripts/enhance_image.py path/to/page.jpg --out outputs/enhance
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/image_enhancement_tests"))
    parser.add_argument("--threshold", type=int, default=170, help="Cut-off for the black-and-white variant")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as opened:
        im = ImageOps.exif_transpose(opened).convert("RGB")
    gray = ImageOps.grayscale(im)

    contrast = ImageOps.autocontrast(gray, cutoff=1)
    contrast.save(args.out / "01_contrast.png")
    ImageEnhance.Sharpness(contrast).enhance(1.8).save(args.out / "02_sharpen.png")
    ImageOps.autocontrast(gray.filter(ImageFilter.MedianFilter(size=3)), cutoff=1).save(
        args.out / "03_denoise_contrast.png"
    )
    contrast.point(lambda p: 255 if p > args.threshold else 0).save(args.out / "04_threshold.png")

    print(f"Original size: {im.size}")
    print(f"Saved variants in: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
