"""Synthetic medical-document images with known transcriptions.

The text is authored here and rendered into the image, so the ground truth is
the source string rather than a guess at a real patient document. Arabic is
shaped only for drawing. The saved transcription keeps logical Unicode.

Real photos under ``data/images`` are not copied into this set. They contain
patient information and do not have a character-checked reference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import arabic_reshaper
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ENGLISH_REPORT = """CITY GENERAL HOSPITAL
Laboratory Report
Patient: Sample Case 01
Date: 2024-03-15
Hemoglobin 13.2 g/dL
WBC 7.4 x10^9/L
Glucose 5.6 mmol/L
"""

ARABIC_PRESCRIPTION = """وصفة طبية
الاسم: حالة تجريبية
التاريخ: 2024-11-02
الدواء: أموكسيسيلين 500 مجم
الجرعة: 1 قرص كل 8 ساعات
لمدة 7 أيام
"""

MIXED_NOTE = """Medication list
Rosuvastatin 10 mg
قرص واحد مساءً
Date: 15/08/2024
HbA1c 6.4 %
"""

DOSAGE_SLIP = """Amoxicillin 500 mg
1 tablet every 8 hours
Metformin 850 mg
1 tablet twice daily
INR 2.5
Collected: 02/10/2024 08:30
"""


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError("No Arabic-capable TrueType font found for the sample dataset")


_RUN = re.compile(r"\s+|[A-Za-z0-9][A-Za-z0-9./:%+-]*|[\u0600-\u06FF]+|[^\s]")


def _arabic_pieces(line: str) -> list[str]:
    """Shape Arabic words, keep numbers left-to-right, then reverse run order.

    ``python-bidi`` reverses a whole date such as ``2024-11-02`` into ``02-11-2024``
    when it sits in an Arabic line. Tokens are shaped separately so the ground-truth
    digits stay in the image.
    """
    pieces = []
    for part in _RUN.findall(line):
        if any("\u0600" <= ch <= "\u06ff" for ch in part):
            pieces.append(get_display(arabic_reshaper.reshape(part)))
        else:
            pieces.append(part)
    pieces.reverse()
    return pieces


def _draw_text(text: str, size: int = 36, width: int = 1000) -> Image.Image:
    font = _font(size)
    rows: list[tuple[str, list[str]]] = []
    for line in text.strip("\n").split("\n"):
        if any("\u0600" <= ch <= "\u06ff" for ch in line):
            rows.append(("rtl", _arabic_pieces(line)))
        else:
            rows.append(("ltr", [line]))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def _piece_box(piece: str) -> tuple[int, int]:
        box = probe.textbbox((0, 0), piece, font=font)
        return box[2] - box[0], box[3] - box[1]

    heights = [max(_piece_box(piece)[1] for piece in pieces) for _direction, pieces in rows]
    line_gap = 16
    height = 80 + sum(heights) + line_gap * (len(rows) - 1)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = 40
    for (direction, pieces), line_height in zip(rows, heights):
        if direction == "ltr":
            draw.text((48, y), pieces[0], fill=(20, 20, 20), font=font)
        else:
            total = sum(_piece_box(piece)[0] for piece in pieces)
            x = width - 48 - total
            for piece in pieces:
                draw.text((x, y), piece, fill=(20, 20, 20), font=font)
                x += _piece_box(piece)[0]
        y += line_height + line_gap
    return image


def _degrade(image: Image.Image, kind: str, rng: np.random.Generator) -> Image.Image:
    if kind == "clean":
        return image
    if kind == "lowres_noisy":
        small = image.resize((image.width // 3, image.height // 3), Image.Resampling.BOX)
        array = np.asarray(small).astype(np.int16)
        noise = rng.normal(0, 18, size=array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
    if kind == "blur_rotate":
        blurred = image.filter(ImageFilter.GaussianBlur(radius=1.6))
        return blurred.rotate(8, expand=True, fillcolor=(255, 255, 255), resample=Image.Resampling.BICUBIC)
    if kind == "faint_uneven":
        faint = ImageEnhance.Contrast(image).enhance(0.45)
        faint = ImageEnhance.Brightness(faint).enhance(1.25)
        array = np.asarray(faint).astype(np.float32)
        gradient = np.linspace(0.55, 1.0, array.shape[1], dtype=np.float32)
        array *= gradient[None, :, None]
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    raise ValueError(kind)


SAMPLES = [
    ("en_report_clean", ENGLISH_REPORT, "clean", ["english", "report", "dosages", "decimals"]),
    ("ar_prescription_clean", ARABIC_PRESCRIPTION, "clean", ["arabic", "prescription", "dosage", "date"]),
    ("mixed_note_clean", MIXED_NOTE, "clean", ["mixed", "medication", "decimals", "date"]),
    ("en_report_lowres_noisy", ENGLISH_REPORT, "lowres_noisy", ["english", "low-resolution", "noisy"]),
    ("ar_prescription_blur_rotate", ARABIC_PRESCRIPTION, "blur_rotate", ["arabic", "blurry", "rotated"]),
    ("mixed_note_faint", MIXED_NOTE, "faint_uneven", ["mixed", "faint", "uneven-lighting"]),
    ("en_dosages", DOSAGE_SLIP, "clean", ["english", "medication", "dosages", "decimals", "date"]),
]


def write_dataset(directory: str | Path, seed: int = 0) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest_samples = []
    for sample_id, text, kind, conditions in SAMPLES:
        image = _degrade(_draw_text(text), kind, rng)
        image_name = f"{sample_id}.png"
        text_name = f"{sample_id}.txt"
        image.save(root / image_name, format="PNG", dpi=(150, 150))
        (root / text_name).write_text(text, encoding="utf-8")
        manifest_samples.append(
            {
                "id": sample_id,
                "image": image_name,
                "text_file": text_name,
                "conditions": conditions,
                "degradation": kind,
                "note": "Synthetic page. Ground truth is the authored text, not a transcription of a real record.",
            }
        )
    manifest = {
        "description": (
            "Synthetic medical pages for CER/WER experiments. "
            "Ground truth is the text used to render each page."
        ),
        "normalization_note": "Score with ocr.evaluation. Raw files in this folder are the references.",
        "samples": manifest_samples,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
