from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


# Recognition backends. Default remains the existing Qwen full-page client.
RECOGNITION_MODES = ("qwen", "trocr", "trocr-full-page", "compare", "hybrid")
MODE_TROCR_FULL_PAGE = "trocr-full-page"
MODE_HYBRID = "hybrid"

# English handwritten TrOCR. This is not an Arabic OCR model.
TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"
DEFAULT_TROCR_CHECKPOINT_DIR = Path(
    r"D:\OCR_Models\hf_cache\microsoft__trocr-base-handwritten"
)

# Preprocessing variants the local OCR path can select.

VARIANT_NAMES = (
    "original",
    "grayscale",
    "denoised",
    "contrast",
    "otsu",
    "adaptive",
    "illumination",
    "sharpened",
    "deskewed",
    "denoised_clahe",
    "clahe_sharpened",
    "combined_gray",
    "combined_otsu",
    "combined_adaptive",
)
