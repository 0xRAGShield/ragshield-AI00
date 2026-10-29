"""Download the model checkpoints this project uses into the local cache.

Usage:
    python scripts/download_models.py                # both models
    python scripts/download_models.py --only trocr   # TrOCR only
    python scripts/download_models.py --dest /data/models

The destination defaults to OCR_MODELS_DIR / HANDWRITTEN_OCR_CACHE, or
~/.cache/ragshield-ocr/hf_cache. Needs: pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.config import TROCR_DIRNAME, TROCR_MODEL_ID, default_cache_dir, load_dotenv  # noqa: E402
from ocr.handwritten_ocr_client import MODEL_ID as QWEN_MODEL_ID  # noqa: E402
from ocr.handwritten_ocr_client import _CHECKPOINT_DIRNAME as QWEN_DIRNAME  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dest", type=Path, default=None, help="Cache directory (default: from env or ~/.cache)")
    parser.add_argument("--only", choices=("qwen", "trocr"), default=None, help="Download just one model")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Download any Hugging Face model (for example Qwen/Qwen2.5-VL-7B-Instruct) into <dest>/<org>__<name>.\n"
        "Then set HANDWRITTEN_OCR_MODEL to the same id.",
    )
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    root = args.dest or default_cache_dir()
    if args.model_id:
        local_dir = root / args.model_id.strip().replace("/", "__")
        print(f"Downloading {args.model_id} -> {local_dir}")
        snapshot_download(repo_id=args.model_id.strip(), local_dir=str(local_dir))
        print("Download completed.")
        return 0
    targets = {"qwen": (QWEN_MODEL_ID, QWEN_DIRNAME), "trocr": (TROCR_MODEL_ID, TROCR_DIRNAME)}
    for key, (repo_id, dirname) in targets.items():
        if args.only and args.only != key:
            continue
        local_dir = root / dirname
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print("Download completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
