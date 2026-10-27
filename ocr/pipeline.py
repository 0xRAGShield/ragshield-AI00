"""End-to-end preprocessing and local handwritten OCR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from ocr.config import VARIANT_NAMES
from ocr.handwritten_ocr_client import HandwrittenOcrClient, Transcription
from ocr.preprocessing import (
    PreprocessConfig,
    Variant,
    build_variants,
    save_variants,
    select_variants,
    validate_image,
)


@dataclass
class CompareResult:
    source: str
    output_dir: str | None
    transcriptions: list[Transcription]
    variant_metadata: list[dict]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "output_dir": self.output_dir,
            "variant_metadata": self.variant_metadata,
            "transcriptions": [item.to_dict() for item in self.transcriptions],
        }


def prepare(
    path: str | Path,
    config: PreprocessConfig | None = None,
    variants: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[list[Variant], list[dict]]:
    image, info = validate_image(path)
    names = list(variants) if variants else None
    built = build_variants(image, info, config, include=names)
    chosen = select_variants(built, names)
    if output_dir is not None:
        save_variants(chosen, output_dir)
    return chosen, [variant.to_metadata() for variant in chosen]


def transcribe_variants(
    variants: Sequence[Variant],
    client: HandwrittenOcrClient,
) -> list[Transcription]:
    results = []
    for variant in variants:
        results.append(client.transcribe(variant.image, preprocessing_method=variant.name))
    return results


def compare_image(
    path: str | Path,
    client: HandwrittenOcrClient,
    variants: Sequence[str] | None = None,
    config: PreprocessConfig | None = None,
    output_dir: str | Path | None = None,
) -> CompareResult:
    chosen_names = list(variants) if variants else list(VARIANT_NAMES)
    prepared, metadata = prepare(path, config=config, variants=chosen_names, output_dir=output_dir)
    transcriptions = transcribe_variants(prepared, client)
    if output_dir is not None:
        destination = Path(output_dir) / "transcriptions.json"
        destination.write_text(
            json.dumps([item.to_dict() for item in transcriptions], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return CompareResult(
        source=str(path),
        output_dir=str(output_dir) if output_dir else None,
        transcriptions=transcriptions,
        variant_metadata=metadata,
    )


def image_from_variant(variant: Variant) -> Image.Image:
    return Image.fromarray(variant.image)
