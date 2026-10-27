"""Load caller-supplied text crops. This module does not detect text regions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ocr.preprocessing import SUPPORTED_SUFFIXES, crop_region

CROPS_MISSING = "crops_missing"
CROPS_PATH_NOT_FOUND = "crops_path_not_found"
CROPS_UNSUPPORTED = "crops_unsupported"
CROPS_JSON_MALFORMED = "crops_json_malformed"
CROPS_EMPTY = "crops_empty"
CROP_IMAGE_MISSING = "crop_image_missing"
CROP_BOX_INVALID = "crop_box_invalid"
CROP_ENTRY_INVALID = "crop_entry_invalid"


@dataclass
class CropSpec:
    crop_id: str
    image: Image.Image
    source: str
    box: tuple[int, int, int, int] | None = None
    language: str = "en"
    path: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "crop_id": self.crop_id,
            "source": self.source,
            "language": self.language,
            "box": list(self.box) if self.box else None,
        }
        if self.path:
            payload["path"] = self.path
        return payload


class CropLoadError(ValueError):
    """Raised when a crop file, box, or sidecar cannot be used."""

    def __init__(self, message: str, code: str = CROPS_UNSUPPORTED) -> None:
        super().__init__(message)
        self.code = code


def load_crops_for_image(
    source_path: str | Path,
    crops_path: str | Path | None = None,
) -> list[CropSpec]:
    """Return crops from a JSON sidecar, a crops directory, or a crop image.

    Does not invent bounding boxes. If nothing is supplied and no sidecar sits
    next to the source image, the list is empty (`crops_missing` at the caller).
    """
    source = Path(source_path)
    with Image.open(source) as opened:
        opened.load()
        page = opened.copy()

    specified = Path(crops_path) if crops_path else _sidecar_path(source)
    if specified is None:
        return []
    if not specified.exists():
        raise CropLoadError(
            f"Crops path not found: {specified}",
            code=CROPS_PATH_NOT_FOUND,
        )

    if specified.is_dir():
        return _load_crop_directory(specified, source)
    if specified.suffix.lower() == ".json":
        return _load_crop_json(specified, page, source)
    if specified.suffix.lower() in SUPPORTED_SUFFIXES:
        return [_load_crop_image(specified, source)]
    raise CropLoadError(
        f"Unsupported crops path: {specified}",
        code=CROPS_UNSUPPORTED,
    )


def _sidecar_path(source: Path) -> Path | None:
    candidate = source.with_suffix(".crops.json")
    return candidate if candidate.is_file() else None


def _load_crop_json(
    path: Path,
    page: Image.Image,
    source: Path,
) -> list[CropSpec]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CropLoadError(
            f"Malformed crop JSON at {path}: {exc}",
            code=CROPS_JSON_MALFORMED,
        ) from exc
    except OSError as exc:
        raise CropLoadError(
            f"Could not read crop JSON at {path}: {exc}",
            code=CROPS_JSON_MALFORMED,
        ) from exc

    rows = payload.get("crops", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise CropLoadError(
            "Crop JSON must be a list or an object with a 'crops' list",
            code=CROPS_JSON_MALFORMED,
        )

    crops: list[CropSpec] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CropLoadError(
                f"Crop entry {index} is not an object",
                code=CROP_ENTRY_INVALID,
            )
        crop_id = str(row.get("id") or row.get("crop_id") or f"crop-{index + 1}")
        language = str(row.get("language") or "en")
        if "box" in row:
            box = _as_box(row["box"], crop_id)
            try:
                image = crop_region(page, box)
            except ValueError as exc:
                raise CropLoadError(
                    f"Invalid crop coordinates for {crop_id}: {exc}",
                    code=CROP_BOX_INVALID,
                ) from exc
            crops.append(
                CropSpec(
                    crop_id=crop_id,
                    image=image,
                    source=str(source),
                    box=box,
                    language=language,
                    path=str(path),
                )
            )
            continue
        image_path = row.get("path") or row.get("image")
        if not image_path:
            raise CropLoadError(
                f"Crop {crop_id} needs a 'box' [left, top, right, bottom] "
                "or an image 'path'",
                code=CROP_ENTRY_INVALID,
            )
        crop_file = Path(image_path)
        if not crop_file.is_file():
            crop_file = path.parent / image_path
        crops.extend(
            _load_named_crop(crop_file, source, crop_id, language, row.get("box"))
        )
    if not crops:
        raise CropLoadError(
            f"Crop JSON at {path} listed no usable crops",
            code=CROPS_EMPTY,
        )
    return crops


def _load_crop_directory(directory: Path, source: Path) -> list[CropSpec]:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise CropLoadError(
            f"Crop directory has no image files: {directory}",
            code=CROPS_EMPTY,
        )
    return [_load_crop_image(path, source) for path in files]


def _load_crop_image(path: Path, source: Path) -> CropSpec:
    return _load_named_crop(path, source, path.stem, "en", None)[0]


def _load_named_crop(
    path: Path,
    source: Path,
    crop_id: str,
    language: str,
    box: object | None,
) -> list[CropSpec]:
    if not path.is_file():
        raise CropLoadError(
            f"Crop image not found: {path}",
            code=CROP_IMAGE_MISSING,
        )
    try:
        with Image.open(path) as opened:
            opened.load()
            image = opened.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise CropLoadError(
            f"Crop image could not be read: {path}: {exc}",
            code=CROP_IMAGE_MISSING,
        ) from exc
    parsed_box = _as_box(box, crop_id) if box is not None else None
    return [
        CropSpec(
            crop_id=crop_id,
            image=image.convert("RGB"),
            source=str(source),
            box=parsed_box,
            language=language,
            path=str(path),
        )
    ]


def _as_box(value: object, crop_id: str = "") -> tuple[int, int, int, int]:
    label = f" for {crop_id}" if crop_id else ""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise CropLoadError(
            f"Crop box{label} must be [left, top, right, bottom]",
            code=CROP_BOX_INVALID,
        )
    try:
        left, top, right, bottom = (int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise CropLoadError(
            f"Crop box{label} has non-integer coordinates: {value}",
            code=CROP_BOX_INVALID,
        ) from exc
    if not (left < right and top < bottom):
        raise CropLoadError(
            f"Crop box{label} is not ordered left<right and top<bottom: {value}",
            code=CROP_BOX_INVALID,
        )
    return left, top, right, bottom
