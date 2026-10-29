
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ocr.orientation import OrientationResult, detect_orientation, rotate_k90, undo_rot90_point
from ocr.page_crop import find_page_quad, warp_page

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
LANCZOS = Image.Resampling.LANCZOS
_DESKEW_ANALYSIS_SIDE = 1000


class ImageValidationError(ValueError):
    """The file is missing, unreadable, empty, or not a supported image."""


@dataclass
class PreprocessConfig:
    enable_upscale: bool = True
    target_dpi: float = 300.0
    max_target_dpi: float = 400.0
    max_scale: float = 2.0
    min_short_side_px: int = 1200
    assumed_dpi_when_missing: float = 96.0
    # Never upscale past this longest side. The OCR client downsizes large images anyway.
    max_output_side: int = 3000
    enable_denoise: bool = True
    denoise_method: str = "median"
    median_ksize: int = 3
    nlmeans_h: float = 7.0
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    enable_otsu: bool = True
    enable_adaptive: bool = True
    adaptive_block_size: int = 31
    adaptive_c: int = 10
    enable_deskew: bool = True
    deskew_min_degrees: float = 0.6
    deskew_max_degrees: float = 25.0
    deskew_coarse_step_degrees: float = 2.0
    deskew_step_degrees: float = 0.25
    # Quarter-turn detection for pages scanned or photographed sideways.
    enable_orientation: bool = True
    orientation_ratio: float = 1.5
    # Cut the sheet of paper out of a photo and flatten its perspective. Conservative:
    # only applied when the sheet is clearly brighter than its surroundings.
    enable_page_crop: bool = True
    enable_sharpen: bool = True
    enable_illumination: bool = True

    def __post_init__(self) -> None:
        if self.denoise_method not in {"median", "nlmeans"}:
            raise ValueError("denoise_method must be 'median' or 'nlmeans'")
        if self.median_ksize % 2 == 0 or self.median_ksize < 3:
            raise ValueError("median_ksize must be an odd integer >= 3")
        if self.max_scale < 1:
            raise ValueError("max_scale must be >= 1")
        if self.deskew_coarse_step_degrees <= 0 or self.deskew_step_degrees <= 0:
            raise ValueError("deskew steps must be positive")
        if self.orientation_ratio < 1.0:
            raise ValueError("orientation_ratio must be >= 1")
        if not 150 <= self.target_dpi <= self.max_target_dpi <= 600:
            raise ValueError(
                "target_dpi must sit between 150 and max_target_dpi (<= 600)"
            )


@dataclass
class ImageInfo:
    path: str | None
    width: int
    height: int
    mode: str
    dpi: tuple[float, float] | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PageGeometry:
    """Maps coordinates between the EXIF-oriented original and combined_gray."""

    scale: float
    angle_degrees: float
    oriented_width: int
    oriented_height: int
    scaled_width: int
    scaled_height: int
    processed_width: int
    processed_height: int
    # Optional stages applied to the EXIF-oriented page before scaling and deskew:
    # a perspective crop of the sheet of paper (oriented -> cropped frame), then
    # counter-clockwise quarter turns (cropped -> rotated frame).
    rotation_k90: int = 0
    rotated_width: int = 0
    rotated_height: int = 0
    page_homography: list[list[float]] | None = None  # oriented frame -> cropped frame
    cropped_width: int = 0
    cropped_height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def map_processed_point_to_oriented(self, x: float, y: float) -> tuple[float, float]:
        matrix, _, _ = rotation_expand_matrix(
            self.scaled_width,
            self.scaled_height,
            self.angle_degrees,
        )
        inverse = cv2.invertAffineTransform(matrix)
        scaled_x = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
        scaled_y = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
        if self.scale != 0:
            scaled_x, scaled_y = scaled_x / self.scale, scaled_y / self.scale
        if self.rotation_k90 % 4:
            scaled_x, scaled_y = undo_rot90_point(
                scaled_x, scaled_y, self.rotation_k90,
                self.rotated_width or self.cropped_width or self.oriented_width,
                self.rotated_height or self.cropped_height or self.oriented_height,
            )
        if self.page_homography is not None:
            back = np.linalg.inv(np.asarray(self.page_homography, dtype=np.float64))
            point = cv2.perspectiveTransform(np.array([[[scaled_x, scaled_y]]], dtype=np.float64), back)[0, 0]
            scaled_x, scaled_y = float(point[0]), float(point[1])
        return scaled_x, scaled_y

    def map_processed_box_to_oriented(
        self,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        mapped = [self.map_processed_point_to_oriented(x, y) for x, y in corners]
        xs = [point[0] for point in mapped]
        ys = [point[1] for point in mapped]
        left = int(np.floor(min(xs)))
        top = int(np.floor(min(ys)))
        right = int(np.ceil(max(xs)))
        bottom = int(np.ceil(max(ys)))
        left = max(0, min(left, max(self.oriented_width - 1, 0)))
        top = max(0, min(top, max(self.oriented_height - 1, 0)))
        right = max(left + 1, min(right, self.oriented_width))
        bottom = max(top + 1, min(bottom, self.oriented_height))
        return left, top, right, bottom


@dataclass
class PreparedPage:
    oriented: Image.Image
    processed: Image.Image
    variant: Variant
    geometry: PageGeometry
    # Original colours, scaled and deskewed exactly like ``processed`` (same size,
    # same coordinates) but without contrast/denoise/sharpen. Line boxes found on
    # ``processed`` can be cut from this image for recognition.
    aligned: Image.Image | None = None


@dataclass
class Variant:
    name: str
    image: np.ndarray
    steps: list[str]
    width: int
    height: int
    scale: float
    dpi: tuple[float, float] | None
    notes: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        return {
            "name": self.name,
            "steps": list(self.steps),
            "width": self.width,
            "height": self.height,
            "scale": self.scale,
            "dpi": list(self.dpi) if self.dpi else None,
            "notes": list(self.notes),
            "interpolation": "lanczos" if self.scale > 1 else None,
        }


def validate_image(path: str | Path) -> tuple[Image.Image, ImageInfo]:
    """Open an image without writing back to the source path."""
    source = Path(path)

    if not source.is_file():
        raise ImageValidationError(f"Image not found: {source}")

    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImageValidationError(
            f"Unsupported format {source.suffix!r}. "
            f"Expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )

    if source.stat().st_size == 0:
        raise ImageValidationError(f"Empty image file: {source}")

    try:
        with Image.open(source) as opened:
            opened.load()
            image = opened.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            f"Corrupted or unreadable image: {source}"
        ) from exc

    return _normalize_loaded(image, str(source))


def validate_array(
    image: Image.Image,
    label: str = "<memory>",
) -> tuple[Image.Image, ImageInfo]:
    return _normalize_loaded(image, label)


def _normalize_loaded(
    image: Image.Image,
    label: str | None,
) -> tuple[Image.Image, ImageInfo]:
    warnings: list[str] = []

    if getattr(image, "n_frames", 1) > 1:
        warnings.append("multi-frame image; only frame 0 is used")
        image.seek(0)

    if image.width < 8 or image.height < 8:
        raise ImageValidationError(
            f"Image is too small ({image.width}x{image.height}) to be a document"
        )

    dpi = _read_dpi(image)
    mode = image.mode

    if mode == "RGBA":
        image = _composite_on_white(image)
        warnings.append("RGBA flattened onto white; source file unchanged")
    elif mode == "LA":
        image = _composite_on_white(image.convert("RGBA"))
        warnings.append("LA flattened onto white; source file unchanged")
    elif mode == "P":
        image = image.convert("RGBA")
        image = _composite_on_white(image)
        warnings.append("palette image converted to RGB")
    elif mode == "CMYK":
        image = image.convert("RGB")
        warnings.append("CMYK converted to RGB")
    elif mode == "1":
        image = image.convert("L")
    elif mode in {"I", "I;16", "F"}:
        image = image.point(
            lambda value: value * (255.0 / max(image.getextrema()[1], 1))
        )
        image = image.convert("L")
        warnings.append(f"{mode} converted to 8-bit grayscale")
    elif mode in {"L", "RGB"}:
        pass
    else:
        try:
            image = image.convert("RGB")
            warnings.append(f"mode {mode} converted to RGB")
        except (ValueError, OSError) as exc:
            raise ImageValidationError(
                f"Unsupported image mode {mode}"
            ) from exc

    if image.mode == "RGBA":
        alpha = np.asarray(image.getchannel("A"))
        if int(alpha.max()) == 0:
            raise ImageValidationError("Image is fully transparent")

    array = np.asarray(image)

    if array.size == 0 or float(array.std()) < 1e-6:
        raise ImageValidationError("Image is empty (all pixels are identical)")

    info = ImageInfo(
        path=label,
        width=image.width,
        height=image.height,
        mode=image.mode,
        dpi=dpi,
        warnings=warnings,
    )

    return image, info


def _composite_on_white(image: Image.Image) -> Image.Image:
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.getchannel("A"))
    return background


def _read_dpi(image: Image.Image) -> tuple[float, float] | None:
    dpi = image.info.get("dpi")

    if (
        isinstance(dpi, tuple)
        and len(dpi) == 2
        and all(isinstance(value, (int, float)) and value > 1 for value in dpi)
    ):
        return float(dpi[0]), float(dpi[1])

    # TIFF resolution tags, in pixels per inch when the unit is inch.
    ifd = getattr(image, "tag_v2", None)

    if ifd is not None and 282 in ifd and 283 in ifd:
        unit = int(ifd.get(296, 2))
        x_res = float(ifd[282])
        y_res = float(ifd[283])

        if unit == 3:
            x_res *= 2.54
            y_res *= 2.54

        if x_res > 1 and y_res > 1:
            return x_res, y_res

    return None


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def upscale_if_needed(
    image: np.ndarray,
    dpi: tuple[float, float] | None,
    config: PreprocessConfig,
) -> tuple[np.ndarray, float, list[str]]:
    """Resample when resolution is below the target."""
    notes: list[str] = []

    if not config.enable_upscale:
        return image, 1.0, ["upscale disabled"]

    height, width = image.shape[:2]
    short_side = min(height, width)

    # A large image gains nothing from resampling, whatever its DPI tag says. Phone
    # photos often carry a 72 DPI tag; trusting it would blow them up to 2x and the
    # OCR client would then shrink them again.
    if short_side >= config.min_short_side_px:
        return image, 1.0, [
            f"short side {short_side}px already >= {config.min_short_side_px}px; "
            "upscaling skipped"
        ]

    if dpi is not None:
        source_dpi = min(dpi)

        if source_dpi >= config.target_dpi:
            return image, 1.0, [
                f"dpi {source_dpi:.0f} already at or above "
                f"{config.target_dpi:.0f}"
            ]

        desired = min(config.target_dpi, config.max_target_dpi) / source_dpi
        notes.append(f"source dpi {source_dpi:.1f}")
    else:
        if short_side >= config.min_short_side_px:
            return image, 1.0, [
                "no dpi metadata; short side already "
                f"{short_side}px so upscaling was skipped"
            ]

        desired = config.target_dpi / config.assumed_dpi_when_missing
        notes.append(
            "no dpi metadata; assumed "
            f"{config.assumed_dpi_when_missing:.0f} DPI "
            "for the scale decision only"
        )

    scale = float(min(config.max_scale, max(desired, 1.0)))
    scale = min(scale, config.max_output_side / max(height, width))

    if scale <= 1.01:
        return image, 1.0, notes + ["scale capped at 1"]

    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )

    pil = Image.fromarray(image)
    resized = pil.resize(new_size, resample=LANCZOS)

    notes.append(
        f"lanczos upscale x{scale:.2f} to {new_size[0]}x{new_size[1]}"
    )

    return np.asarray(resized), scale, notes


def denoise(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if config.denoise_method == "nlmeans":
        return cv2.fastNlMeansDenoising(
            gray,
            None,
            h=config.nlmeans_h,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    return cv2.medianBlur(gray, config.median_ksize)


def apply_clahe(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    tile = max(2, config.clahe_tile_size)

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(tile, tile),
    )

    return clahe.apply(gray)


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return binary


def adaptive_binarize(
    gray: np.ndarray,
    config: PreprocessConfig,
) -> np.ndarray:
    height, width = gray.shape[:2]
    block = config.adaptive_block_size

    if block % 2 == 0:
        block += 1

    max_block = min(height, width)

    if max_block % 2 == 0:
        max_block -= 1

    block = min(block, max_block)

    if block < 3:
        return otsu_binarize(gray)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        config.adaptive_c,
    )


def correct_illumination(gray: np.ndarray) -> np.ndarray:
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25,
        sigmaY=25,
    )

    background = np.maximum(background, 1)

    return cv2.divide(gray, background, scale=255)


def sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)

    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def deskew_angle(gray: np.ndarray, config: PreprocessConfig) -> float:
    """Return the correction angle in degrees. Zero means leave the image.

    Searches a coarse grid over +-deskew_max_degrees on a downscaled binary image,
    then refines around the best coarse angle. Much faster than rotating the
    full-size page at every step, and it can afford a wide range.
    """
    if not config.enable_deskew:
        return 0.0

    small = gray
    longest = max(gray.shape[:2])
    if longest > _DESKEW_ANALYSIS_SIDE:
        factor = _DESKEW_ANALYSIS_SIDE / longest
        small = cv2.resize(
            gray,
            (max(1, int(gray.shape[1] * factor)), max(1, int(gray.shape[0] * factor))),
            interpolation=cv2.INTER_AREA,
        )
    blur = cv2.GaussianBlur(small, (3, 3), 0)
    _threshold, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    if int(binary.sum()) < 255 * 40:
        return 0.0

    limit = config.deskew_max_degrees
    coarse = _best_angle(binary, _angle_grid(-limit, limit, config.deskew_coarse_step_degrees))
    fine = _best_angle(
        binary,
        _angle_grid(
            max(-limit, coarse - config.deskew_coarse_step_degrees),
            min(limit, coarse + config.deskew_coarse_step_degrees),
            config.deskew_step_degrees,
        ),
    )
    if abs(fine) < config.deskew_min_degrees:
        return 0.0
    return fine


def _angle_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [start + index * step for index in range(count + 1)]


def _best_angle(binary: np.ndarray, angles: list[float]) -> float:
    height, width = binary.shape[:2]
    center = (width / 2.0, height / 2.0)
    best_angle, best_score = 0.0, -1.0
    for angle in angles:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary, matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        score = float(rotated.sum(axis=1).astype(np.float64).var())
        # Strictly greater keeps the smallest |angle| on ties because the grid is ordered
        # from negative to positive; prefer the angle closest to zero explicitly.
        if score > best_score or (score == best_score and abs(angle) < abs(best_angle)):
            best_score, best_angle = score, float(angle)
    return best_angle


def rotation_expand_matrix(
    width: int,
    height: int,
    angle: float,
) -> tuple[np.ndarray, int, int]:
    """Return the 2x3 affine matrix used by rotate_expand, plus output size."""
    if abs(angle) < 1e-3:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return identity, int(width), int(height)

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(np.ceil(height * sin + width * cos))
    new_height = int(np.ceil(height * cos + width * sin))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return matrix.astype(np.float64), new_width, new_height


def rotate_expand(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by angle degrees and expand the canvas."""
    if abs(angle) < 1e-3:
        return image

    height, width = image.shape[:2]
    matrix, new_width, new_height = rotation_expand_matrix(width, height, angle)
    border = (255, 255, 255) if image.ndim == 3 else 255

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )




def build_variants(
    image: Image.Image,
    info: ImageInfo,
    config: PreprocessConfig | None = None,
    include: list[str] | tuple[str, ...] | None = None,
) -> list[Variant]:
    config = config or PreprocessConfig()

    # Correct EXIF orientation without changing the source file.
    oriented = ImageOps.exif_transpose(image)
    original = np.asarray(oriented).copy()
    gray = to_gray(original)

    variants: list[Variant] = [
        _variant(
            "original", original, ["decoded"], 1.0, info.dpi,
            list(info.warnings),
        ),
        _variant(
            "grayscale", gray, ["grayscale"], 1.0, info.dpi,
        ),
    ]

    if config.enable_denoise:
        denoised = denoise(gray, config)
        variants.append(
            _variant(
                "denoised", denoised,
                ["grayscale", config.denoise_method],
                1.0, info.dpi,
            )
        )
    else:
        denoised = gray

    if config.enable_clahe:
        contrast = apply_clahe(gray, config)
        variants.append(
            _variant(
                "contrast", contrast,
                ["grayscale", "clahe"], 1.0, info.dpi,
            )
        )
        denoised_clahe = apply_clahe(denoised, config)
        variants.append(
            _variant(
                "denoised_clahe", denoised_clahe,
                ["grayscale", config.denoise_method, "clahe"],
                1.0, info.dpi,
            )
        )
        clahe_sharpened = (
            sharpen(contrast) if config.enable_sharpen else contrast
        )
        if config.enable_sharpen:
            variants.append(
                _variant(
                    "clahe_sharpened", clahe_sharpened,
                    ["grayscale", "clahe", "unsharp"],
                    1.0, info.dpi,
                )
            )

    if config.enable_illumination:
        illumination = correct_illumination(gray)
        variants.append(
            _variant(
                "illumination", illumination,
                ["grayscale", "illumination"],
                1.0, info.dpi,
            )
        )

    if config.enable_sharpen:
        variants.append(
            _variant(
                "sharpened", sharpen(gray),
                ["grayscale", "unsharp"], 1.0, info.dpi,
            )
        )

    if config.enable_otsu:
        variants.append(
            _variant(
                "otsu", otsu_binarize(gray),
                ["grayscale", "otsu"], 1.0, info.dpi,
            )
        )

    if config.enable_adaptive:
        variants.append(
            _variant(
                "adaptive", adaptive_binarize(gray, config),
                ["grayscale", "adaptive-gaussian"], 1.0, info.dpi,
            )
        )

    # Build a deskew-only variant separately from the full pipeline.
    if config.enable_deskew:
        angle = deskew_angle(gray, config)
        deskewed = rotate_expand(gray, angle)
        steps = ["grayscale"]
        notes = []
        if abs(angle) >= config.deskew_min_degrees:
            steps.append(f"deskew:{angle:.1f}deg")
            notes.append("rotation uses an expanded white canvas")
        else:
            notes.append("deskew skipped: no significant skew detected")

        variants.append(
            _variant(
                "deskewed", deskewed, steps, 1.0, info.dpi, notes,
            )
        )

    # Combined pipeline includes scaling, illumination, denoising,
    # CLAHE, deskew, and sharpening when enabled in the config.
    combined = _combined_variant(original, info, config)
    combined.name = "combined_gray"
    variants.append(combined)

    if config.enable_otsu:
        variants.append(
            _variant(
                "combined_otsu",
                otsu_binarize(combined.image),
                combined.steps + ["otsu"],
                combined.scale, combined.dpi, combined.notes.copy(),
            )
        )

    if config.enable_adaptive:
        variants.append(
            _variant(
                "combined_adaptive",
                adaptive_binarize(combined.image, config),
                combined.steps + ["adaptive-gaussian"],
                combined.scale, combined.dpi, combined.notes.copy(),
            )
        )

    if not include:
        return variants

    return select_variants(variants, list(include))





def apply_combined_preprocess(
    original: np.ndarray,
    info: ImageInfo,
    config: PreprocessConfig,
) -> tuple[Variant, PageGeometry]:
    """Run the project's default combined grayscale stack (no Otsu/adaptive).

    Order: optional Lanczos upscale, grayscale, illumination, denoise, CLAHE,
    deskew with expanded-canvas rotation, then unsharp. This is the same body
    used by build_variants for combined_gray.
    """

    steps = ["grayscale"]
    notes: list[str] = []
    oriented_height, oriented_width = original.shape[:2]

    # 1. Cut the sheet out of a photo and flatten perspective. Done first so a dark
    #    table around the paper cannot confuse the orientation check.
    working = original
    homography = None
    if config.enable_page_crop:
        quad = find_page_quad(to_gray(working))
        if quad is not None:
            working, matrix = warp_page(working, quad)
            homography = matrix.tolist()
            steps.append("page-crop")
            notes.append("page cropped and flattened from a photo")
    cropped_height, cropped_width = working.shape[:2]

    # 2. Quarter turn for sideways pages.
    rotation_k90 = 0
    if config.enable_orientation:
        orientation = detect_orientation(to_gray(working), config.orientation_ratio)
        notes.extend(orientation.notes)
        if orientation.needs_rotation:
            rotation_k90 = orientation.k90
            working = rotate_k90(working, rotation_k90)
            steps.append(f"rotate:{rotation_k90 * 90}deg-ccw")
    rotated_height, rotated_width = working.shape[:2]

    scaled, scale, scale_notes = upscale_if_needed(
        working,
        info.dpi,
        config,
    )
    notes.extend(scale_notes)

    if scale > 1.0:
        steps.append("lanczos-upscale")

    gray = to_gray(scaled)
    scaled_height, scaled_width = gray.shape[:2]
    angle = 0.0

    if config.enable_illumination:
        gray = correct_illumination(gray)
        steps.append("illumination")

    if config.enable_denoise:
        gray = denoise(gray, config)
        steps.append(config.denoise_method)

    if config.enable_clahe:
        gray = apply_clahe(gray, config)
        steps.append("clahe")

    if config.enable_deskew:
        angle = deskew_angle(gray, config)

        if abs(angle) >= config.deskew_min_degrees:
            gray = rotate_expand(gray, angle)
            steps.append(f"deskew:{angle:.1f}deg")
            notes.append("rotation uses an expanded white canvas")
        else:
            angle = 0.0
            notes.append("deskew skipped: no significant skew detected")
    else:
        notes.append("deskew disabled")

    if config.enable_sharpen:
        gray = sharpen(gray)
        steps.append("unsharp")

    achieved_dpi = None
    if info.dpi is not None:
        achieved_dpi = (
            info.dpi[0] * scale,
            info.dpi[1] * scale,
        )

    processed_height, processed_width = gray.shape[:2]
    variant = _variant(
        "combined",
        gray,
        steps,
        scale,
        achieved_dpi or info.dpi,
        notes,
    )
    geometry = PageGeometry(
        scale=scale,
        angle_degrees=float(angle),
        oriented_width=int(oriented_width),
        oriented_height=int(oriented_height),
        scaled_width=int(scaled_width),
        scaled_height=int(scaled_height),
        processed_width=int(processed_width),
        processed_height=int(processed_height),
        rotation_k90=int(rotation_k90),
        rotated_width=int(rotated_width),
        rotated_height=int(rotated_height),
        page_homography=homography,
        cropped_width=int(cropped_width),
        cropped_height=int(cropped_height),
    )
    return variant, geometry


def align_color(original: np.ndarray, geometry: PageGeometry) -> np.ndarray:
    """Apply only the geometric steps (scale, deskew) of the combined pipeline to colour."""
    image = original
    if geometry.page_homography is not None:
        border = (255, 255, 255) if image.ndim == 3 else 255
        image = cv2.warpPerspective(
            image,
            np.asarray(geometry.page_homography, dtype=np.float64),
            (geometry.cropped_width, geometry.cropped_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
    if geometry.rotation_k90 % 4:
        image = rotate_k90(image, geometry.rotation_k90)
    if geometry.scale > 1.0:
        resized = Image.fromarray(image).resize(
            (geometry.scaled_width, geometry.scaled_height), resample=LANCZOS
        )
        image = np.asarray(resized)
    return rotate_expand(image, geometry.angle_degrees)


def prepare_hybrid_page(
    image: Image.Image,
    info: ImageInfo,
    config: PreprocessConfig | None = None,
) -> PreparedPage:
    """EXIF-orient the page, then apply combined_gray for detection and OCR crops."""

    config = config or PreprocessConfig()
    oriented = ImageOps.exif_transpose(image)
    if oriented.mode != "RGB":
        oriented = oriented.convert("RGB")
    array = np.asarray(oriented).copy()
    variant, geometry = apply_combined_preprocess(array, info, config)
    variant.name = "combined_gray"
    processed = Image.fromarray(variant.image)
    if processed.mode != "RGB":
        processed = processed.convert("RGB")
    aligned = Image.fromarray(align_color(array, geometry)).convert("RGB")
    if aligned.size != processed.size:  # cannot happen today; never mis-map boxes if it does
        aligned = None
    return PreparedPage(
        oriented=oriented,
        processed=processed,
        variant=variant,
        geometry=geometry,
        aligned=aligned,
    )


def _combined_variant(
    original: np.ndarray,
    info: ImageInfo,
    config: PreprocessConfig,
) -> Variant:
    """Build an enhanced grayscale image for handwritten OCR."""

    variant, _geometry = apply_combined_preprocess(original, info, config)
    return variant


def _variant(
    name: str,
    image: np.ndarray,
    steps: list[str],
    scale: float,
    dpi: tuple[float, float] | None,
    notes: list[str] | None = None,
) -> Variant:
    array = np.ascontiguousarray(image)
    height, width = array.shape[:2]

    return Variant(
        name=name,
        image=array,
        steps=steps,
        width=width,
        height=height,
        scale=scale,
        dpi=dpi,
        notes=notes or [],
    )


def select_variants(
    variants: list[Variant],
    names: list[str] | tuple[str, ...] | None,
) -> list[Variant]:
    if not names:
        return list(variants)

    by_name = {variant.name: variant for variant in variants}
    missing = [name for name in names if name not in by_name]

    if missing:
        known = ", ".join(by_name)
        raise KeyError(
            f"Unknown variant(s): {', '.join(missing)}. Known: {known}"
        )

    return [by_name[name] for name in names]


def save_variant(variant: Variant, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{variant.name}.png"
    image = Image.fromarray(variant.image)

    save_kwargs = {}

    if variant.dpi is not None:
        save_kwargs["dpi"] = variant.dpi

    image.save(path, format="PNG", **save_kwargs)

    meta_path = directory / f"{variant.name}.json"
    meta_path.write_text(
        json.dumps(variant.to_metadata(), indent=2),
        encoding="utf-8",
    )

    return path


def save_variants(
    variants: list[Variant],
    directory: str | Path,
) -> list[Path]:
    return [save_variant(variant, directory) for variant in variants]


def config_to_dict(config: PreprocessConfig) -> dict:
    return asdict(config)


def crop_region(
    image: Image.Image | np.ndarray,
    box: tuple[int, int, int, int],
    scale: float = 1.0,
) -> Image.Image:
    
    if scale < 1.0:
        raise ValueError("scale must be >= 1.0")

    if isinstance(image, np.ndarray):
        if image.ndim not in (2, 3):
            raise ImageValidationError(
                "Unsupported image array dimensions"
            )
        source = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        source = image.copy()
    else:
        raise TypeError(
            "image must be a PIL image or NumPy array"
        )

    source = ImageOps.exif_transpose(source).convert("RGB")

    left, top, right, bottom = box
    width, height = source.size

    if not (
        0 <= left < right <= width
        and 0 <= top < bottom <= height
    ):
        raise ValueError(
            f"Invalid crop box {box} for image size {width}x{height}"
        )

    cropped = source.crop((left, top, right, bottom))

    if scale > 1.0:
        new_size = (
            round(cropped.width * scale),
            round(cropped.height * scale),
        )
        cropped = cropped.resize(new_size, resample=LANCZOS)

    return cropped
