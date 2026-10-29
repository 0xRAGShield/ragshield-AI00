"""Detect pages that were photographed or scanned sideways.

Text lines produce strong row-wise ink peaks when the page is upright. When the
page is turned 90 degrees the peaks move to the columns. Comparing the two tells
us whether a quarter turn is needed. Which way to turn (and whether the page is
upside down) cannot be known from ink geometry alone, so this module reports
the ambiguity and lets the caller break the tie, for example with recognition
confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

_ANALYSIS_SIDE = 1000


@dataclass
class OrientationResult:
    k90: int  # counter-clockwise quarter turns to apply (0 or 1 by default)
    needs_rotation: bool
    ambiguous: bool  # a quarter turn is needed but clockwise vs counter-clockwise is a guess
    horizontal_score: float
    vertical_score: float
    notes: list[str] = field(default_factory=list)


def rotate_k90(image: np.ndarray, k: int) -> np.ndarray:
    """Rotate counter-clockwise by k quarter turns."""
    return np.ascontiguousarray(np.rot90(image, k % 4))


def undo_rot90_point(x: float, y: float, k: int, rotated_width: int, rotated_height: int) -> tuple[float, float]:
    """Map a point in the rotated frame back to the frame before ``rotate_k90(.., k)``."""
    width, height = float(rotated_width), float(rotated_height)
    for _ in range(k % 4):
        x, y = height - y, x
        width, height = height, width
    return x, y


def detect_orientation(gray: np.ndarray, ratio: float = 1.5) -> OrientationResult:
    """Decide whether the text runs along rows (upright) or along columns (sideways).

    Text lines make the ink profile alternate between line and gap. That structure
    is measured inside the occupied text block only (margins and empty halves of a
    page say nothing about line direction) and after compensating for tilt, so a
    slightly rotated page is not mistaken for a sideways one.
    """
    binary = _ink(gray)
    if binary is None:
        return OrientationResult(0, False, False, 0.0, 0.0, ["too little ink to judge orientation"])
    horizontal = _line_structure(binary)
    vertical = _line_structure(np.ascontiguousarray(binary.T))
    if vertical > ratio * max(horizontal, 1e-9):
        return OrientationResult(
            k90=1,
            needs_rotation=True,
            ambiguous=True,
            horizontal_score=horizontal,
            vertical_score=vertical,
            notes=[
                "text lines run vertically; page looks turned by 90 degrees. "
                "Direction (clockwise or counter-clockwise) is a guess without a recognizer score."
            ],
        )
    return OrientationResult(0, False, False, horizontal, vertical)


_TILT_ANGLES = tuple(range(-24, 25, 3))


def _line_structure(binary: np.ndarray) -> float:
    """Best row-profile line structure over small tilts (higher = clearer horizontal lines)."""
    height, width = binary.shape
    center = (width / 2.0, height / 2.0)
    best = 0.0
    for angle in _TILT_ANGLES:
        if angle == 0:
            rotated = binary
        else:
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(binary, matrix, (width, height), flags=cv2.INTER_NEAREST)
        best = max(best, _block_peakiness(rotated.sum(axis=1)))
    return best


def _block_peakiness(profile: np.ndarray) -> float:
    """Squared coefficient of variation of the profile inside the occupied span."""
    peak = float(profile.max())
    if peak <= 0:
        return 0.0
    occupied = np.nonzero(profile > 0.02 * peak)[0]
    if len(occupied) < 8:
        return 0.0
    block = profile[occupied[0] : occupied[-1] + 1]
    mean = float(block.mean())
    return float(block.var() / (mean * mean)) if mean > 0 else 0.0


def _ink(gray: np.ndarray) -> np.ndarray | None:
    small = gray
    longest = max(gray.shape[:2])
    if longest > _ANALYSIS_SIDE:
        factor = _ANALYSIS_SIDE / longest
        small = cv2.resize(gray, (max(1, int(gray.shape[1] * factor)), max(1, int(gray.shape[0] * factor))), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (3, 3), 0)
    _threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (binary > 0).astype(np.uint8)
    if int(ink.sum()) < 40:
        return None
    return ink
