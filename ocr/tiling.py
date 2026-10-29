"""Split tall pages into horizontal bands at blank rows.

A vision-language model sees one image at a limited resolution. Squeezing a whole
phone photo into that budget shrinks handwriting until it is unreadable. Instead
the page is cut into a few bands, each sent at a usable scale. Cuts are placed on
the emptiest rows near an even split, so a line of text is not sliced in half.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# A page whose scaled height is within this factor of the limit stays in one piece.
SINGLE_BAND_TOLERANCE = 1.15
_PROFILE_WIDTH = 1000


def needs_tiling(width: int, height: int, max_side: int, tolerance: float = SINGLE_BAND_TOLERANCE) -> bool:
    if max_side < 64 or width < 1 or height < 1:
        return False
    scale = min(1.0, max_side / width)
    return height * scale > max_side * tolerance


def plan_bands(
    gray: np.ndarray,
    max_side: int,
    tolerance: float = SINGLE_BAND_TOLERANCE,
) -> list[tuple[int, int]]:
    """Return (top, bottom) row ranges covering the page, in original pixels.

    The page is scaled so its width fits ``max_side``; the number of bands is the
    smallest that keeps every band within ``max_side`` tall at that scale.
    """
    height, width = gray.shape[:2]
    if not needs_tiling(width, height, max_side, tolerance):
        return [(0, height)]

    scale = min(1.0, max_side / width)
    count = math.ceil(height * scale / max_side)
    band_height = height / count
    profile = _smoothed_ink_rows(gray)
    to_profile = len(profile) / height  # original row -> profile row

    cuts: list[int] = [0]
    window = 0.2 * band_height
    for index in range(1, count):
        target = index * band_height
        low = max(cuts[-1] + int(0.3 * band_height), int(target - window))
        high = min(int(target + window), height - int(0.3 * band_height))
        if high <= low:
            cuts.append(int(round(target)))
            continue
        rows = np.arange(low, high + 1)
        ink = profile[np.clip((rows * to_profile).astype(int), 0, len(profile) - 1)]
        # Emptiest row wins; the distance penalty keeps ties near the even split.
        penalty = np.abs(rows - target) / max(window, 1.0) * (ink.max() * 0.02 + 1e-6)
        cuts.append(int(rows[int(np.argmin(ink + penalty))]))
    cuts.append(height)
    return [(top, bottom) for top, bottom in zip(cuts, cuts[1:]) if bottom - top > 0]


def _smoothed_ink_rows(gray: np.ndarray) -> np.ndarray:
    small = gray
    if gray.shape[1] > _PROFILE_WIDTH:
        factor = _PROFILE_WIDTH / gray.shape[1]
        small = cv2.resize(gray, (_PROFILE_WIDTH, max(1, int(gray.shape[0] * factor))), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (3, 3), 0)
    _threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows = (binary > 0).sum(axis=1).astype(np.float64)
    window = max(3, len(rows) // 100)
    kernel = np.ones(window) / window
    return np.convolve(rows, kernel, mode="same")
