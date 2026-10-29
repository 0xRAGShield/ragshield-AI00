"""Find the sheet of paper in a photo and flatten it.

Deliberately conservative. A wrong crop throws text away, which is worse than not
cropping, so a quadrilateral is accepted only when it clearly looks like a sheet
of paper on a darker background. Anything doubtful returns None and the page is
used as it is.
"""

from __future__ import annotations

import cv2
import numpy as np

_ANALYSIS_SIDE = 800
MIN_AREA_FRACTION = 0.25
MAX_AREA_FRACTION = 0.96
MIN_CONTRAST = 25.0
MAX_OUTPUT_SIDE = 6000


def find_page_quad(gray: np.ndarray) -> np.ndarray | None:
    """Return the four corners (TL, TR, BR, BL) in full-resolution pixels, or None."""
    height, width = gray.shape[:2]
    longest = max(height, width)
    factor = min(1.0, _ANALYSIS_SIDE / longest)
    small = cv2.resize(gray, (max(1, int(width * factor)), max(1, int(height * factor))), interpolation=cv2.INTER_AREA) if factor < 1 else gray
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _threshold, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(small.shape[0] * small.shape[1])
    contour = max(contours, key=cv2.contourArea)
    fraction = cv2.contourArea(contour) / image_area
    if not MIN_AREA_FRACTION <= fraction <= MAX_AREA_FRACTION:
        return None

    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    quad = None
    for epsilon in (0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, epsilon * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float64)
            break
    if quad is None or not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return None

    # Paper must be clearly brighter than what surrounds it.
    mask = np.zeros(small.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    inside = float(small[mask > 0].mean())
    outside_pixels = small[mask == 0]
    if outside_pixels.size == 0 or inside - float(outside_pixels.mean()) < MIN_CONTRAST:
        return None

    ordered = order_corners(quad) / factor
    return ordered


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()  # y - x
    return np.array(
        [points[np.argmin(sums)], points[np.argmin(diffs)], points[np.argmax(sums)], points[np.argmax(diffs)]],
        dtype=np.float64,
    )


def warp_page(image: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the quad. Returns the new image and the 3x3 matrix (source -> output)."""
    top_left, top_right, bottom_right, bottom_left = quad
    out_width = int(round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))))
    out_height = int(round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))))
    scale = min(1.0, MAX_OUTPUT_SIDE / max(out_width, out_height, 1))
    out_width, out_height = max(1, int(out_width * scale)), max(1, int(out_height * scale))
    destination = np.array(
        [[0, 0], [out_width, 0], [out_width, out_height], [0, out_height]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    border = (255, 255, 255) if image.ndim == 3 else 255
    warped = cv2.warpPerspective(
        image, matrix, (out_width, out_height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )
    return warped, matrix.astype(np.float64)
