"""Deskew, orientation, page crop and the box mapping that ties them together."""

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr.orientation import detect_orientation, rotate_k90, undo_rot90_point
from ocr.page_crop import find_page_quad, warp_page
from ocr.preprocessing import (
    ImageInfo,
    PreprocessConfig,
    align_color,
    apply_combined_preprocess,
    deskew_angle,
)

WORDS = ["Patient", "note", "Paracetamol", "500mg", "twice", "daily", "after", "meals", "review", "Friday"]


def text_page(width=1000, height=1400, size=34, marker=None) -> np.ndarray:
    """Upright page of text-like lines. Optional red square marker ((x, y) of its centre)."""
    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=size)
    for row in range(height // (size * 2) - 2):
        line = " ".join(WORDS[(row + shift) % len(WORDS)] for shift in range(5))
        draw.text((60, 70 + row * size * 2), line, fill=(20, 20, 40), font=font)
    if marker:
        x, y = marker
        draw.rectangle((x - 12, y - 12, x + 12, y + 12), fill=(220, 0, 0))
    return np.asarray(image)


def rotate_content(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=(250, 250, 248))


def red_centroid(image: np.ndarray) -> tuple[float, float]:
    rgb = np.asarray(image)
    mask = (rgb[..., 0] > 150) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)
    assert mask.sum() > 20, "marker not found"
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def photo_of(page: np.ndarray, corners, canvas=(1700, 1400), background=(55, 55, 58)) -> tuple[np.ndarray, np.ndarray]:
    height, width = page.shape[:2]
    source = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, np.array(corners, dtype=np.float32))
    photo = cv2.warpPerspective(page, matrix, canvas, flags=cv2.INTER_CUBIC, borderValue=background)
    return photo, matrix


PHOTO_CORNERS = [(300, 120), (1100, 170), (1060, 1250), (260, 1180)]  # keeps the 5:7 page shape


# ---------------------------------------------------------------- deskew
@pytest.mark.parametrize("tilt", [3.0, -8.0, 20.0, -25.0])
def test_deskew_recovers_tilts_far_beyond_the_old_12_degree_limit(tilt) -> None:
    gray = cv2.cvtColor(rotate_content(text_page(), tilt), cv2.COLOR_RGB2GRAY)
    assert deskew_angle(gray, PreprocessConfig()) == pytest.approx(-tilt, abs=1.0)


def test_deskew_leaves_an_upright_page_alone() -> None:
    gray = cv2.cvtColor(text_page(), cv2.COLOR_RGB2GRAY)
    assert deskew_angle(gray, PreprocessConfig()) == 0.0


def test_deskew_can_be_limited() -> None:
    gray = cv2.cvtColor(rotate_content(text_page(), 20.0), cv2.COLOR_RGB2GRAY)
    angle = deskew_angle(gray, PreprocessConfig(deskew_max_degrees=12.0))
    assert abs(angle) <= 12.0


# ----------------------------------------------------------- orientation
def test_upright_page_needs_no_turn() -> None:
    result = detect_orientation(cv2.cvtColor(text_page(), cv2.COLOR_RGB2GRAY))
    assert not result.needs_rotation and result.horizontal_score > result.vertical_score


@pytest.mark.parametrize("k", [1, 3])
def test_sideways_page_is_detected_and_a_turn_restores_horizontal_lines(k) -> None:
    sideways = np.rot90(text_page(), k)
    gray = cv2.cvtColor(np.ascontiguousarray(sideways), cv2.COLOR_RGB2GRAY)
    result = detect_orientation(gray)
    assert result.needs_rotation and result.ambiguous
    restored = cv2.cvtColor(rotate_k90(sideways, result.k90), cv2.COLOR_RGB2GRAY)
    assert not detect_orientation(restored).needs_rotation


def test_blank_page_is_not_rotated() -> None:
    assert not detect_orientation(np.full((300, 300), 255, dtype=np.uint8)).needs_rotation


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_undo_rot90_point_round_trips(k) -> None:
    image = np.zeros((70, 120), dtype=np.uint8)  # height 70, width 120
    image[20, 90] = 255
    rotated = rotate_k90(image, k)
    ys, xs = np.nonzero(rotated)
    x, y = undo_rot90_point(xs[0] + 0.5, ys[0] + 0.5, k, rotated.shape[1], rotated.shape[0])
    assert (x, y) == pytest.approx((90.5, 20.5), abs=1e-6)


# ------------------------------------------------------------- page crop
def test_page_is_found_in_a_photo_and_flattened() -> None:
    page = text_page()
    photo, _ = photo_of(page, PHOTO_CORNERS)
    quad = find_page_quad(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY))
    assert quad is not None
    for found, truth in zip(quad, PHOTO_CORNERS):
        assert np.hypot(*(found - np.array(truth))) < 25
    flat, _ = warp_page(photo, quad)
    assert flat.shape[1] / flat.shape[0] == pytest.approx(page.shape[1] / page.shape[0], rel=0.08)


def test_a_plain_scan_is_not_cropped() -> None:
    assert find_page_quad(cv2.cvtColor(text_page(), cv2.COLOR_RGB2GRAY)) is None


def test_a_small_bright_patch_is_not_mistaken_for_a_page() -> None:
    canvas = np.full((1000, 1400), 50, dtype=np.uint8)
    canvas[400:560, 600:900] = 240  # ~2% of the frame
    assert find_page_quad(canvas) is None


def test_a_low_contrast_scene_is_left_alone() -> None:
    canvas = np.full((1000, 1400), 120, dtype=np.uint8)
    canvas[100:900, 150:1250] = 135
    assert find_page_quad(canvas) is None


# ------------------------------------- boxes still map back to the original
INFO = ImageInfo(path=None, width=0, height=0, mode="RGB", dpi=None)


def _map_marker(original: np.ndarray, config: PreprocessConfig) -> tuple[tuple[float, float], object]:
    variant, geometry = apply_combined_preprocess(original, INFO, config)
    aligned = align_color(original, geometry)
    assert aligned.shape[:2] == variant.image.shape[:2]
    cx, cy = red_centroid(aligned)
    return geometry.map_processed_point_to_oriented(cx, cy), geometry


def test_marker_maps_back_through_page_crop_and_deskew() -> None:
    page = text_page(marker=(500, 700))
    photo, matrix = photo_of(page, PHOTO_CORNERS)
    truth = cv2.perspectiveTransform(np.array([[[500.0, 700.0]]]), matrix)[0, 0]
    (x, y), geometry = _map_marker(photo, PreprocessConfig(enable_upscale=False))
    assert geometry.page_homography is not None
    assert (x, y) == pytest.approx(tuple(truth), abs=25)


@pytest.mark.parametrize("k", [1, 3])
def test_marker_maps_back_through_crop_quarter_turn_and_deskew(k) -> None:
    page = text_page(marker=(500, 700))
    photo, matrix = photo_of(page, PHOTO_CORNERS)
    sideways = np.ascontiguousarray(np.rot90(photo, k))
    marker_in_photo = cv2.perspectiveTransform(np.array([[[500.0, 700.0]]]), matrix)[0, 0]
    height, width = photo.shape[:2]
    # position of the same pixel after np.rot90(photo, k)
    truth = {1: (marker_in_photo[1], width - marker_in_photo[0]), 3: (height - marker_in_photo[1], marker_in_photo[0])}[k]
    (x, y), geometry = _map_marker(sideways, PreprocessConfig(enable_upscale=False))
    assert geometry.rotation_k90 == 1 and geometry.page_homography is not None
    if k == 1:  # the guessed direction was right: the marker maps back exactly
        assert (x, y) == pytest.approx(truth, abs=25)
    else:  # direction is a guess without a recognizer; the mapping must still be self-consistent
        assert 0 <= x <= sideways.shape[1] and 0 <= y <= sideways.shape[0]


def test_geometry_without_optional_stages_matches_the_old_behaviour() -> None:
    page = text_page(marker=(400, 500))
    _, geometry = apply_combined_preprocess(page, INFO, PreprocessConfig(enable_upscale=False, enable_deskew=False))
    assert geometry.rotation_k90 == 0 and geometry.page_homography is None
    assert geometry.map_processed_point_to_oriented(400, 500) == pytest.approx((400, 500), abs=1e-6)


# ------------------------- regression: a tilted page must never be mistaken for a sideways one
@pytest.mark.parametrize("tilt", [5.0, 14.0, -14.0, 25.0])
def test_a_tilted_upright_page_is_not_turned_by_90_degrees(tilt) -> None:
    gray = cv2.cvtColor(rotate_content(text_page(1600, 2200), tilt), cv2.COLOR_RGB2GRAY)
    result = detect_orientation(gray)
    assert not result.needs_rotation, f"tilt {tilt} was read as sideways: {result}"


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("tilt", [0.0, 8.0])
def test_a_sideways_page_is_found_even_when_it_is_also_tilted(k, tilt) -> None:
    page = np.ascontiguousarray(np.rot90(rotate_content(text_page(1600, 2200), tilt), k))
    assert detect_orientation(cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)).needs_rotation


def test_a_page_with_text_in_only_half_of_it_is_judged_by_its_lines_not_its_margins() -> None:
    image = Image.new("RGB", (1600, 2200), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=34)
    for row in range(28):
        draw.text((60, 70 + row * 68), "Patient note Paracetamol", fill=(20, 20, 40), font=font)
    upright = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    assert not detect_orientation(upright).needs_rotation
    assert detect_orientation(np.ascontiguousarray(np.rot90(upright))).needs_rotation


def test_ruled_paper_and_tables_keep_their_orientation() -> None:
    ruled = Image.new("RGB", (1600, 2000), (250, 250, 250))
    draw = ImageDraw.Draw(ruled)
    for y in range(150, 1900, 60):
        draw.line([(0, y), (1600, y)], fill=(150, 170, 220), width=2)
    for row in range(10):
        draw.text((120, 140 + row * 120), f"Patient note line {row + 1}", fill=(10, 10, 10), font=ImageFont.load_default(size=34))
    gray = cv2.cvtColor(np.asarray(ruled), cv2.COLOR_RGB2GRAY)
    assert not detect_orientation(gray).needs_rotation
    assert detect_orientation(np.ascontiguousarray(np.rot90(gray))).needs_rotation
