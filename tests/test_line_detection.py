from PIL import Image, ImageDraw

from ocr.line_detection import LineDetectConfig, detect_lines, save_line_crops
from ocr.preprocessing import validate_image


def _lines_image(path, size=(220, 140), bands=((18, 28), (70, 80), (110, 120))):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y0, y1 in bands:
        draw.rectangle((12, y0, size[0] - 12, y1), fill="black")
    image.save(path)
    return path


def test_blank_image_does_not_invent_boxes(tmp_path):
    path = tmp_path / "blank.png"
    image = Image.new("RGB", (80, 60), (250, 250, 250))
    image.putpixel((0, 0), (249, 249, 249))
    image.save(path)
    loaded, _info = validate_image(path)
    result = detect_lines(loaded, LineDetectConfig(min_area=40))
    assert result.lines == []
    assert any("No text lines" in note or "no candidate" in note for note in result.warnings)


def test_one_and_multiple_horizontal_lines(tmp_path):
    one = tmp_path / "one.png"
    _lines_image(one, bands=((40, 52),))
    loaded, _info = validate_image(one)
    single = detect_lines(loaded)
    assert len(single.lines) >= 1
    box = single.lines[0].bbox
    assert box[2] > box[0] and box[3] > box[1]

    many = tmp_path / "many.png"
    _lines_image(many)
    loaded, _info = validate_image(many)
    result = detect_lines(loaded)
    assert len(result.lines) >= 2
    orders = [line.reading_order for line in result.lines]
    assert orders == sorted(orders)
    tops = [line.bbox[1] for line in result.lines]
    assert tops == sorted(tops)


def test_boundary_grayscale_rgba_and_save(tmp_path):
    path = tmp_path / "edge.png"
    image = Image.new("RGBA", (160, 80), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 8, 159, 20), fill=(0, 0, 0, 255))
    draw.rectangle((2, 50, 40, 62), fill=(0, 0, 0, 255))
    image.save(path)
    loaded, info = validate_image(path)
    assert loaded.mode == "RGB"
    result = detect_lines(loaded)
    assert result.lines
    for line in result.lines:
        x1, y1, x2, y2 = line.bbox
        assert 0 <= x1 < x2 <= loaded.width
        assert 0 <= y1 < y2 <= loaded.height
    saved = save_line_crops(result.lines, tmp_path / "lines")
    assert (tmp_path / "lines" / f"{saved[0].line_id}.png").is_file()
    assert saved[0].crop_path.startswith("lines/")


def test_small_image_warns(tmp_path):
    image = Image.new("RGB", (6, 6), "white")
    image.putpixel((1, 1), (0, 0, 0))
    result = detect_lines(image)
    assert result.lines == []
    assert any("too small" in note.lower() for note in result.warnings)


# ------------------------------------------------------------------
# Regression cases found in code review: each failed before the detector was improved.
# ------------------------------------------------------------------
import pytest  # noqa: E402
from PIL import ImageFont  # noqa: E402


def _font(size):
    return ImageFont.load_default(size=size)


def _text_page(rows, size=(1600, 2000), start=140, pitch=120, ruled=False, font_size=34):
    image = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(image)
    if ruled:
        for y in range(150, size[1] - 100, 60):
            draw.line([(0, y), (size[0], y)], fill=(150, 170, 220), width=2)
    for index in range(rows):
        draw.text((120, start + index * pitch), f"Patient note line {index + 1} Paracetamol 500mg", fill=(10, 10, 10), font=_font(font_size))
    return image


def test_control_plain_page_finds_every_line():
    assert len(detect_lines(_text_page(10)).lines) == 10


def test_ruled_notebook_paper_does_not_turn_rules_into_lines():
    result = detect_lines(_text_page(10, ruled=True))
    assert len(result.lines) == 10  # used to be 30: every ruling line became a "text line"
    assert any("ruling lines" in note for note in result.warnings)


def test_ruling_removal_can_be_turned_off():
    result = detect_lines(_text_page(10, ruled=True), LineDetectConfig(remove_rules=False))
    assert len(result.lines) > 10


def test_thick_text_bars_are_not_mistaken_for_ruling_lines(tmp_path):
    path = tmp_path / "bars.png"
    image = Image.new("RGB", (1600, 400), "white")
    draw = ImageDraw.Draw(image)
    for top in (60, 160, 260):
        draw.rectangle((100, top, 1500, top + 30), fill="black")  # 30 px tall, wider than a rule
    image.save(path)
    loaded, _ = validate_image(path)
    assert len(detect_lines(loaded).lines) == 3


def test_table_cells_are_read_instead_of_dropping_the_whole_table():
    image = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(image)
    for x in (100, 600, 1100, 1500):
        draw.line([(x, 100), (x, 900)], fill=0, width=3)
    for y in range(100, 901, 160):
        draw.line([(100, y), (1500, y)], fill=0, width=3)
    for row in range(5):
        for col in range(3):
            draw.text((130 + col * 500, 150 + row * 160), f"cell {row}{col}", fill=0, font=_font(34))
    result = detect_lines(image)
    assert len(result.lines) == 15  # used to be 0: the table was one "tall region" and was dropped


def test_lines_that_touch_are_split_instead_of_dropped():
    image = Image.new("RGB", (1600, 1100), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    for index in range(9):  # 52 px pitch for a 56 px font: ascenders and descenders overlap
        draw.text((100, 40 + index * 52), f"gypsy Typography jqpy {index + 1} ggg yyy", fill=(10, 10, 10), font=_font(56))
    assert len(detect_lines(image, LineDetectConfig(split_tall=False)).lines) == 0
    result = detect_lines(image)
    assert len(result.lines) == 9
    assert any("split" in note for note in result.warnings)


def _two_fields_on_one_row(left_taller: bool):
    image = Image.new("RGB", (1600, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((1000, 100), "Tg Ygp right field" if not left_taller else "right field", fill=0, font=_font(34))
    draw.text((60, 100), "Tg Ygp left" if left_taller else "left", fill=0, font=_font(34))
    return image


@pytest.mark.parametrize("left_taller", [True, False])
def test_same_row_order_no_longer_depends_on_a_pixel_of_ascender_height(left_taller):
    result = detect_lines(_two_fields_on_one_row(left_taller))
    assert [line.row_id for line in result.lines] == [1, 1]
    assert [line.bbox[0] < 800 for line in result.lines] == [True, False]  # left field first, always


def test_right_to_left_direction_reads_the_right_field_first():
    result = detect_lines(_two_fields_on_one_row(True), LineDetectConfig(direction="rtl"))
    assert [line.bbox[0] > 800 for line in result.lines] == [True, False]


def test_rows_are_ordered_top_to_bottom_whatever_the_direction():
    page = _text_page(4)
    for direction in ("ltr", "rtl"):
        tops = [line.bbox[1] for line in detect_lines(page, LineDetectConfig(direction=direction)).lines]
        assert tops == sorted(tops)


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError):
        LineDetectConfig(direction="up")
