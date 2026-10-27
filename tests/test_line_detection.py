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
