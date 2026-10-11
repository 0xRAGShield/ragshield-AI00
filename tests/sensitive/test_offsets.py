from __future__ import annotations

from app.sensitive.offsets import TokenAnnotation, reconstruct_entities


def test_reconstruct_bio_subwords_and_invalid_i_tag() -> None:
    text = "Ahmed Mohamed then Sara"
    annotations = [
        TokenAnnotation(0, 0, 0, "O", 1.0, True),
        TokenAnnotation(1, 0, 5, "B-PERSON", 0.9, False),
        TokenAnnotation(2, 6, 13, "I-PERSON", 0.8, False),
        TokenAnnotation(3, 14, 18, "O", 0.99, False),
        TokenAnnotation(4, 19, 23, "I-PERSON", 0.7, False),
        TokenAnnotation(5, 23, 23, "O", 1.0, True),
    ]
    spans = reconstruct_entities(text, annotations)
    assert [span.text for span in spans] == ["Ahmed Mohamed", "Sara"]
    assert all(text[span.start:span.end] == span.text for span in spans)
    assert all(span.entity == "PERSON" for span in spans)


def test_reconstruct_consecutive_same_type_starts_new_entity() -> None:
    text = "Omar Sara"
    annotations = [
        TokenAnnotation(1, 0, 4, "B-PERSON", 0.9, False),
        TokenAnnotation(2, 5, 9, "B-PERSON", 0.8, False),
    ]
    spans = reconstruct_entities(text, annotations)
    assert [span.text for span in spans] == ["Omar", "Sara"]


def test_reconstruct_skips_special_tokens_and_o() -> None:
    text = "hi"
    annotations = [
        TokenAnnotation(0, 0, 0, "B-PERSON", 0.99, True),
        TokenAnnotation(1, 0, 2, "O", 0.99, False),
    ]
    assert reconstruct_entities(text, annotations) == []


def test_reconstruct_arabic_digits_and_punctuation() -> None:
    text = "هاتف: 01012345678."
    phone = "01012345678"
    start = text.index(phone)
    annotations = [
        TokenAnnotation(1, 0, start, "O", 0.9, False),
        TokenAnnotation(2, start, start + 5, "B-PHONE", 0.95, False),
        TokenAnnotation(3, start + 5, start + len(phone), "I-PHONE", 0.94, False),
        TokenAnnotation(4, start + len(phone), len(text), "O", 0.9, False),
    ]
    spans = reconstruct_entities(text, annotations)
    assert len(spans) == 1
    assert spans[0].text == phone
    assert spans[0].entity == "PHONE"
    assert text[spans[0].start:spans[0].end] == phone


def test_reconstruct_empty_text() -> None:
    assert reconstruct_entities("", []) == []
