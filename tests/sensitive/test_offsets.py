from __future__ import annotations

from app.sensitive.offsets import (
    EntitySpan,
    TokenAnnotation,
    canonicalize_egyptian_national_ids,
    polish_entity_span,
    reconstruct_entities,
    replace_matched_wrappers_with_spaces,
    trim_entity_boundaries,
)


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


def test_trim_entity_boundaries_strips_wrapping_punctuation() -> None:
    text = "تم تسجيل المريض تحت المرجع HSP-773902."
    start = text.index("HSP-773902")
    end = start + len("HSP-773902.")
    trimmed_start, trimmed_end = trim_entity_boundaries(
        text,
        start,
        end,
        "MEDICAL_RECORD_NUMBER",
    )
    assert text[trimmed_start:trimmed_end] == "HSP-773902"

    phone_text = "The account holder can be reached through +20 (100) 765-4321."
    phone = "+20 (100) 765-4321"
    phone_start = phone_text.index(phone)
    phone_end = phone_start + len(phone) + 1
    p_start, p_end = trim_entity_boundaries(
        phone_text,
        phone_start,
        phone_end,
        "PHONE",
    )
    assert phone_text[p_start:p_end] == phone

    wrapped = "Email=<MANAR.TEST@OUTLOOK.COM>"
    email = "MANAR.TEST@OUTLOOK.COM"
    e_start = wrapped.index("<")
    e_end = wrapped.index(">") + 1
    t_start, t_end = trim_entity_boundaries(wrapped, e_start, e_end, "EMAIL")
    assert wrapped[t_start:t_end] == email

    egyptian = "Contact: (01012345678)."
    egyptian_phone = "01012345678"
    wrapped_start = egyptian.index("(")
    wrapped_end = egyptian.index(")") + 1
    phone_start, phone_end = trim_entity_boundaries(
        egyptian,
        wrapped_start,
        wrapped_end,
        "PHONE",
    )
    assert egyptian[phone_start:phone_end] == egyptian_phone


def test_polish_keeps_plus_and_arabic_indic_digits() -> None:
    text = "Call +201012345678 now"
    span = EntitySpan("+201012345678", "PHONE", 5, 18, 0.9)
    polished = polish_entity_span(text, span)
    assert polished is not None
    assert polished.text == "+201012345678"

    arabic = "رقم ٠١٠١٢٣٤٥٦٧٨."
    digits = "٠١٠١٢٣٤٥٦٧٨"
    start = arabic.index(digits)
    polished_ar = polish_entity_span(
        arabic,
        EntitySpan(digits + ".", "PHONE", start, start + len(digits) + 1, 0.9),
    )
    assert polished_ar is not None
    assert polished_ar.text == digits
    assert arabic[polished_ar.start:polished_ar.end] == digits


def test_canonicalize_duplicate_national_id_labels() -> None:
    number = "29805151234567"
    text = f"رقم البطاقة القومية هو {number}."
    start = text.index(number)
    end = start + len(number)
    resolved = canonicalize_egyptian_national_ids(
        [
            EntitySpan(number, "NATIONAL_ID", start, end, 0.8),
            EntitySpan(number, "ID_CARD_NUMBER", start, end, 0.99),
            EntitySpan("Ahmed", "PERSON", 0, 5, 0.9),
        ]
    )
    national = [
        span for span in resolved if span.text == number
    ]
    assert len(national) == 1
    assert national[0].entity == "ID_CARD_NUMBER"
    assert any(span.entity == "PERSON" for span in resolved)


def test_replace_matched_wrappers_preserves_length_and_inner_offsets() -> None:
    text = "Patient ('John Smith') was seen."
    name = "John Smith"
    unwrapped = replace_matched_wrappers_with_spaces(text)

    assert len(unwrapped) == len(text)
    start = text.index(name)
    end = start + len(name)
    assert text[start:end] == name
    assert unwrapped[start:end] == name
    assert unwrapped[start - 1] == " "
    assert unwrapped[end] == " "

    apostrophe = "Call O'Brien today."
    assert replace_matched_wrappers_with_spaces(apostrophe) == apostrophe
