from __future__ import annotations

from app.sensitive.detector import SensitiveDataDetector
from app.sensitive.model import RawEntityPrediction
from app.sensitive.offsets import (
    EntitySpan,
    TokenAnnotation,
    integrate_rule_spans,
    merge_adjacent_spans,
    reconstruct_entities,
)
from app.sensitive.rules import (
    find_rule_entities,
    is_generic_clinical_address,
)


class FakeNERModel:
    def __init__(
        self,
        predictions: dict[str, list[RawEntityPrediction]],
    ) -> None:
        self._predictions = predictions
        self._model_name = "fake-ner-model"

    @property
    def model_name(self) -> str:
        return self._model_name

    def predict(self, text: str) -> list[RawEntityPrediction]:
        return list(self._predictions.get(text, []))


def _labels(text: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}

    for entity in find_rule_entities(text):
        assert text[entity.start:entity.end] == entity.text
        grouped.setdefault(entity.label, []).append(entity.text)

    return grouped


# ============================================================
# PHONE RULES
# ============================================================

def test_phone_formats_are_complete_spans() -> None:
    assert _labels("555-123-4567")["PHONE"] == ["555-123-4567"]
    assert _labels("(555) 123-4567")["PHONE"] == ["(555) 123-4567"]
    assert _labels("Phone: (555) 123-4567")["PHONE"] == ["(555) 123-4567"]
    assert _labels("+966 50 123 4567")["PHONE"] == ["+966 50 123 4567"]
    assert _labels("الجوال +966 50 123 4567")["PHONE"] == ["+966 50 123 4567"]
    assert _labels("01012345678")["PHONE"] == ["01012345678"]


def test_egyptian_phone_formats() -> None:
    assert _labels("Call me at 01012345678")["PHONE"] == ["01012345678"]
    assert _labels("Phone: +201012345678")["PHONE"] == ["+201012345678"]
    assert _labels("Phone: 00201012345678")["PHONE"] == ["00201012345678"]


def test_phone_rules_do_not_absorb_letters_or_short_numbers() -> None:
    text = "Call me at 555-123-4567 tomorrow."
    assert _labels(text)["PHONE"] == ["555-123-4567"]
    assert "PHONE" not in _labels("The follow-up is in 12 days.")
    assert "PHONE" not in _labels("The study was published in 2024.")
    assert "PHONE" not in _labels("ICD-10 code E11.9 was recorded in the clinic.")


def test_separate_phone_numbers_stay_separate() -> None:
    text = "اتصل بي على 01012345678 أو 01123456789."
    assert _labels(text)["PHONE"] == ["01012345678", "01123456789"]


# ============================================================
# CREDIT CARD RULES
# ============================================================

def test_credit_card_formats_prefer_the_full_number() -> None:
    dashed = "The credit card number is 4532-1488-0343-6467."
    spaced = "4532 1488 0343 6467"
    compact = "The credit card number is 4532148803436467."

    assert _labels(dashed)["CREDIT_CARD"] == ["4532-1488-0343-6467"]
    assert _labels(spaced)["CREDIT_CARD"] == ["4532 1488 0343 6467"]
    assert _labels(compact)["CREDIT_CARD"] == ["4532148803436467"]

    assert "PHONE" not in _labels(spaced)
    assert "CREDIT_CARD" not in _labels(
        "Reference 1234567890123456 without card context."
    )


def test_luhn_is_not_required_for_contextual_cards() -> None:
    entities = find_rule_entities(
        "The credit card number is 4532148803436467."
    )
    cards = [
        entity
        for entity in entities
        if entity.label == "CREDIT_CARD"
    ]

    assert len(cards) == 1
    assert cards[0].text == "4532148803436467"
    assert cards[0].score < 0.99


# ============================================================
# MEDICAL RECORD NUMBER RULES
# ============================================================

def test_medical_identifiers_cover_the_full_value() -> None:
    assert "4455-6677-8" in _labels(
        "MRN 4455-6677-8"
    )["MEDICAL_RECORD_NUMBER"][0]

    assert _labels(
        "Patient ID: 99887766"
    )["MEDICAL_RECORD_NUMBER"] == ["99887766"]

    assert _labels(
        "رقم الملف الطبي 445566778"
    )["MEDICAL_RECORD_NUMBER"] == ["445566778"]

    assert _labels(
        "رقم المريض: 99887766"
    )["MEDICAL_RECORD_NUMBER"] == ["99887766"]

    assert _labels(
        "The patient's medical record number is MRN123456."
    )["MEDICAL_RECORD_NUMBER"] == ["MRN123456"]


def test_national_id_requires_national_id_context() -> None:
    assert _labels(
        "رقم البطاقة القومية 12345678901234"
    )["ID_CARD_NUMBER"] == ["12345678901234"]

    assert "ID_CARD_NUMBER" not in _labels(
        "Reference number 12345678901234"
    )


def test_egyptian_card_word_classifies_14_digit_as_national_id() -> None:
    text = "بطاقتي 30507162101546"
    number = "30507162101546"
    assert _labels(text)["ID_CARD_NUMBER"] == [number]
    assert "CREDIT_CARD" not in _labels(text)

    start = text.index(number)
    detector = SensitiveDataDetector(
        model=FakeNERModel(
            {
                text: [
                    RawEntityPrediction(
                        number,
                        "CREDIT_CARD",
                        start,
                        start + len(number),
                        0.94,
                    )
                ]
            }
        )
    )
    result = detector.detect(text)
    matches = [
        entity
        for entity in result.entities
        if entity.text == number
    ]
    assert len(matches) == 1
    assert matches[0].entity == "ID_CARD_NUMBER"
    assert text[matches[0].start:matches[0].end] == number
    assert not any(entity.entity == "CREDIT_CARD" for entity in result.entities)


def test_explicit_credit_card_context_keeps_16_digit_card() -> None:
    text = "The credit card number is 4111111111111111."
    assert _labels(text)["CREDIT_CARD"] == ["4111111111111111"]
    assert "ID_CARD_NUMBER" not in _labels(text)


def test_ambiguous_14_digit_number_is_not_national_id() -> None:
    text = "The tracking number is 30507162101546."
    labels = _labels(text)
    assert "ID_CARD_NUMBER" not in labels
    assert "CREDIT_CARD" not in labels


def test_invoice_total_is_not_a_credit_card() -> None:
    text = "The invoice total is 1234567890123456 Egyptian pounds."
    labels = _labels(text)
    assert "CREDIT_CARD" not in labels
    assert "ID_CARD_NUMBER" not in labels

    start = text.index("1234567890123456")
    detector = SensitiveDataDetector(
        model=FakeNERModel(
            {
                text: [
                    RawEntityPrediction(
                        "1234567890123456",
                        "CREDIT_CARD",
                        start,
                        start + 16,
                        0.91,
                    )
                ]
            }
        )
    )
    assert detector.detect(text).entities == ()


# ============================================================
# ADDRESS RULES
# ============================================================

def test_addresses_include_street_and_city_but_not_clinic_words() -> None:
    assert _labels(
        "221B Baker Street, London"
    )["ADDRESS"] == ["221B Baker Street, London"]

    assert _labels(
        "I live at 25 Nile Street, Giza, Egypt."
    )["ADDRESS"] == ["25 Nile Street, Giza, Egypt"]

    assert _labels(
        "عنواني 15 شارع التحرير، القاهرة."
    )["ADDRESS"] == ["15 شارع التحرير، القاهرة"]

    assert "ADDRESS" not in _labels(
        "The clinic department reviewed the chart."
    )

    assert is_generic_clinical_address("العيادة")
    assert not is_generic_clinical_address(
        "15 شارع التحرير، القاهرة"
    )


def test_clinical_words_are_not_rule_detected_as_addresses() -> None:
    for text in (
        "The patient needs to drink more water.",
        "Artificial intelligence is changing healthcare.",
        "The patient takes Metformin daily.",
        "The clinic department reviewed the chart.",
        "المريض حضر إلى العيادة اليوم.",
    ):
        assert "ADDRESS" not in _labels(text)


# ============================================================
# OFFSET / NER PERSON TESTS
# These do not test PERSON detection by rules.py.
# ============================================================

def test_reconstruct_keeps_separate_person_begin_tags() -> None:
    text = "Omar Sara"

    spans = reconstruct_entities(
        text,
        [
            TokenAnnotation(1, 0, 4, "B-PERSON", 0.9, False),
            TokenAnnotation(2, 5, 9, "B-PERSON", 0.8, False),
        ],
    )

    assert [span.text for span in spans] == ["Omar", "Sara"]


def test_merge_keeps_separate_person_spans() -> None:
    people = merge_adjacent_spans(
        "Ahmed called Ahmed",
        [
            EntitySpan("Ahmed", "PERSON", 0, 5, 0.9),
            EntitySpan("Ahmed", "PERSON", 13, 18, 0.9),
        ],
    )

    assert [span.text for span in people] == ["Ahmed", "Ahmed"]


# ============================================================
# SPAN MERGING
# ============================================================

def test_merge_joins_address_and_phone_fragments_only() -> None:
    address = merge_adjacent_spans(
        "25 Nile Street, Giza",
        [
            EntitySpan("25", "ADDRESS", 0, 2, 0.9),
            EntitySpan("Nile Street", "ADDRESS", 3, 14, 0.9),
            EntitySpan("Giza", "ADDRESS", 16, 20, 0.9),
        ],
    )

    assert [span.text for span in address] == [
        "25 Nile Street, Giza"
    ]

    phones = merge_adjacent_spans(
        "01012345678 أو 01123456789",
        [
            EntitySpan("01012345678", "PHONE", 0, 11, 0.9),
            EntitySpan("01123456789", "PHONE", 15, 26, 0.9),
        ],
    )

    assert [span.text for span in phones] == [
        "01012345678",
        "01123456789",
    ]

    broken_phone = merge_adjacent_spans(
        "(555) 123-4567",
        [
            EntitySpan("(555)", "PHONE", 0, 5, 0.6),
            EntitySpan("3-4567", "PHONE", 8, 14, 0.6),
        ],
    )

    assert [span.text for span in broken_phone] == [
        "(555) 123-4567"
    ]


# ============================================================
# RULE + MODEL SPAN INTEGRATION
# ============================================================

def test_rule_replaces_partial_model_phone_span() -> None:
    text = "Phone: (555) 123-4567"

    model_spans = [
        EntitySpan("(555)", "PHONE", 7, 12, 0.61),
        EntitySpan("3-4567", "PHONE", 15, 21, 0.58),
    ]

    rules = [
        EntitySpan(
            rule.text,
            rule.label,
            rule.start,
            rule.end,
            rule.score,
        )
        for rule in find_rule_entities(text)
    ]

    combined = integrate_rule_spans(model_spans, rules)

    phones = [
        span.text
        for span in combined
        if span.entity == "PHONE"
    ]

    assert phones == ["(555) 123-4567"]


def test_exact_model_span_keeps_model_score() -> None:
    exact = integrate_rule_spans(
        [EntitySpan("Omar", "PERSON", 8, 12, 0.98)],
        [EntitySpan("Omar", "PERSON", 8, 12, 0.93)],
    )

    assert exact[0].confidence == 0.98


# ============================================================
# DETECTOR INTEGRATION
# ============================================================

def test_detector_drops_generic_clinic_and_completes_patient_id() -> None:
    clinic = "المريض حضر إلى العيادة اليوم."
    clinic_start = clinic.index("العيادة")

    patient = "Patient ID: 99887766"

    detector = SensitiveDataDetector(
        model=FakeNERModel(
            {
                clinic: [
                    RawEntityPrediction(
                        "العيادة",
                        "ADDRESS",
                        clinic_start,
                        clinic_start + len("العيادة"),
                        0.99,
                    )
                ],
                patient: [
                    RawEntityPrediction(
                        "99887766",
                        "ID_CARD_NUMBER",
                        12,
                        20,
                        0.75,
                    )
                ],
            }
        )
    )

    assert detector.detect(clinic).entities == ()

    detected = detector.detect(patient)

    assert [
        (entity.text, entity.entity)
        for entity in detected.entities
    ] == [("99887766", "MEDICAL_RECORD_NUMBER")]

    assert (
        patient[
            detected.entities[0].start:detected.entities[0].end
        ]
        == "99887766"
    )


def test_phone_rule_does_not_drop_unrelated_person_span() -> None:
    text = "Contact Omar Ahmed, phone 01012345678."
    detector = SensitiveDataDetector(
        model=FakeNERModel(
            {
                text: [
                    RawEntityPrediction(
                        "Omar Ahmed",
                        "PERSON",
                        8,
                        18,
                        0.97,
                    )
                ]
            }
        )
    )
    result = detector.detect(text)
    extracted = {(entity.text, entity.entity) for entity in result.entities}
    assert ("Omar Ahmed", "PERSON") in extracted
    assert ("01012345678", "PHONE") in extracted


def test_duplicate_national_id_labels_are_canonicalized() -> None:
    cases = (
        "رقم البطاقة القومية هو 29805151234567.",
        "بيانات مقدم الطلب: الرقم القومي 30112251234567.",
        "National ID: 30112251234567.",
        "رقم البطاقة القومية: 30112251234567.",
    )
    for text in cases:
        number = "29805151234567" if "29805151234567" in text else "30112251234567"
        start = text.index(number)
        detector = SensitiveDataDetector(
            model=FakeNERModel(
                {
                    text: [
                        RawEntityPrediction(
                            number,
                            "NATIONAL_ID",
                            start,
                            start + len(number),
                            0.88,
                        )
                    ]
                }
            )
        )
        result = detector.detect(text)
        matches = [
            entity
            for entity in result.entities
            if entity.text == number
        ]
        assert len(matches) == 1, text
        assert matches[0].entity == "ID_CARD_NUMBER"
        assert text[matches[0].start:matches[0].end] == number


def test_punctuation_and_complete_phone_rule_spans() -> None:
    phone_text = "The account holder can be reached through +20 (100) 765-4321."
    assert _labels(phone_text)["PHONE"] == ["+20 (100) 765-4321"]

    hospital = "تم تسجيل المريض تحت المرجع HSP-773902."
    assert _labels(hospital)["MEDICAL_RECORD_NUMBER"] == ["HSP-773902"]

    mixed = "Patient data: الاسم كريم مصطفى, reference HSP-773902."
    mixed_start = mixed.index("HSP-773902")
    detector = SensitiveDataDetector(
        model=FakeNERModel(
            {
                mixed: [
                    RawEntityPrediction(
                        "كريم مصطفى",
                        "PERSON",
                        mixed.index("كريم مصطفى"),
                        mixed.index("كريم مصطفى") + len("كريم مصطفى"),
                        0.96,
                    ),
                    RawEntityPrediction(
                        "HSP-773902.",
                        "MEDICAL_RECORD_NUMBER",
                        mixed_start,
                        mixed_start + len("HSP-773902."),
                        0.7,
                    ),
                ]
            }
        )
    )
    mixed_result = detector.detect(mixed)
    extracted = {(entity.text, entity.entity) for entity in mixed_result.entities}
    assert ("كريم مصطفى", "PERSON") in extracted
    assert ("HSP-773902", "MEDICAL_RECORD_NUMBER") in extracted
    assert ("HSP-773902.", "MEDICAL_RECORD_NUMBER") not in extracted

    wrapped_email = "Email=<MANAR.TEST@OUTLOOK.COM>"
    assert _labels(wrapped_email)["EMAIL"] == ["MANAR.TEST@OUTLOOK.COM"]

    wrapped_phone = "Phone number (personal): [01098765432]."
    assert _labels(wrapped_phone)["PHONE"] == ["01098765432"]

    partial_phone = "Reach me on +20 (100) 765-4321."
    phone_start = partial_phone.index("(100)")
    detector = SensitiveDataDetector(
        model=FakeNERModel(
            {
                partial_phone: [
                    RawEntityPrediction(
                        "(100) 765-4321",
                        "PHONE",
                        phone_start,
                        phone_start + len("(100) 765-4321"),
                        0.6,
                    )
                ]
            }
        )
    )
    phones = [
        entity.text
        for entity in detector.detect(partial_phone).entities
        if entity.entity == "PHONE"
    ]
    assert phones == ["+20 (100) 765-4321"]


def test_context_sensitive_identifier_classification() -> None:
    assert _labels(
        "The patient's medical record number is MRN123456."
    )["MEDICAL_RECORD_NUMBER"] == ["MRN123456"]
    assert _labels(
        "The hospital file reference is HSP-773902."
    )["MEDICAL_RECORD_NUMBER"] == ["HSP-773902"]
    assert _labels(
        "الرقم الوظيفي للموظف EMP-10245."
    )["EMPLOYEE_ID"] == ["EMP-10245"]

    visitor = "The registration code assigned to the visitor is VIS-80421."
    visitor_labels = _labels(visitor)
    assert "MEDICAL_RECORD_NUMBER" not in visitor_labels
    assert "EMPLOYEE_ID" not in visitor_labels

    sample = "The sample code is ABC-12345 and the batch number is B-9081."
    sample_labels = _labels(sample)
    assert "MEDICAL_RECORD_NUMBER" not in sample_labels
    assert "EMPLOYEE_ID" not in sample_labels

    visitor_start = visitor.index("VIS-80421")
    detector = SensitiveDataDetector(
        model=FakeNERModel(
            {
                visitor: [
                    RawEntityPrediction(
                        "VIS-80421",
                        "MEDICAL_RECORD_NUMBER",
                        visitor_start,
                        visitor_start + len("VIS-80421"),
                        0.93,
                    )
                ]
            }
        )
    )
    assert detector.detect(visitor).entities == ()

    combined = "Ahmed, الرقم الوظيفي للموظف EMP-10245."
    emp_start = combined.index("EMP-10245")
    detector = SensitiveDataDetector(
        model=FakeNERModel(
            {
                combined: [
                    RawEntityPrediction("Ahmed", "PERSON", 0, 5, 0.97),
                    RawEntityPrediction(
                        "EMP-10245",
                        "MEDICAL_RECORD_NUMBER",
                        emp_start,
                        emp_start + len("EMP-10245"),
                        0.8,
                    ),
                ]
            }
        )
    )
    extracted = {
        (entity.text, entity.entity)
        for entity in detector.detect(combined).entities
    }
    assert ("Ahmed", "PERSON") in extracted
    assert ("EMP-10245", "EMPLOYEE_ID") in extracted
    assert ("EMP-10245", "MEDICAL_RECORD_NUMBER") not in extracted


def test_low_confidence_prediction_is_not_dropped_at_default_threshold() -> None:
    text = "Zed arrived today."
    annotations = [
        TokenAnnotation(1, 0, 3, "B-PERSON", 0.55, False)
    ]

    class TokenModel:
        model_name = "token-model"

        def run_token_classification(
            self,
            value: str,
        ) -> list[TokenAnnotation]:
            assert value == text
            return list(annotations)

    result = SensitiveDataDetector(
        model=TokenModel()
    ).detect(text)

    assert [
        (entity.text, entity.entity)
        for entity in result.entities
    ] == [("Zed", "PERSON")]

    assert result.entities[0].confidence == 0.55
