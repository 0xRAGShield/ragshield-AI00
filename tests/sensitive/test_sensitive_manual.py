
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "Security_layer"

sys.path.insert(0, str(APP_ROOT))

from app.sensitive import SensitiveDataDetector


detector = SensitiveDataDetector()

sentences = [
    # English: Personal Information
    "Sarah Ahmed works as a software engineer.",
    "Please contact me at john.smith@yahoo.com.",
    "My phone number is +201012345678.",
    "You can reach me on 01234567890 or email me at sara@example.com.",

    # English: Sensitive Identifiers
    "My national ID number is 29805151234567.",
    "The patient's medical record number is MRN123456.",
    "My passport number is A12345678.",
    "The credit card number is 4111111111111111.",
    "My employee ID is EMP-10245.",

    # English: Address and Dates
    "I live at 25 Nile Street, Giza, Egypt.",
    "The patient lives in 10 El Tahrir Street, Cairo.",
    "My date of birth is 15/03/2004.",

    # English: Mixed Information
    "Ahmed Hassan can be reached at 01012345678.",
    "Patient Sara Ali, phone 01123456789, email sara.ali@gmail.com.",
    "John Smith's passport number is A12345678 and his email is john@example.com.",

    # Arabic: Personal Information
    "أنا اسمي منار عبدالباقي.",
    "اسمي أحمد محمد ورقم تليفوني 01012345678.",
    "ممكن تتواصلوا مع سارة على sara@gmail.com.",
    "رقم موبايلي 01123456789.",
    "البريد الإلكتروني الخاص بي هو manar@example.com.",

    # Arabic: Sensitive Identifiers
    "رقم البطاقة القومية هو 29805151234567.",
    "رقم جواز السفر A12345678.",
    "رقم الملف الطبي للمريض MRN123456.",
    "رقم بطاقة الائتمان 4111111111111111.",
    "الرقم الوظيفي للموظف EMP-10245.",

    # Arabic: Address and Medical Context
    "عنواني 15 شارع التحرير، القاهرة.",
    "المريضة سارة أحمد تسكن في الجيزة.",
    "تاريخ ميلادي 15/03/2004.",
    "اسم المريض محمد علي ورقم ملفه الطبي MRN123456.",

    # Negative Cases: No Sensitive Information
    "Artificial intelligence is changing healthcare.",
    "The weather is nice today.",
    "The patient needs to drink more water.",
    "Machine learning models require good training data.",
    "No personal information is included in this message.",

    # Edge Cases
    "Email: test@example.com",
    "Phone: 01012345678",
    "Name: Ahmed",
    "My email is test@example.com and my email is test@example.com.",
    "اتصل بي على 01012345678 أو 01123456789.",
    "إيميلي هو manar@gmail.com ورقم تليفوني 01012345678.",
]


def extract_entities(result):
    """
    Supports detector results represented as key-value pairs
    or objects exposing an entities attribute.
    """
    if hasattr(result, "entities"):
        return result.entities or ()

    try:
        data = dict(result)
        return data.get("entities", ())
    except (TypeError, ValueError):
        return ()


for sentence in sentences:
    print("\n" + "=" * 80)
    print(f"Sentence: {sentence}")
    print("-" * 80)

    try:
        result = detector.detect(sentence)
        entities = extract_entities(result)

        if not entities:
            print("No sensitive data detected.")
            continue

        print("Sensitive data detected:")

        for entity in entities:
            print(
                f"Text: {entity.text!r} | "
                f"Type: {entity.entity} | "
                f"Start: {entity.start} | "
                f"End: {entity.end} | "
                f"Confidence: {entity.confidence:.3f}"
            )

    except Exception as exc:
        print(f"Detection failed: {type(exc).__name__}: {exc}")


from app.sensitive.offsets import reconstruct_entities

debug_cases = [
    "The patient's medical record number is MRN123456.",
    "رقم الملف الطبي للمريض MRN123456.",
    "My full name is John Smith.",
    "رقم البطاقة القومية هو 29805151234567.",
]

for text in debug_cases:
    print("\n" + "=" * 70)
    print(f"DEBUG: {text}")

    annotations = detector._model.run_token_classification(text)

    for ann in annotations:
        if ann.is_special:
            continue

        print(
            f"{text[ann.start:ann.end]!r:20} "
            f"{ann.label:30} "
            f"[{ann.start}:{ann.end}] "
            f"confidence={ann.confidence:.3f}"
        )

    spans = reconstruct_entities(
        text,
        annotations,
        confidence_threshold=detector._config.confidence_threshold,
    )

    print("\nReconstructed entities:")
    for span in spans:
        print(
            f"{span.text!r} -> {span.entity} "
            f"[{span.start}:{span.end}] "
            f"confidence={span.confidence:.3f}"
        )
