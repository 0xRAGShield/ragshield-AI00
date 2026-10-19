
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "Security_layer"
sys.path.insert(0, str(APP_ROOT))

from app.sensitive import SensitiveDataDetector
from app.sensitive.labels import DEFAULT_PRETRAINED_MODEL, NUM_LABELS


# Every expected entity uses the same format:
# (exact expected text, expected entity type)
CASES = [
    # 56. SINGLE-QUOTED PERSON
{
    "text": "The registered patient is 'Daniel Patrick Lewis'.",
    "expected": {("Daniel Patrick Lewis", "PERSON")},
},

# 57. DOUBLE-QUOTED PERSON
{
    "text": 'The emergency contact is "شهيناز".',
    "expected": {("شهيناز", "PERSON")},
},

# 58. CURLY-QUOTED PERSON
{
    "text": "Patient name: “Michael Scott”.",
    "expected": {("Michael Scott", "PERSON")},
},

# 59. NESTED WRAPPERS
{
    "text": 'Patient name: ("Daniel Lewis").',
    "expected": {("Daniel Lewis", "PERSON")},
},

# 60. SQUARE-BRACKETED PERSON
{
    "text": "The patient is [Sophia Miller].",
    "expected": {("Sophia Miller", "PERSON")},
},

# 61. ARABIC NAME IN BRACKETS
{
    "text": "اسم المريض [عمر خالد]",
    "expected": {("عمر خالد", "PERSON")},
},

# 62. QUOTED DOCTOR NAME
{
    "text": 'Reviewed by "Dr. Michael Adams" yesterday.',
    "expected": {("Dr. Michael Adams", "PERSON")},
},

# 63. QUOTED EMAIL
{
    "text": 'Send the file to "<daniel.lewis@example.org>".',
    "expected": {("daniel.lewis@example.org", "EMAIL")},
},

# 64. PARENTHESIZED EGYPTIAN PHONE
{
    "text": "Patient contact: (01098765432).",
    "expected": {("01098765432", "PHONE")},
},

# 65. QUOTED INTERNATIONAL PHONE
{
    "text": 'Emergency number: "+20 100 234 5678".',
    "expected": {("+20 100 234 5678", "PHONE")},
},

# 66. BRACKETED NATIONAL ID
{
    "text": "National ID: [28706151234568].",
    "expected": {("28706151234568", "NATIONAL_ID")},
},

# 67. QUOTED CREDIT CARD
{
    "text": 'Payment card: "4111 1111 1111 1111".',
    "expected": {("4111 1111 1111 1111", "CREDIT_CARD")},
},

# 68. QUOTED PASSPORT
{
    "text": 'Passport reference: "A76543219".',
    "expected": {("A76543219", "PASSPORT_NUMBER")},
},

# 69. QUOTED MEDICAL RECORD NUMBER
{
    "text": 'Patient MRN: (MRN-625184).',
    "expected": {("MRN-625184", "MEDICAL_RECORD_NUMBER")},
},

# 70. QUOTED USERNAME
{
    "text": 'Portal username: "daniel_lewis83".',
    "expected": {("daniel_lewis83", "USERNAME")},
},

# 71. QUOTED PASSWORD
{
    "text": 'Temporary password: "SafeClinic#739".',
    "expected": {("SafeClinic#739", "PASSWORD")},
},

# 72. QUOTED IP ADDRESS
{
    "text": 'Workstation IP: "192.168.10.24".',
    "expected": {("192.168.10.24", "IP_ADDRESS")},
},

# 73. APOSTROPHE INSIDE A NAME
{
    "text": "The patient is منار.",
    "expected": {("منار", "PERSON")},
},

# 74. APOSTROPHE INSIDE AN ORDINARY WORD
{
    "text": "The patient's appointment is confirmed.",
    "expected": set(),
},

# 75. MULTIPLE OCCURRENCES OF THE SAME NAME
{
    "text": "Muhab called Muhab about Muhab's results.",
    "expected": {("Muhab", "PERSON")},
},

# 76. TWO DIFFERENT PEOPLE
{
    "text": "Nurse Emily Carter contacted patient Daniel Lewis.",
    "expected": {
        ("Emily Carter", "PERSON"),
        ("Daniel Lewis", "PERSON"),
    },
},

# 77. SAME NAME WITH DIFFERENT WRAPPERS
{
    "text": 'Patient "Daniel Lewis" met [Daniel Lewis].',
    "expected": {("Daniel Lewis", "PERSON")},
},

# 78. PERSON AND EMAIL WITH SAME NAME
{
    "text": "Rachel Green (rachel.green@example.org) is the patient.",
    "expected": {
        ("Rachel Green", "PERSON"),
        ("rachel.green@example.org", "EMAIL"),
    },
},

# 79. PERSON, PHONE, AND NATIONAL ID
{
    "text": "DR. Sarah, phone 01098765432, national ID 28706151234568.",
    "expected": {
        ("DR.Sarah", "PERSON"),
        ("01098765432", "PHONE"),
        ("28706151234568", "NATIONAL_ID"),
    },
},

# 80. TWO EMAILS
{
    "text": "Contact daniel@example.org or rachel@example.net.",
    "expected": {
        ("daniel@example.org", "EMAIL"),
        ("rachel@example.net", "EMAIL"),
    },
},

# 81. PHONE WITH AREA-CODE PARENTHESES
{
    "text": "Office phone: (212) 555-0184.",
    "expected": {("(212) 555-0184", "PHONE")},
},

# 82. EMAIL WITH PLUS ADDRESSING
{
    "text": "Email: daniel.lewis+records@example.org.",
    "expected": {("daniel.lewis+records@example.org", "EMAIL")},
},

# 83. EMAIL WITH SUBDOMAIN
{
    "text": "Send results to daniel@records.hospital.example.org.",
    "expected": {("daniel@records.hospital.example.org", "EMAIL")},
},

# 84. ARABIC-INDIC DIGITS IN PHONE
{
    "text": "رقم الموبايل ٠١٠٩٨٧٦٥٤٣٢",
    "expected": {("٠١٠٩٨٧٦٥٤٣٢", "PHONE")},
},

# 85. MIXED NUMERAL SYSTEMS IN PHONE
{
    "text": "رقم التواصل 010٩٨٧٦٥٤٣٢",
    "expected": {("010٩٨٧٦٥٤٣٢", "PHONE")},
},

# 86. ARABIC NAME WITH DIACRITICS
{
    "text": "اسم المريض مُحَمَّد علي",
    "expected": {("مُحَمَّد علي", "PERSON")},
},

# 87. LONG ARABIC NAME
{
    "text": "اسم المريضة فاطمة أحمد عبد الرحمن مصطفى",
    "expected": {("فاطمة أحمد عبد الرحمن مصطفى", "PERSON")},
},

# 88. FRANCO NAME WITH MULTIPLE DIGITS
{
    "text": "El patient esmo Mohamed 3abdallah.",
    "expected": {("Mohamed 3abdallah", "PERSON")},
},

# 89. FRANCO WITH EMAIL AND PHONE
{
    "text": "Esm el patient Karim Hassan, email karim.hassan@example.org, phone 01098765432.",
    "expected": {
        ("Karim Hassan", "PERSON"),
        ("karim.hassan@example.org", "EMAIL"),
        ("01098765432", "PHONE"),
    },
},

# 90. FRANCO NEGATIVE CASE
{
    "text": "El doctor katab el report wel patient ra7 el bet.",
    "expected": set(),
},

# 91. ORDINARY WORD IN QUOTES
{
    "text": 'The nurse said "the patient is stable".',
    "expected": set(),
},

# 92. ORDINARY PHRASE IN BRACKETS
{
    "text": "The treatment plan is [rest and hydration].",
    "expected": set(),
},

# 93. ORDINARY NUMBER IS NOT AUTOMATICALLY PII
{
    "text": "The patient completed 12 exercises in 3 sessions.",
    "expected": set(),
},

# 94. ORDINARY DATE WITHOUT PERSONAL CONTEXT
{
    "text": "The conference starts on 14 March 2027.",
    "expected": set(),
},

# 95. MEDICAL TERM IS NOT A PERSON
{
    "text": "The patient was diagnosed with Type 2 Diabetes Mellitus.",
    "expected": set(),
},

# 96. EMAIL NEXT TO PUNCTUATION
{
    "text": "(daniel.lewis@example.org), please confirm receipt.",
    "expected": {("daniel.lewis@example.org", "EMAIL")},
},

# 97. PERSON NEXT TO PUNCTUATION
{
    "text": "Patient: [john smith], admitted today.",
    "expected": {("john smith", "PERSON")},
},

# 98. TWO WRAPPED ENTITIES IN ONE SENTENCE
{
    "text": 'Patient "Rachel Green", phone (01098765432).',
    "expected": {
        ("Rachel Green", "PERSON"),
        ("01098765432", "PHONE"),
    },
},

# 99. ARABIC NAME AND QUOTED NATIONAL ID
{
    "text": 'اسم المريض "عمر خالد" ورقمه القومي "28706151234568".',
    "expected": {
        ("عمر خالد", "PERSON"),
        ("28706151234568", "NATIONAL_ID"),
    },
},

# 100. ALL ENTITIES MUST KEEP ORIGINAL OFFSETS
{
    "text": 'Patient "Rachel Green", email (rachel.green@example.org), phone [01098765432].',
    "expected": {
        ("Rachel Green", "PERSON"),
        ("rachel.green@example.org", "EMAIL"),
        ("01098765432", "PHONE"),
    },
},
{
    "text": "اسم المريض \"يوسف محمود\"",
    "expected": {("يوسف محمود", "PERSON")},
},

{
    "text": "المريضة (سلمى إبراهيم) حضرت للمتابعة.",
    "expected": {("سلمى إبراهيم", "PERSON")},
},

{
    "text": "اسم المريض [كريم عبد الرحمن].",
    "expected": {("كريم عبد الرحمن", "PERSON")},
},

{
    "text": "بيانات المريضة: «نورهان أحمد».",
    "expected": {("نورهان أحمد", "PERSON")},
},

{
    "text": "المريض اسمه محمود أحمد عبد الله.",
    "expected": {("محمود أحمد عبد الله", "PERSON")},
},

{
    "text": "اسم صاحبة الملف فاطمة محمد عبد الرحمن علي.",
    "expected": {("فاطمة محمد عبد الرحمن علي", "PERSON")},
},

{
    "text": "الدكتور أحمد سامح راجع نتائج التحاليل.",
    "expected": {("أحمد سامح", "PERSON")},
},

{
    "text": "الدكتورة منى عادل كتبت الروشتة.",
    "expected": {("منى عادل", "PERSON")},
},

{
    "text": "تم تسجيل بيانات المريض واسمه خالد مصطفى.",
    "expected": {("خالد مصطفى", "PERSON")},
},

{
    "text": "رقم تليفون أحمد سمير هو 01098765432.",
    "expected": {
        ("أحمد سمير", "PERSON"),
        ("01098765432", "PHONE"),
    },
},

{
    "text": "كلم أحمد حسن على 01012345678 أو منى عادل على 01123456789.",
    "expected": {
        ("أحمد حسن", "PERSON"),
        ("01012345678", "PHONE"),
        ("منى عادل", "PERSON"),
        ("01123456789", "PHONE"),
    },
},

{
    "text": "ابعت التقرير على nour.mohamed@example.com لو سمحت.",
    "expected": {("nour.mohamed@example.com", "EMAIL")},
},

{
    "text": "الإيميل المسجل هو (sara.ali@example.org).",
    "expected": {("sara.ali@example.org", "EMAIL")},
},

{
    "text": "الرقم القومي بتاع المريض هو 29807151234567.",
    "expected": {("29807151234567", "NATIONAL_ID")},
},

{
    "text": "الرقم القومي: (29807151234567).",
    "expected": {("29807151234567", "NATIONAL_ID")},
},

{
    "text": "الرقم القومي للمريضة ٢٩٨٠٧١٥١٢٣٤٥٦٧.",
    "expected": {("٢٩٨٠٧١٥١٢٣٤٥٦٧", "NATIONAL_ID")},
},

{
    "text": "رقم الموبايل ٠١٠١٢٣٤٥٦٧٨.",
    "expected": {("٠١٠١٢٣٤٥٦٧٨", "PHONE")},
},

{
    "text": "رقم التواصل 010 1234 5678.",
    "expected": {("010 1234 5678", "PHONE")},
},

{
    "text": "رقم الملف الطبي للمريض MRN-853621.",
    "expected": {("MRN-853621", "MEDICAL_RECORD_NUMBER")},
},

{
    "text": "ملف المريض (CLN-842615) موجود في النظام.",
    "expected": {("CLN-842615", "MEDICAL_RECORD_NUMBER")},
},

{
    "text": "تاريخ ميلاد المريضة هو 17/08/2000.",
    "expected": {("17/08/2000", "PRIVATE_DATE")},
},

{
    "text": "تاريخ الميلاد ١٧/٠٨/٢٠٠٠.",
    "expected": {("١٧/٠٨/٢٠٠٠", "PRIVATE_DATE")},
},

{
    "text": "عنوان المريض ٢٥ شارع الهرم، الجيزة.",
    "expected": {("٢٥ شارع الهرم، الجيزة", "ADDRESS")},
},

{
    "text": "عنوان السكن شقة ٧، عمارة ١٢، مدينة نصر، القاهرة.",
    "expected": {("شقة ٧، عمارة ١٢، مدينة نصر، القاهرة", "ADDRESS")},
},

{
    "text": "باسورد الحساب المؤقت هو SecureMed#628.",
    "expected": {("SecureMed#628", "PASSWORD")},
},

{
    "text": "اليوزر نيم الخاص بالمريض هو ahmed_samir92.",
    "expected": {("ahmed_samir92", "USERNAME")},
},

{
    "text": "بيانات المريضة منى خالد، الإيميل mona.khaled@example.com، ورقم الملف MRN-853621.",
    "expected": {
        ("منى خالد", "PERSON"),
        ("mona.khaled@example.com", "EMAIL"),
        ("MRN-853621", "MEDICAL_RECORD_NUMBER"),
    },
},

{
    "text": "المريض عمر سامح، رقمه القومي 29807151234567 وتاريخ ميلاده 17/08/2000.",
    "expected": {
        ("عمر سامح", "PERSON"),
        ("29807151234567", "NATIONAL_ID"),
        ("17/08/2000", "PRIVATE_DATE"),
    },
},

{
    "text": "المريض اتحسن بعد العلاج وحالته مستقرة.",
    "expected": set(),
},

{
    "text": "المريض راح قسم الطوارئ وعمل أشعة وتحاليل.",
    "expected": set(),
},
   
]


def _entity_key(entity) -> tuple[str, str]:
    """Normalize a detected entity to (text, entity_type)."""
    return (entity.text, entity.entity)


def main() -> int:
    detector = SensitiveDataDetector()
    model = detector._model

    print(f"Model path: {detector.model_name}")
    print(f"Configured default: {DEFAULT_PRETRAINED_MODEL}")
    print(f"Device: {getattr(model, 'device', 'unknown')}")
    print(f"Canonical label count: {NUM_LABELS}")
    print(
        "Loaded model labels: "
        f"{len(getattr(model, 'id2label', {})) or 'n/a'}"
    )
    print(f"Total test cases: {len(CASES)}")
    print()

    failures = 0
    total_missed = 0
    total_extra = 0
    offset_errors = 0

    for index, case in enumerate(CASES, start=1):
        text = case["text"]
        expected = set(case["expected"])

        print("=" * 80)
        print(f"Test: {index}/{len(CASES)}")
        print(f"Sentence: {text}")

        result = detector.detect(text)
        detected = {
            _entity_key(entity)
            for entity in result.entities
        }

        # Validate that every returned span matches the source text.
        case_offset_errors = []

        for entity in result.entities:
            slice_ok = text[entity.start:entity.end] == entity.text

            print(
                f"  text={entity.text!r} "
                f"type={entity.entity} "
                f"confidence={entity.confidence:.3f} "
                f"start={entity.start} "
                f"end={entity.end} "
                f"slice_ok={slice_ok}"
            )

            if not slice_ok:
                case_offset_errors.append(
                    (entity.text, entity.entity, entity.start, entity.end)
                )

        # Exact entity comparison: text AND type must match.
        missed = expected - detected
        extra = detected - expected

        if missed or extra or case_offset_errors:
            failures += 1

            if missed:
                total_missed += len(missed)
                print(f"  MISSED: {sorted(missed)}")

            if extra:
                total_extra += len(extra)
                print(f"  FALSE POSITIVES / WRONG SPANS: {sorted(extra)}")

            if case_offset_errors:
                offset_errors += len(case_offset_errors)
                print(f"  OFFSET ERRORS: {case_offset_errors}")

        elif not detected:
            print("  PASS: No sensitive data detected.")

        else:
            print("  PASS: All expected entities detected exactly.")

    total_cases = len(CASES)
    passed = total_cases - failures

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total cases: {total_cases}")
    print(f"Passed cases: {passed}")
    print(f"Cases with mismatches: {failures}")
    print(f"Missed expected entities: {total_missed}")
    print(f"Extra / wrong-span entities: {total_extra}")
    print(f"Invalid offsets: {offset_errors}")

    if total_cases:
        print(f"Case pass rate: {passed / total_cases * 100:.2f}%")

    print(
        "\nNote: Case pass rate is not entity-level Precision, Recall, or F1."
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
