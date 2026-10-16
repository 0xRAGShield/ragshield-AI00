
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
    # 1. PERSON
    {
        "text": "The applicant is Olivia Bennett.",
        "expected": {("Olivia Bennett", "PERSON")},
    },
    {
        "text": "المريض اسمه أحمد ياسين.",
        "expected": {("أحمد ياسين", "PERSON")},
    },
    {
        "text": "Patient name: Isabella Rose Carter.",
        "expected": {("Isabella Rose Carter", "PERSON")},
    },
    {
        "text": "اسم المريضة ليلى عبد العزيز",
        "expected": {("ليلى عبد العزيز", "PERSON")},
    },
    
    # 2. EMAIL
    {
        "text": "Send the report to olivia.bennett@samplemail.org.",
        "expected": {("olivia.bennett@samplemail.org", "EMAIL")},
    },
    {
        "text": "البريد الإلكتروني هو layla.ahmed+clinic@samplemail.org",
        "expected": {("layla.ahmed+clinic@samplemail.org", "EMAIL")},
    },
    {
        "text": "Email: SUPPORT_TEAM42@EXAMPLE.NET",
        "expected": {("SUPPORT_TEAM42@EXAMPLE.NET", "EMAIL")},
    },

    # 3. PHONE
    {
        "text": "Reach the applicant at 01234567890.",
        "expected": {("01234567890", "PHONE")},
    },
    {
        "text": "رقم التواصل 01156789012",
        "expected": {("01156789012", "PHONE")},
    },
    {
        "text": "International number: +20 122 345 6789",
        "expected": {("+20 122 345 6789", "PHONE")},
    },
    {
        "text": "Call either 01045678912 or 01567890123.",
        "expected": {
            ("01045678912", "PHONE"),
            ("01567890123", "PHONE"),
        },
    },

    # 4. ADDRESS
    {
        "text": "The applicant lives at 42 Lotus Avenue, Alexandria, Egypt.",
        "expected": {("42 Lotus Avenue, Alexandria, Egypt", "ADDRESS")},
    },
    {
        "text": "عنوان المريض ٢٢ شارع النصر، الإسكندرية",
        "expected": {("٢٢ شارع النصر، الإسكندرية", "ADDRESS")},
    },
    {
        "text": "Delivery address: Flat 6, Building 14, Heliopolis, Cairo.",
        "expected": {("Flat 6, Building 14, Heliopolis, Cairo", "ADDRESS")},
    },

    # 5. ID CARD NUMBER
    {
        "text": "Civil ID number: 29901011234567.",
        "expected": {("29901011234567", "NATIONAL_ID")},
    },
    {
        "text": "رقم الهوية الشخصية 30101011234567",
        "expected": {("30101011234567", "NATIONAL_ID")},
    },

    # 6. PASSPORT NUMBER
    {
        "text": "Passport reference: B98765432.",
        "expected": {("B98765432", "PASSPORT_NUMBER")},
    },
    {
        "text": "رقم جواز السفر C76543210",
        "expected": {("C76543210", "PASSPORT_NUMBER")},
    },

    # 7. MEDICAL RECORD NUMBER
    {
        "text": "Medical file ID: MED-628491.",
        "expected": {("MED-628491", "MEDICAL_RECORD_NUMBER")},
    },
    {
        "text": "رقم ملف الحالة الطبية CLN-735204",
        "expected": {("CLN-735204", "MEDICAL_RECORD_NUMBER")},
    },

    # 8. CREDIT CARD
    {
        "text": "Payment card number: 5555555555554444.",
        "expected": {("5555555555554444", "CREDIT_CARD")},
    },
    {
        "text": "بيانات بطاقة الدفع 4000000000000002",
        "expected": {("4000000000000002", "CREDIT_CARD")},
    },

    # 9. EMPLOYEE ID
    {
        "text": "Staff employee code: STAFF-62841.",
        "expected": {("STAFF-62841", "EMPLOYEE_ID")},
    },
    {
        "text": "رقم الموظف بالمؤسسة HR-52917",
        "expected": {("HR-52917", "EMPLOYEE_ID")},
    },

    # 10. PRIVATE DATE
    {
        "text": "The patient's date of birth is 21/07/1998.",
        "expected": {("21/07/1998", "PRIVATE_DATE")},
    },
    {
        "text": "تاريخ ميلاد صاحب الملف 09-11-2001",
        "expected": {("09-11-2001", "PRIVATE_DATE")},
    },

    # 11. USERNAME
    {
        "text": "Portal username: olivia_bennett82",
        "expected": {("olivia_bennett82", "USERNAME")},
    },
    {
        "text": "اسم المستخدم في النظام layla.clinic27",
        "expected": {("layla.clinic27", "USERNAME")},
    },

    # 12. PASSWORD
    {
        "text": "Temporary password: BlueRiver!739",
        "expected": {("BlueRiver!739", "PASSWORD")},
    },
    {
        "text": "كلمة المرور المؤقتة هي Clinic#4821",
        "expected": {("Clinic#4821", "PASSWORD")},
    },

    # 13. IP ADDRESS
    {
        "text": "The workstation IP is 10.24.18.7.",
        "expected": {("10.24.18.7", "IP_ADDRESS")},
    },
    {
        "text": "Server address: 172.20.4.19",
        "expected": {("172.20.4.19", "IP_ADDRESS")},
    },

    # 14. SOCIAL SECURITY NUMBER
    {
        "text": "Social security number: 219-48-6731.",
        "expected": {("219-48-6731", "SOCIAL_SECURITY_NUMBER")},
    },

    # 15. DRIVER LICENSE
    {
        "text": "Driver license number: D123-987-654-321.",
        "expected": {("D123-987-654-321", "DRIVER_LICENSE")},
    },

    # 16. NATIONAL ID / SPOKEN NATIONAL ID
    {
        "text": "My national identification number is 28706151234568.",
        "expected": {("28706151234568", "NATIONAL_ID")},
    },
    {
        "text": (
            "My spoken national ID is two eight seven zero six one "
            "five one two three four five six eight."
        ),
        "expected": {
            (
                "two eight seven zero six one five one two three "
                "four five six eight",
                "SPOKEN_NATIONAL_ID",
            )
        },
    },

    # 17. ANOTHER MEDICAL RECORD NUMBER
    {
        "text": "Patient record number: PAT-830625.",
        "expected": {("PAT-830625", "MEDICAL_RECORD_NUMBER")},
    },

    # 18. MULTIPLE PII TYPES
    {
        "text": (
            "Patient: Ethan Cole, email ethan.cole@samplemail.org, "
            "phone 01234567890."
        ),
        "expected": {
            ("Ethan Cole", "PERSON"),
            ("ethan.cole@samplemail.org", "EMAIL"),
            ("01234567890", "PHONE"),
        },
    },
    {
        "text": (
            "بيانات المريضة مريم عادل، هاتف 01156789012، "
            "ملف CLN-735204"
        ),
        "expected": {
            ("مريم عادل", "PERSON"),
            ("01156789012", "PHONE"),
            ("CLN-735204", "MEDICAL_RECORD_NUMBER"),
        },
    },

    # 19. FRANCO ARABIC / MIXED LANGUAGE
    {
        "text": "Momken teb3ato el report 3ala layla.test@samplemail.org",
        "expected": {("layla.test@samplemail.org", "EMAIL")},
    },
    {
        "text": "El raqam bta3 Olivia howa 01234567890",
        "expected": {("01234567890", "PHONE"),
                     ("Olivia", "PERSON")
                     },
    },
    {
        "text": "call Ahmed Yassin 3ala 01156789012",
        "expected": {
            ("Ahmed Yassin", "PERSON"),
            ("01156789012", "PHONE"),
        },
    },

    # 20. NEGATIVE CASES
    {
        "text": "The outpatient department opens at eight.",
        "expected": set(),
    },
    {
        "text": "Artificial intelligence supports medical research.",
        "expected": set(),
    },
    {
        "text": "The system processed 250 records successfully.",
        "expected": set(),
    },
    {
        "text": "The hospital has a cardiology and neurology department.",
        "expected": set(),
    },
    {
        "text": "The appointment starts at 10:30 AM.",
        "expected": set(),
    },

    # 21. ADDITIONAL ARABIC PERSON=
    {
        "text": "المريضة اسمها سارة أحمد عبد الله",
        "expected": {("سارة أحمد عبد الله", "PERSON")},
    },

    # 22. ADDITIONAL ENGLISH PERSON=
    {
        "text": "The appointment is for Emily Rose Carter.",
        "expected": {("Emily Rose Carter", "PERSON")},
    },

    # 23. FRANCO ARABIC PERSON=
    {
        "text": "El patient esmo Karim Mohamed w 3ando appointment.",
        "expected": {("Karim Mohamed", "PERSON")},
    },

    # 24. ADDITIONAL EGYPTIAN PHONE
    {
        "text": "للتواصل مع المريض اتصل على 01012345678",
        "expected": {("01012345678", "PHONE")},
    },

    # 25. INTERNATIONAL PHONE
    {
        "text": "Contact the patient at +1 202-555-0147.",
        "expected": {("+1 202-555-0147", "PHONE")},
    },

    # 26. ADDITIONAL EMAIL
    {
        "text": "Send the report to nour.ahmed@example.com",
        "expected": {("nour.ahmed@example.com", "EMAIL")},
    },

    # 27. ARABIC ADDRESS
    {
        "text": "عنوان المريض 15 شارع التحرير، الدقي، الجيزة",
        "expected": {("15 شارع التحرير، الدقي، الجيزة", "ADDRESS")},
    },

    # 28. ENGLISH ADDRESS
    {
        "text": "The patient lives at 24 Oxford Street, London.",
        "expected": {("24 Oxford Street, London", "ADDRESS")},
    },

    # 29. NATIONAL ID
    {
        "text": "الرقم القومي للمريض 29807151234567",
        "expected": {("29807151234567", "NATIONAL_ID")},
    },

    # 30. ADDITIONAL PASSPORT
    {
        "text": "Passport number: A12345678",
        "expected": {("A12345678", "PASSPORT_NUMBER")},
    },

    # 31. ADDITIONAL MEDICAL RECORD NUMBER
    {
        "text": "Patient MRN: MRN-784521",
        "expected": {("MRN-784521", "MEDICAL_RECORD_NUMBER")},
    },

    # 32. ADDITIONAL CREDIT CARD
    {
        "text": "The card number is 4111 1111 1111 1111",
        "expected": {("4111 1111 1111 1111", "CREDIT_CARD")},
    },

    # 33. ADDITIONAL EMPLOYEE ID
    {
        "text": "Employee ID: EMP-20481",
        "expected": {("EMP-20481", "EMPLOYEE_ID")},
    },

    # 34. DATE OF BIRTH
    {
        "text": "Date of birth: 14 March 2001",
        "expected": {("14 March 2001", "PRIVATE_DATE")},
    },

    # 35. ADDITIONAL USERNAME
    {
        "text": "The patient's username is manar_ahmed22",
        "expected": {("manar_ahmed22", "USERNAME")},
    },

    # 36. ADDITIONAL PASSWORD
    {
        "text": "Temporary password: MedSecure#927",
        "expected": {("MedSecure#927", "PASSWORD")},
    },

    # 37. ADDITIONAL IP ADDRESS
    {
        "text": "The request came from IP address 192.168.1.25",
        "expected": {("192.168.1.25", "IP_ADDRESS")},
    },

    # 38. THREE PII TYPES IN ONE SENTENCE
    {
        "text": (
            "Patient Salma Hassan, phone 01123456789, "
            "email salma.test@example.com"
        ),
        "expected": {
            ("Salma Hassan", "PERSON"),
            ("01123456789", "PHONE"),
            ("salma.test@example.com", "EMAIL"),
        },
    },

    # 39. NEGATIVE: ORDINARY MEDICAL SENTENCE
    {
        "text": "The patient has a headache and needs more rest.",
        "expected": set(),
    },

    # 40. NEGATIVE: FRANCO / MIXED LANGUAGE
    {
        "text": (
            "Momken nراجع نتيجة التحليل bokra ba3d el doctor visit?"
        ),
        "expected": set(),
    },
        # 41. Arabic name in an English sentence
    {
        "text": "Patient name: أحمد محمود حسن.",
        "expected": {("أحمد محمود حسن", "PERSON")},
    },

    # 42. Arabic female name in an English sentence
    {
        "text": "The appointment is for مريم خالد إبراهيم.",
        "expected": {("مريم خالد إبراهيم", "PERSON")},
    },

    # 43. Egyptian Arabic: name and phone
    {
        "text": "اسم المريض محمد علي ورقم تليفونه 01012345678",
        "expected": {
            ("محمد علي", "PERSON"),
            ("01012345678", "PHONE"),
        },
    },

    # 44. Egyptian Arabic: multiple PII entities
    {
        "text": (
            "بيانات المريضة نور أحمد، الإيميل nour.ahmed@example.com، "
            "ورقم الموبايل 01123456789"
        ),
        "expected": {
            ("نور أحمد", "PERSON"),
            ("nour.ahmed@example.com", "EMAIL"),
            ("01123456789", "PHONE"),
        },
    },

    # 45. English sentence with Arabic name and email
    {
        "text": "Please send the report to سارة محمد at sara.mohamed@example.com.",
        "expected": {
            ("سارة محمد", "PERSON"),
            ("sara.mohamed@example.com", "EMAIL"),
        },
    },

    # 46. Arabic address with a mixed-language sentence
    {
        "text": "Patient address: ١٥ شارع التحرير، الدقي، الجيزة",
        "expected": {
            ("١٥ شارع التحرير، الدقي، الجيزة", "ADDRESS"),
        },
    },

    # 47. Egyptian national ID
    {
        "text": "الرقم القومي الخاص بأحمد محمود هو 29807151234567",
        "expected": {
            ("أحمد محمود", "PERSON"),
            ("29807151234567", "NATIONAL_ID"),
        },
    },

    # 48. Arabic name with medical record number
    {
        "text": "ملف المريض يوسف إبراهيم رقمه الطبي MRN-625184",
        "expected": {
            ("يوسف إبراهيم", "PERSON"),
            ("MRN-625184", "MEDICAL_RECORD_NUMBER"),
        },
    },

    # 49. Franco Arabic with Arabic name
    {
        "text": "El patient esmo أحمد سامي w ra2mo 01012345678",
        "expected": {
            ("أحمد سامي", "PERSON"),
            ("01012345678", "PHONE"),
        },
    },

    # 50. Franco Arabic with Arabic name and email
    {
        "text": "Momken teb3ato el report le منى إبراهيم 3ala mona.ibrahim@example.com",
        "expected": {
            ("منى إبراهيم", "PERSON"),
            ("mona.ibrahim@example.com", "EMAIL"),
        },
    },

    # 51. Arabic name written in English context
    {
        "text": "The patient is called كريم مصطفى and his file is CLN-735204.",
        "expected": {
            ("كريم مصطفى", "PERSON"),
            ("CLN-735204", "MEDICAL_RECORD_NUMBER"),
        },
    },

    # 52. Multiple Arabic names and phone numbers
    {
        "text": (
            "للتواصل مع أحمد حسن 01012345678 أو منى علي 01123456789"
        ),
        "expected": {
            ("أحمد حسن", "PERSON"),
            ("01012345678", "PHONE"),
            ("منى علي", "PERSON"),
            ("01123456789", "PHONE"),
        },
    },

    # 53. Arabic date of birth
    {
        "text": "تاريخ ميلاد المريضة سارة محمود هو 14/03/2001",
        "expected": {
            ("سارة محمود", "PERSON"),
            ("14/03/2001", "PRIVATE_DATE"),
        },
    },

    # 54. Arabic username in an English sentence
    {
        "text": "اسم المستخدم الخاص بـ خالد سمير هو khaled_samir27",
        "expected": {
            ("خالد سمير", "PERSON"),
            ("khaled_samir27", "USERNAME"),
        },
    },

    # 55. Arabic password context
    {
        "text": "كلمة المرور الخاصة بحساب المريض هي MedSecure#927",
        "expected": {
            ("MedSecure#927", "PASSWORD"),
        },
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
