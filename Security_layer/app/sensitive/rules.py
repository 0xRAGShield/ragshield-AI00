
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleEntity:
    text: str
    label: str
    start: int
    end: int
    score: float


# NATIONAL ID

_NATIONAL_ID_PATTERN = re.compile(
    r"(?<!\d)\d{14}(?!\d)"
)

_NATIONAL_ID_CONTEXT = re.compile(
    r"(?:"
    r"national\s+id|"
    r"national\s+identity|"
    r"identification\s+number|"
    r"id\s+number|"
    r"\bid\s*[:#]\s*|"
    r"رقم\s*(?:البطاقة\s*القومية|الرقم\s*القومي)|"
    r"البطاقة\s*القومية|"
    r"بطاقة\s*الرقم\s*القومي|"
    r"الرقم\s*القومي|"
    r"بطاقتي(?!\s*الائتمان)"
    r")",
    re.IGNORECASE,
)

# CREDIT CARD

_CREDIT_CARD_CONTEXT = re.compile(
    r"(?:"
    r"credit\s+card|"
    r"debit\s+card|"
    r"card\s+number|"
    r"بطاقة\s*الائتمان|"
    r"بطاقة\s*الخصم|"
    r"بطاقتي\s*الائتمان|"
    r"رقم\s*البطاقة(?!\s*القومية)"
    r")",
    re.IGNORECASE,
)

_INVOICE_AMOUNT_CONTEXT = re.compile(
    r"(?:invoice\s+total|egyptian\s+pounds|\bpounds\b|جنيه)",
    re.IGNORECASE,
)

_GROUPED_CARD_PATTERN = re.compile(
    r"(?<!\d)(\d{4}(?:[ -]\d{4}){3})(?!\d)"
)

_COMPACT_CARD_PATTERN = re.compile(
    r"(?<!\d)(\d{13,19})(?!\d)"
)


# MEDICAL RECORD NUMBER / EMPLOYEE ID

_MRN_PREFIX_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(MRN[-]?\d{3,12})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_MEDICAL_ID_PATTERN = re.compile(
    r"(?P<cue>(?<![A-Za-z0-9])MRN(?![A-Za-z])|"
    r"patient\s+id\b|medical\s+record(?:\s+number)?|"
    r"hospital\s+file\s+(?:reference|id|number)|"
    r"رقم\s+الملف\s+الطبي|رقم\s+المريض|رقم\s+الملف|"
    r"تم\s+تسجيل\s+المريض(?:\s+تحت\s+المرجع)?)"
    r"(?P<sep>\s*(?:is\s*)?[:#\-]?\s*)"
    r"(?P<ident>(?:HSP-\d{3,12}|"
    r"[A-Za-z]*\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*))",
    re.IGNORECASE,
)

_EMPLOYEE_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(EMP(?:LOYEE)?-?\d{3,12})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_EMPLOYEE_CONTEXT = re.compile(
    r"(?:employee\s+id|staff\s+id|الرقم\s+الوظيفي|رقم\s+الموظف)",
    re.IGNORECASE,
)

_MEDICAL_CONTEXT = re.compile(
    r"(?:medical\s+record|hospital\s+file|patient\s+id|"
    r"رقم\s+الملف\s+الطبي|رقم\s+المريض|رقم\s+الملف|"
    r"تم\s+تسجيل\s+المريض|المرجع)",
    re.IGNORECASE,
)

_VISITOR_OR_GENERIC_CODE_CONTEXT = re.compile(
    r"(?:registration\s+code|visitor(?:\s+id)?|sample\s+code|"
    r"batch\s+number|build\s+number|application\s+version)",
    re.IGNORECASE,
)


# PHONE

_PHONE_PATTERNS = (
    # International numbers with a parenthetical area code:
    # +20 (100) 765-4321
    re.compile(
        r"(?<![\w])\+\d{1,3}\s*\(\d{2,4}\)\s*\d{3}[\s.-]\d{4}(?![\w])"
    ),

    # International numbers with separators, e.g. +20 10 1234 5678
    re.compile(
        r"(?<![\w])\+\d{1,3}(?:[\s-]\d{2,4}){2,4}(?![\w])"
    ),

    # International numbers without separators, e.g. +201012345678
    re.compile(
        r"(?<![\w])\+\d{10,15}(?!\d)"
    ),

    # US-style numbers: (123) 456-7890
    re.compile(
        r"(?<![\w])\(\d{3}\)\s*\d{3}[\s.-]\d{4}(?![\w])"
    ),

    # Separated numbers: 123-456-7890
    re.compile(
        r"(?<![\w])\d{3}[\s.-]\d{3}[\s.-]\d{4}(?![\w])"
    ),

    # Egyptian mobile numbers: 01012345678, 01112345678,
    # 01212345678, 01512345678
    re.compile(
        r"(?<!\d)01[0125]\d{8}(?!\d)"
    ),

    # Egyptian mobile numbers with country code: +201012345678
    # or 00201012345678 (country code + 10-digit mobile)
    re.compile(
        r"(?<!\d)(?:\+20|0020)[ -]?1[0125](?:[ -]?\d){8}(?!\d)"
    ),
)


# EMAIL

_EMAIL_PATTERN = re.compile(
    r"(?<![\w.])[A-Za-z0-9._%+\-]+@"
    r"[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?![\w])"
)

# ADDRESS

_ENGLISH_ADDRESS_PATTERN = re.compile(
    r"(?<![\w])(\d{1,5}[A-Za-z]?\s+"
    r"(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|Road|Avenue|Lane|Drive|Boulevard|Way|St\.?)"
    r"(?:,\s*[A-Z][a-z]+){0,2})"
)

_ARABIC_LETTER = r"[\u0621-\u064A]"

_ARABIC_ADDRESS_PATTERN = re.compile(
    rf"(?<!\d)(\d{{1,5}}\s+(?:شارع|طريق|حي)\s+"
    rf"{_ARABIC_LETTER}+(?:\s*،\s*{_ARABIC_LETTER}+){{0,2}})"
)


# SHARED HELPERS

def _passes_luhn(number: str) -> bool:

    digits = [int(char) for char in number]

    if not 13 <= len(digits) <= 19:
        return False

    checksum = 0

    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2

            if digit > 9:
                digit -= 9

        checksum += digit

    return checksum % 10 == 0


def _has_context(
    pattern: re.Pattern[str],
    text: str,
    start: int,
    end: int,
) -> bool:

    context_start = max(0, start - 100)
    context_end = min(len(text), end + 30)

    return bool(
        pattern.search(text[context_start:context_end])
    )


def _is_national_id_context(
    text: str,
    start: int,
    end: int,
) -> bool:

    return _has_context(
        _NATIONAL_ID_CONTEXT,
        text,
        start,
        end,
    )


def _is_credit_card_context(
    text: str,
    start: int,
    end: int,
) -> bool:
    return _has_context(
        _CREDIT_CARD_CONTEXT,
        text,
        start,
        end,
    )


def _is_invoice_amount_context(
    text: str,
    start: int,
    end: int,
) -> bool:
    return _has_context(
        _INVOICE_AMOUNT_CONTEXT,
        text,
        start,
        end,
    )


def _digit_payload(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


def adjust_numeric_identifier_label(
    text: str,
    start: int,
    end: int,
    label: str,
) -> str | None:
    
    digits = _digit_payload(text[start:end])
    national = _is_national_id_context(text, start, end)
    credit = _is_credit_card_context(text, start, end)

    if len(digits) == 14 and label in {
        "CREDIT_CARD",
        "ID_CARD_NUMBER",
        "NATIONAL_ID",
    }:
        if credit and not national:
            return "CREDIT_CARD"
        if national:
            return "ID_CARD_NUMBER"

    if (
        len(digits) == 16
        and label == "CREDIT_CARD"
        and _is_invoice_amount_context(text, start, end)
        and not credit
    ):
        return None

    return label


_CLINICAL_ADDRESS_NOISE = re.compile(
    r"(?i)^(?:the\s+)?(?:clinic|department|hospital|ward|chart|"
    r"cardiology(?:\s+department)?|العيادة|العياده|المستشفى|القسم)$"
)


def is_generic_clinical_address(value: str) -> bool:
    

    stripped = value.strip()
    if not stripped:
        return True

    if any(char.isdigit() for char in stripped):
        return False

    if re.search(
        r"(?i)\b(?:street|road|avenue|lane|drive|boulevard|way)\b",
        stripped,
    ):
        return False

    if any(word in stripped for word in ("شارع", "طريق", "حي")):
        return False

    return bool(_CLINICAL_ADDRESS_NOISE.fullmatch(stripped))


def _append(
    found: list[RuleEntity],
    text: str,
    start: int,
    end: int,
    label: str,
    score: float,
) -> None:

    if end <= start or end > len(text):
        return

    value = text[start:end]

    if not value.strip():
        return

    found.append(
        RuleEntity(
            text=value,
            label=label,
            start=start,
            end=end,
            score=score,
        )
    )


# NATIONAL ID DETECTION

def _find_national_ids(
    text: str,
    found: list[RuleEntity],
) -> None:
    for match in _NATIONAL_ID_PATTERN.finditer(text):
        if _is_national_id_context(
            text,
            match.start(),
            match.end(),
        ):
            _append(
                found,
                text,
                match.start(),
                match.end(),
                "ID_CARD_NUMBER",
                0.99,
            )


# CREDIT CARD DETECTION

def _find_credit_cards(
    text: str,
    found: list[RuleEntity],
) -> None:
    # Grouped card numbers, e.g. 4111 1111 1111 1111.
    for match in _GROUPED_CARD_PATTERN.finditer(text):
        number = re.sub(r"[ -]", "", match.group(1))

        if (
            len(number) == 14
            and _is_national_id_context(
                text,
                match.start(1),
                match.end(1),
            )
        ):
            continue

        score = (
            0.99
            if _passes_luhn(number)
            else 0.93
        )

        _append(
            found,
            text,
            match.start(1),
            match.end(1),
            "CREDIT_CARD",
            score,
        )

    # Compact card numbers, e.g. 4111111111111111.
    for match in _COMPACT_CARD_PATTERN.finditer(text):
        number = match.group(1)

        national_id_context = _is_national_id_context(
            text,
            match.start(1),
            match.end(1),
        )

        # Explicit national-ID context takes precedence.
        if len(number) == 14 and national_id_context:
            continue

        if _is_invoice_amount_context(
            text,
            match.start(1),
            match.end(1),
        ):
            continue

        context = _has_context(
            _CREDIT_CARD_CONTEXT,
            text,
            match.start(1),
            match.end(1),
        )

        luhn = _passes_luhn(number)

        # Do not classify an unlabelled 14-digit number as a card.
        if len(number) == 14 and not context:
            continue

        # For other lengths, require explicit context or a valid
        # Luhn checksum and a commonly used card length.
        if not context and not (
            luhn and len(number) in {13, 15, 16, 19}
        ):
            continue

        _append(
            found,
            text,
            match.start(1),
            match.end(1),
            "CREDIT_CARD",
            0.99 if luhn else 0.93,
        )


# MEDICAL RECORD NUMBER DETECTION

def _is_visitor_or_generic_code_context(
    text: str,
    start: int,
    end: int,
) -> bool:
    return _has_context(
        _VISITOR_OR_GENERIC_CODE_CONTEXT,
        text,
        start,
        end,
    )


def is_spurious_identifier(
    text: str,
    start: int,
    end: int,
    label: str,
) -> bool:
   
    if label not in {"MEDICAL_RECORD_NUMBER", "EMPLOYEE_ID"}:
        return False

    value = text[start:end].strip().upper()
    if _is_visitor_or_generic_code_context(text, start, end):
        if _has_context(_MEDICAL_CONTEXT, text, start, end) and label == "MEDICAL_RECORD_NUMBER":
            return False
        if _has_context(_EMPLOYEE_CONTEXT, text, start, end) and label == "EMPLOYEE_ID":
            return False
        return True

    return value.startswith("VIS-")


def _find_medical_ids(
    text: str,
    found: list[RuleEntity],
) -> None:
    for match in _MRN_PREFIX_PATTERN.finditer(text):
        _append(
            found,
            text,
            match.start(1),
            match.end(1),
            "MEDICAL_RECORD_NUMBER",
            0.99,
        )

    for match in _MEDICAL_ID_PATTERN.finditer(text):
        ident = match.group("ident")
        if sum(char.isdigit() for char in ident) < 5:
            continue
        if ident.upper().startswith("VIS-"):
            continue
        if ident.upper().startswith("EMP"):
            continue
        if _is_visitor_or_generic_code_context(
            text,
            match.start("ident"),
            match.end("ident"),
        ):
            continue

        cue = match.group("cue")
        start = (
            match.start("cue")
            if cue.upper().startswith("MRN")
            else match.start("ident")
        )
        _append(
            found,
            text,
            start,
            match.end("ident"),
            "MEDICAL_RECORD_NUMBER",
            0.99,
        )


def _find_employee_ids(
    text: str,
    found: list[RuleEntity],
) -> None:
    for match in _EMPLOYEE_CODE_PATTERN.finditer(text):
        if not _has_context(
            _EMPLOYEE_CONTEXT,
            text,
            match.start(1),
            match.end(1),
        ):
            continue
        if _is_visitor_or_generic_code_context(
            text,
            match.start(1),
            match.end(1),
        ):
            continue

        _append(
            found,
            text,
            match.start(1),
            match.end(1),
            "EMPLOYEE_ID",
            0.99,
        )


# PHONE DETECTION

def _find_phones(
    text: str,
    found: list[RuleEntity],
) -> None:
    for pattern in _PHONE_PATTERNS:
        for match in pattern.finditer(text):
            _append(
                found,
                text,
                match.start(),
                match.end(),
                "PHONE",
                0.99,
            )


# EMAIL DETECTION

def _find_emails(
    text: str,
    found: list[RuleEntity],
) -> None:
    for match in _EMAIL_PATTERN.finditer(text):
        _append(
            found,
            text,
            match.start(),
            match.end(),
            "EMAIL",
            0.99,
        )


# ADDRESS DETECTION

def _trim_generic_address_tail(
    text: str,
    start: int,
    end: int,
) -> int:
    span = text[start:end]

    match = re.search(
        r"(?:\s*،\s*|\s*,\s*)([^\s,،]+)$",
        span,
    )

    if match and is_generic_clinical_address(match.group(1)):
        return start + match.start()

    return end


def _find_addresses(
    text: str,
    found: list[RuleEntity],
) -> None:
    for pattern in (
        _ENGLISH_ADDRESS_PATTERN,
        _ARABIC_ADDRESS_PATTERN,
    ):
        for match in pattern.finditer(text):
            end = _trim_generic_address_tail(
                text,
                match.start(1),
                match.end(1),
            )

            _append(
                found,
                text,
                match.start(1),
                end,
                "ADDRESS",
                0.96,
            )

# OVERLAP RESOLUTION

def _prefer_longer(
    found: list[RuleEntity],
) -> list[RuleEntity]:
    priority = {
        "CREDIT_CARD": 100,
        "MEDICAL_RECORD_NUMBER": 95,
        "EMPLOYEE_ID": 92,
        "ID_CARD_NUMBER": 90,
        "PHONE": 80,
        "EMAIL": 78,
        "ADDRESS": 60,
    }

    ordered = sorted(
        found,
        key=lambda item: (
            item.start,
            -(item.end - item.start),
            -priority.get(item.label, 0),
        ),
    )

    kept: list[RuleEntity] = []

    for entity in ordered:
        container = next(
            (
                other
                for other in kept
                if other.start <= entity.start
                and other.end >= entity.end
            ),
            None,
        )

        if container is None:
            kept.append(entity)
            continue

        if (
            container.label == entity.label
            or priority.get(container.label, 0)
            >= priority.get(entity.label, 0)
        ):
            continue

        kept.append(entity)

    return kept


# PERSON SPAN CLEANUP

# Conservative list of clinical words that should not be included
# at the end of a PERSON entity.
#
# Keep this list limited to unambiguous role/clinical words.
# Do not add common words that can legitimately be part of names.

_PERSON_CLINICAL_TAILS = (
    "المريض",
    "المريضة",
    "مريض",
    "مريضة",
    "المصاب",
    "المصابة",
    "patient",
    "the patient",
)

# English honorific immediately before a PERSON name. Keep original casing.
_PERSON_HONORIFIC_BEFORE = re.compile(
    r"(?<![A-Za-z])((?:Dr|DR|dr)\.)\s+$"
)

# Arabic preposition letters that can cliticize onto a following name.
# Do not strip these unless a grammatical cue is present.
_ARABIC_PERSON_PROCLITICS = frozenset("بل")
_ARABIC_PERSON_PROCLITIC_CUE = re.compile(
    r"(?:الخاص|الخاصة|الخاصه|التابع|التابعة)\s*$"
)
_ARABIC_NAME_LETTER = re.compile(r"[\u0621-\u064A]")

# Cue-scoped English PERSON recovery. This is not a generic Title-Case rule.
_ENGLISH_GIVEN_NAME = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"
_ENGLISH_NAME_SPAN = rf"{_ENGLISH_GIVEN_NAME}(?:\s+{_ENGLISH_GIVEN_NAME}){{0,3}}"
_ENGLISH_MULTI_NAME_SPAN = (
    rf"{_ENGLISH_GIVEN_NAME}(?:\s+{_ENGLISH_GIVEN_NAME}){{1,3}}"
)
_ENGLISH_DOCTOR_NAME_SPAN = (
    rf"(?:Dr|DR|dr)\.\s+{_ENGLISH_GIVEN_NAME}"
    rf"(?:\s+{_ENGLISH_GIVEN_NAME}){{0,2}}"
)
_NAME_BOUNDARY = r"(?=\s|[.,;:!?\"'”’)]|$)"

_CONTEXTUAL_PERSON_PATTERNS = (
    re.compile(
        rf"\bPatient(?:\s+name)?[:\s]+"
        rf"(?P<name>{_ENGLISH_MULTI_NAME_SPAN}){_NAME_BOUNDARY}"
    ),
    re.compile(
        rf"\bContact\s+(?P<name>{_ENGLISH_MULTI_NAME_SPAN})\s+for\b"
    ),
    re.compile(
        rf"\blists\s+[\"'“‘]?(?P<name>{_ENGLISH_MULTI_NAME_SPAN})"
        rf"[\"'”’]?\s+as\b"
    ),
    re.compile(
        rf"\b(?:applicant|patient)\s+is\s+"
        rf"(?P<name>(?:{_ENGLISH_DOCTOR_NAME_SPAN}|{_ENGLISH_NAME_SPAN}))"
        rf"{_NAME_BOUNDARY}"
    ),
    re.compile(
        rf"\b(?P<name>{_ENGLISH_DOCTOR_NAME_SPAN}){_NAME_BOUNDARY}"
    ),
)

_PERSON_NAME_BLOCKLIST = frozenset(
    {
        "patient",
        "doctor",
        "clinic",
        "hospital",
        "department",
        "cardiology",
        "neurology",
        "outpatient",
        "inpatient",
        "summary",
        "discharge",
        "chart",
        "record",
        "medical",
        "street",
        "road",
        "avenue",
        "lane",
        "drive",
        "boulevard",
        "box",
        "today",
        "tomorrow",
        "essential",
        "intelligence",
        "artificial",
        "contact",
        "applicant",
    }
)


def trim_person_clinical_tail(
    text: str,
    start: int,
    end: int,
    label: str,
) -> tuple[int, int] | None:
   
    if label != "PERSON":
        return start, end

    if not (0 <= start < end <= len(text)):
        return None

    value = text[start:end]
    trimmed = value.rstrip()

    if not trimmed:
        return None

    for tail in _PERSON_CLINICAL_TAILS:
        # Match a whole word at the end, case-insensitively.
        pattern = re.compile(
            rf"(?:^|\s+){re.escape(tail)}\s*$",
            re.IGNORECASE,
        )
        match = pattern.search(trimmed)

        if match:
            new_end = start + match.start()
            # If the match starts at whitespace, exclude that whitespace.
            while new_end > start and text[new_end - 1].isspace():
                new_end -= 1

            if new_end <= start:
                return None

            return start, new_end

    return start, end


def expand_person_honorific(
    text: str,
    start: int,
    end: int,
    label: str,
) -> tuple[int, int] | None:
    if label != "PERSON":
        return start, end

    if not (0 <= start < end <= len(text)):
        return None

    match = _PERSON_HONORIFIC_BEFORE.search(text[:start])
    if match is None:
        return start, end

    new_start = match.start(1)
    if new_start >= end or text[new_start:end].strip() in {"Dr.", "DR.", "dr."}:
        return start, end

    return new_start, end


def trim_arabic_person_proclitic(
    text: str,
    start: int,
    end: int,
    label: str,
) -> tuple[int, int] | None:
    if label != "PERSON":
        return start, end

    if not (0 <= start < end <= len(text)):
        return None

    first = text[start]
    if first not in _ARABIC_PERSON_PROCLITICS:
        return start, end

    if start + 1 >= end or _ARABIC_NAME_LETTER.match(text[start + 1]) is None:
        return start, end

    if _ARABIC_PERSON_PROCLITIC_CUE.search(text[:start]) is None:
        return start, end

    new_start = start + 1
    while new_start < end and text[new_start].isspace():
        new_start += 1

    if new_start >= end:
        return start, end

    return new_start, end


def _is_plausible_english_person_name(value: str) -> bool:
    tokens = value.split()
    if not tokens:
        return False

    if tokens[0].rstrip(".").lower() == "dr":
        tokens = tokens[1:]

    if not 1 <= len(tokens) <= 4:
        return False

    for token in tokens:
        if token.lower() in _PERSON_NAME_BLOCKLIST:
            return False
        if re.fullmatch(_ENGLISH_GIVEN_NAME, token) is None:
            return False

    return True


def _find_contextual_persons(
    text: str,
    found: list[RuleEntity],
) -> None:
    """Recover English PERSON spans the model missed, using narrow cues."""

    for pattern in _CONTEXTUAL_PERSON_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start("name"), match.end("name")
            value = text[start:end]
            if not _is_plausible_english_person_name(value):
                continue
            _append(found, text, start, end, "PERSON", 0.86)


# PUBLIC ENTRY POINT

def find_rule_entities(text: str) -> list[RuleEntity]:
    """Detect sensitive entities using regex-based rules only."""

    found: list[RuleEntity] = []

    _find_medical_ids(text, found)
    _find_employee_ids(text, found)
    _find_national_ids(text, found)
    _find_credit_cards(text, found)
    _find_phones(text, found)
    _find_emails(text, found)
    _find_addresses(text, found)
    _find_contextual_persons(text, found)

    return _prefer_longer(found)
