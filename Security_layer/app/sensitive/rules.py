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


# 1. MEDICAL RECORD NUMBER (MRN)
_MRN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])MRN[- ]?[A-Za-z0-9]{4,}(?![A-Za-z0-9])",
    re.IGNORECASE,
)


# 2. NATIONAL ID

_NATIONAL_ID_PATTERN = re.compile(r"(?<!\d)\d{14}(?!\d)")

_NATIONAL_ID_CONTEXT = re.compile(
    r"(?:national\s+id|national\s+identity|id\s+number|"
    r"رقم\s*(?:البطاقة\s*القومية|الرقم\s*القومي)|"
    r"البطاقة\s*القومية|الرقم\s*القومي)",
    re.IGNORECASE,
)


# 3. CREDIT CARD
_CREDIT_CARD_CONTEXT = re.compile(
    r"(?:credit\s+card|debit\s+card|card\s+number|"
    r"بطاقة\s*الائتمان|بطاقة\s*الخصم|رقم\s*البطاقة)",
    re.IGNORECASE,
)

# Matches 13-19 digits, optionally separated by spaces or hyphens.
_CREDIT_CARD_PATTERN = re.compile(
    r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
)


def _passes_luhn(number: str) -> bool:
    """Return True if a numeric string passes the Luhn checksum."""
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


def _has_credit_card_context(
    text: str,
    start: int,
    end: int,
) -> bool:
    """Check nearby text for credit-card-related wording."""
    context_start = max(0, start - 100)
    context_end = min(len(text), end + 30)

    context = text[context_start:context_end]

    return bool(_CREDIT_CARD_CONTEXT.search(context))


def find_rule_entities(text: str) -> list[RuleEntity]:
    """Find sensitive entities using high-confidence pattern rules."""
    found: list[RuleEntity] = []

    # MRN
    for match in _MRN_PATTERN.finditer(text):
        found.append(
            RuleEntity(
                text=match.group(),
                label="MEDICAL_RECORD_NUMBER",
                start=match.start(),
                end=match.end(),
                score=0.99,
            )
        )

    # National ID
    for match in _NATIONAL_ID_PATTERN.finditer(text):
        context_start = max(0, match.start() - 80)
        context_end = min(len(text), match.end() + 20)

        context = text[context_start:context_end]

        if _NATIONAL_ID_CONTEXT.search(context):
            found.append(
                RuleEntity(
                    text=match.group(),
                    label="ID_CARD_NUMBER",
                    start=match.start(),
                    end=match.end(),
                    score=0.99,
                )
            )

    # Credit Card
    for match in _CREDIT_CARD_PATTERN.finditer(text):
        # Remove spaces and hyphens before validation.
        number = re.sub(r"[ -]", "", match.group())

        # Require both a valid Luhn checksum and nearby context.
        if (
            _passes_luhn(number)
            and _has_credit_card_context(
                text,
                match.start(),
                match.end(),
            )
        ):
            found.append(
                RuleEntity(
                    text=match.group(),
                    label="CREDIT_CARD",
                    start=match.start(),
                    end=match.end(),
                    score=0.99,
                )
            )

    return found
