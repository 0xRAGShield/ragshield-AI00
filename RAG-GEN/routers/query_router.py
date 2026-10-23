"""Deterministic rule-based gate between casual chat and retrieval.

Classification uses normalized greeting phrases and a medical denylist.
It does not load a model, use a GPU, or draw random samples, so it is
safe to call from async code. Anything that is not a pure greeting
defaults to RAG.
"""

from __future__ import annotations

import logging
import re
import unicodedata


logger = logging.getLogger("ragshield.router")

_MAX_CHAT_LENGTH = 80

_ALEF_FOLDS = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
    }
)

_TATWEEL = "\u0640"

_GREETING_PHRASES: tuple[str, ...] = (
    "اهلا",
    "مرحبا",
    "سلام",
    "السلام",
    "السلام عليكم",
    "ازيك",
    "عامل ايه",
    "صباح الخير",
    "مساء الخير",
    "صباح الفل",
    "كيفك",
    "اخبارك",
    "كيف حالك",
    "هاي",
    "hi",
    "hello",
    "hey",
    "greetings",
    "salam",
)

_DENY_TERMS: tuple[str, ...] = (
    "جرعة",
    "دواء",
    "علاج",
    "مرض",
    "ألم",
    "الم",
    "مغص",
    "دكتور",
    "صداع",
    "برشام",
    "حقن",
    "mg",
    "dose",
    "tablet",
    "وارفارين",
    "حبوب",
    "warfarin",
    "doctor",
    "pain",
)

_FILLER_PARTICLES = frozenset({"يا", "و"})

_ARABIC_LETTER = re.compile(r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]")

_ARABIC_CHAT_RESPONSE = (
    "أهلاً بك! أنا مساعدك الطبي. كيف يمكنني مساعدتك اليوم؟"
)
_ENGLISH_CHAT_RESPONSE = (
    "Hello. I am your medical assistant. How can I help you today?"
)


def _normalize(text: str) -> str:
    lowered = text.lower()
    stripped: list[str] = []

    for char in lowered:
        if char == _TATWEEL:
            continue

        category = unicodedata.category(char)

        if category in {"Mn", "Me"}:
            continue

        stripped.append(char)

    folded = "".join(stripped).translate(_ALEF_FOLDS)
    return re.sub(r"\s+", " ", folded).strip()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in phrase.split())
    return re.compile(rf"(?<!\w){body}(?!\w)")


def _patterns_longest_first(
    phrases: tuple[str, ...],
) -> tuple[re.Pattern[str], ...]:
    normalized = tuple(
        dict.fromkeys(
            item
            for phrase in phrases
            if (item := _normalize(phrase))
        )
    )
    ordered = sorted(normalized, key=len, reverse=True)
    return tuple(_phrase_pattern(phrase) for phrase in ordered)


_GREETING_PATTERNS = _patterns_longest_first(_GREETING_PHRASES)
_DENY_PATTERNS = _patterns_longest_first(_DENY_TERMS)


def _contains_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def _strip_greetings(text: str) -> tuple[str, bool]:
    matched = False

    for pattern in _GREETING_PATTERNS:
        text, count = pattern.subn(" ", text)

        if count:
            matched = True

    return re.sub(r"\s+", " ", text).strip(), matched


def _strip_punctuation_and_fillers(text: str) -> str:
    characters: list[str] = []

    for char in text:
        if unicodedata.category(char).startswith("P"):
            characters.append(" ")
        else:
            characters.append(char)

    tokens = "".join(characters).split()
    kept = [token for token in tokens if token not in _FILLER_PARTICLES]
    return " ".join(kept)


def _classify_query(query: object) -> str:
    if not isinstance(query, str) or not query.strip():
        return "RAG"

    if any(char.isdigit() for char in query):
        return "RAG"

    normalized = _normalize(query)

    if not normalized or len(normalized) > _MAX_CHAT_LENGTH:
        return "RAG"

    if _contains_any(normalized, _DENY_PATTERNS):
        return "RAG"

    remainder, matched_greeting = _strip_greetings(normalized)
    remainder = _strip_punctuation_and_fillers(remainder)

    if not matched_greeting or remainder:
        return "RAG"

    return "CHAT"


def classify_query(query: str) -> str:
    """Return exactly \"CHAT\" or \"RAG\"."""

    try:
        return _classify_query(query)
    except Exception:
        logger.exception(
            "query classification failed; defaulting to RAG"
        )
        return "RAG"


def generate_chat_response(query: str) -> str:
    """Return a fixed Arabic or English greeting."""

    if not isinstance(query, str):
        raise TypeError("query must be a string.")

    if not query.strip():
        raise ValueError("query cannot be empty.")

    if _ARABIC_LETTER.search(query):
        return _ARABIC_CHAT_RESPONSE

    return _ENGLISH_CHAT_RESPONSE
