from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.models.evidence_models import SourceDocument


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?؟])")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")

_SEPARATOR_LINE = re.compile(
    r"^[ \t]*[-_=─━—]{5,}[ \t]*\n?",
    re.MULTILINE,
)


@dataclass(frozen=True)
class CleanLimits:
    """Safety limits for document text cleaning."""

    max_input_chars: int = 10_000_000
    max_output_chars: int = 10_000_000

    def __post_init__(self) -> None:
        if self.max_input_chars <= 0:
            raise ValueError(
                "max_input_chars must be greater than zero."
            )

        if self.max_output_chars <= 0:
            raise ValueError(
                "max_output_chars must be greater than zero."
            )


def clean_document_text(
    text: str,
    limits: CleanLimits | None = None,
) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not text.strip():
        raise ValueError("text cannot be empty.")

    limits = limits or CleanLimits()

    if len(text) > limits.max_input_chars:
        raise ValueError(
            "text exceeds the maximum allowed input size."
        )

    cleaned = unicodedata.normalize(
        "NFKC",
        text,
    )

    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("\r", "\n")

    cleaned = _CONTROL.sub("", cleaned)

    cleaned = _SEPARATOR_LINE.sub(
        "",
        cleaned,
    )

    cleaned = _TRAILING_SPACE.sub(
        "\n",
        cleaned,
    )

    cleaned = _MULTI_SPACE.sub(
        " ",
        cleaned,
    )

    cleaned = _SPACE_BEFORE_PUNCT.sub(
        r"\1",
        cleaned,
    )

    cleaned = _MULTI_NL.sub(
        "\n\n",
        cleaned,
    )

    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError(
            "Text became empty after cleaning."
        )

    if len(cleaned) > limits.max_output_chars:
        raise ValueError(
            "cleaned text exceeds the maximum allowed output size."
        )

    return cleaned


def clean_document(
    document: SourceDocument,
    limits: CleanLimits | None = None,
) -> SourceDocument:
    if not isinstance(document, SourceDocument):
        raise TypeError(
            "document must be a SourceDocument."
        )

    cleaned_text = clean_document_text(
        document.text,
        limits=limits,
    )

    return document.model_copy(
        update={
            "text": cleaned_text,
        },
    )