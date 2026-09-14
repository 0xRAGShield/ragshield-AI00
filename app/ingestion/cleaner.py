from __future__ import annotations
import re
import unicodedata
from app.models.evidence_models import SourceDocument


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?])")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_SEPARATOR_LINE = re.compile(r"^\s*(?:[-_=─━—]{5,})\s*$",re.MULTILINE)


def clean_document_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL.sub("", cleaned)
    cleaned = _TRAILING_SPACE.sub("\n", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _MULTI_NL.sub("\n\n", cleaned)
    cleaned = _SEPARATOR_LINE.sub("", cleaned)

    return cleaned.strip()


def clean_document(document: SourceDocument) -> SourceDocument:
    cleaned_text = clean_document_text(document.text)

    if not cleaned_text:
        raise ValueError(
            f"Document became empty after cleaning: {document.source_id}"
        )

    return document.model_copy(update={"text": cleaned_text})