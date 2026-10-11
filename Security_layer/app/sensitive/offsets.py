
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.sensitive.labels import parse_bio_label


@dataclass(frozen=True)
class TokenAnnotation:
    token_index: int
    start: int
    end: int
    label: str
    confidence: float
    is_special: bool = False


@dataclass(frozen=True)
class EntitySpan:
    text: str
    entity: str
    start: int
    end: int
    confidence: float


def is_special_or_empty_offset(
    start: int,
    end: int,
    *,
    is_special: bool = False,
) -> bool:
    if is_special:
        return True
    if start == 0 and end == 0:
        return True
    return end <= start


def extract_entity_text(text: str, start: int, end: int) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("start and end must be integers.")

    if start < 0 or end <= start or end > len(text):
        raise ValueError("Invalid character span for the provided text.")

    return text[start:end]


def trim_character_span(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    if start < 0 or end > len(text) or end < start:
        raise ValueError("Invalid character span for the provided text.")

    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


def clamp_character_span(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    start = max(0, start)
    end = min(len(text), end)
    return start, end


def align_entity_span(
    text: str,
    *,
    start: int,
    end: int,
    entity_text: str,
) -> tuple[int, int]:
    """
    Align a tokenizer character span to entity_text inside that span.

    Offsets stay relative to the original text. This does not search the
    whole document, so repeated substrings cannot jump to another match.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not isinstance(entity_text, str) or not entity_text:
        raise ValueError("entity_text must be a non-empty string.")

    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("start and end must be integers.")

    if start < 0 or end < 0 or end <= start:
        raise ValueError("Invalid character span.")

    start, end = clamp_character_span(text, start, end)
    normalized_entity = entity_text.strip()

    if not normalized_entity:
        raise ValueError(
            "entity_text must contain non-whitespace characters."
        )

    if text[start:end] == normalized_entity:
        return start, end

    span_text = text[start:end]
    relative_index = span_text.find(normalized_entity)

    if relative_index >= 0:
        resolved_start = start + relative_index
        resolved_end = resolved_start + len(normalized_entity)

        if text[resolved_start:resolved_end] == normalized_entity:
            return resolved_start, resolved_end

    trimmed_start, trimmed_end = trim_character_span(text, start, end)

    if (
        trimmed_end > trimmed_start
        and text[trimmed_start:trimmed_end] == normalized_entity
    ):
        return trimmed_start, trimmed_end

    raise ValueError(
        "Unable to align entity span with the original input text."
    )


def _get_entity_type(label: str) -> str | None:
    """Return the entity type from a BIO label, or None for O/invalid labels."""
    prefix, entity_type = parse_bio_label(label)

    if prefix == "O" or entity_type is None:
        return None

    return entity_type


def _remove_redundant_overlapping_annotations(
    token_annotations: Sequence[TokenAnnotation],
) -> list[TokenAnnotation]:
    """
    Remove a shorter same-type annotation when a longer, equally or more
    confident annotation starts at the same character offset.

    Example:
        B-PHONE [35, 36] -> "0"
        B-PHONE [35, 38] -> "010"

    Keep the longer annotation and discard the redundant shorter one.
    """
    filtered_annotations: list[TokenAnnotation] = []

    for annotation in token_annotations:
        if annotation.is_special or is_special_or_empty_offset(
            annotation.start,
            annotation.end,
            is_special=annotation.is_special,
        ):
            filtered_annotations.append(annotation)
            continue

        entity_type = _get_entity_type(annotation.label)

        if entity_type is None:
            filtered_annotations.append(annotation)
            continue

        is_redundant = False

        for other in token_annotations:
            if other is annotation or other.is_special:
                continue

            if is_special_or_empty_offset(
                other.start,
                other.end,
                is_special=other.is_special,
            ):
                continue

            other_type = _get_entity_type(other.label)

            if other_type is None:
                continue

            if (
                other_type == entity_type
                and other.start == annotation.start
                and other.end > annotation.end
            ):
                is_redundant = True
                break

        if not is_redundant:
            filtered_annotations.append(annotation)

    return filtered_annotations


def reconstruct_entities(
    text: str,
    token_annotations: Sequence[TokenAnnotation],
    *,
    confidence_threshold: float = 0.0,
) -> list[EntitySpan]:
    """
    Decode BIO token annotations into character spans in text.

    Each annotation keeps its original tokenizer/model token_index.
    Special tokens and empty offsets are ignored. Invalid I-tags start
    a new entity. Redundant overlapping annotations are filtered first.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not text:
        return []

    annotations = _remove_redundant_overlapping_annotations(
        token_annotations
    )

    entities: list[EntitySpan] = []

    current_type: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    current_scores: list[float] = []

    def _close_current() -> None:
        nonlocal current_type
        nonlocal current_start
        nonlocal current_end
        nonlocal current_scores

        if (
            current_type is None
            or current_start is None
            or current_end is None
            or not current_scores
        ):
            current_type = None
            current_start = None
            current_end = None
            current_scores = []
            return

        confidence = sum(current_scores) / len(current_scores)

        if confidence >= confidence_threshold:
            span = _finalize_span(
                text,
                entity=current_type,
                start=current_start,
                end=current_end,
                confidence=confidence,
            )

            if span is not None:
                entities.append(span)

        current_type = None
        current_start = None
        current_end = None
        current_scores = []

    for annotation in annotations:
        if annotation.is_special or is_special_or_empty_offset(
            annotation.start,
            annotation.end,
            is_special=annotation.is_special,
        ):
            continue

        prefix, entity_type = parse_bio_label(annotation.label)

        if prefix == "O" or entity_type is None:
            _close_current()
            continue

        continues_current = (
            prefix == "I"
            and current_type == entity_type
            and current_start is not None
            and current_end is not None
        )

        if continues_current:
            current_end = max(current_end, annotation.end)
            current_scores.append(annotation.confidence)
            continue

        _close_current()

        current_type = entity_type
        current_start = annotation.start
        current_end = annotation.end
        current_scores = [annotation.confidence]

    _close_current()

    return entities


def _finalize_span(
    text: str,
    *,
    entity: str,
    start: int,
    end: int,
    confidence: float,
) -> EntitySpan | None:
    start, end = clamp_character_span(text, start, end)

    if end <= start:
        return None

    start, end = trim_character_span(text, start, end)

    if end <= start:
        return None

    entity_text = extract_entity_text(text, start, end)

    start, end = align_entity_span(
        text,
        start=start,
        end=end,
        entity_text=entity_text,
    )

    entity_text = extract_entity_text(text, start, end)

    if text[start:end] != entity_text:
        return None

    return EntitySpan(
        text=entity_text,
        entity=entity,
        start=start,
        end=end,
        confidence=confidence,
    )
