
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.sensitive.labels import parse_bio_label


_MERGEABLE_ENTITY_TYPES = frozenset({"ADDRESS", "PHONE"})
_PHONE_GAP_PATTERN = re.compile(r"^[\s\d()+./-]*$")
_ADDRESS_GAP_PATTERN = re.compile(r"^[\s,،.-]*$")
_STRUCTURED_RULE_LABELS = frozenset(
    {
        "PHONE",
        "EMAIL",
        "CREDIT_CARD",
        "ID_CARD_NUMBER",
        "NATIONAL_ID",
        "MEDICAL_RECORD_NUMBER",
        "EMPLOYEE_ID",
        "ADDRESS",
    }
)
_NATIONAL_ID_LABELS = frozenset({"ID_CARD_NUMBER", "NATIONAL_ID"})
_LEADING_WRAP_CHARS = frozenset("<[{(\"'\u00ab")
_TRAILING_WRAP_CHARS = frozenset(".,;:!?)]}>\"'\u00bb\u060c\u061b\u06d4")


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


def trim_entity_boundaries(
    text: str,
    start: int,
    end: int,
    entity_type: str | None = None,
) -> tuple[int, int]:
    """
    Strip wrapping punctuation that is outside the entity.

    Preserves ``+`` on phone numbers, internal email/IP dots, Arabic text,
    Arabic-Indic digits, and parentheses that belong to a phone format.
    """
    start, end = clamp_character_span(text, start, end)
    start, end = trim_character_span(text, start, end)
    entity_type = (entity_type or "").upper() or None

    while end > start:
        char = text[end - 1]
        if char.isspace() or char in _TRAILING_WRAP_CHARS:
            if (
                entity_type == "PHONE"
                and char == ")"
                and "(" in text[start:end - 1]
            ):
                break
            end -= 1
            continue
        break

    while start < end:
        char = text[start]
        if entity_type == "PHONE" and char == "+":
            break
        if (
            entity_type == "PHONE"
            and char == "("
            and ")" in text[start + 1:end]
        ):
            break
        if char.isspace() or char in _LEADING_WRAP_CHARS:
            start += 1
            continue
        break

    return start, end


def polish_entity_span(text: str, span: EntitySpan) -> EntitySpan | None:
    start, end = trim_entity_boundaries(
        text,
        span.start,
        span.end,
        span.entity,
    )
    if end <= start:
        return None
    value = extract_entity_text(text, start, end)
    if text[start:end] != value:
        return None
    return EntitySpan(
        text=value,
        entity=span.entity,
        start=start,
        end=end,
        confidence=span.confidence,
    )


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
    start, end = trim_entity_boundaries(text, start, end, entity)

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


def _spans_overlap(left: EntitySpan, right: EntitySpan) -> bool:
    return left.start < right.end and right.start < left.end


def _same_span(left: EntitySpan, right: EntitySpan) -> bool:
    return (
        left.start == right.start
        and left.end == right.end
        and left.entity == right.entity
    )


def _extend_span(
    text: str,
    current: EntitySpan,
    incoming: EntitySpan,
) -> EntitySpan:
    start = min(current.start, incoming.start)
    end = max(current.end, incoming.end)
    scores = [current.confidence, incoming.confidence]
    return EntitySpan(
        text=text[start:end],
        entity=current.entity,
        start=start,
        end=end,
        confidence=sum(scores) / len(scores),
    )


def merge_adjacent_spans(
    text: str,
    spans: Sequence[EntitySpan],
) -> list[EntitySpan]:
    """
    Join adjacent ADDRESS or PHONE fragments using the original text.

    PERSON and other entity types are left as separate spans.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not spans:
        return []

    ordered = sorted(spans, key=lambda span: (span.start, span.end, span.entity))
    merged: list[EntitySpan] = []

    for span in ordered:
        if not merged:
            merged.append(span)
            continue

        previous = merged[-1]
        if previous.entity != span.entity or span.entity not in _MERGEABLE_ENTITY_TYPES:
            merged.append(span)
            continue

        if span.start <= previous.end:
            merged[-1] = _extend_span(text, previous, span)
            continue

        gap = text[previous.end:span.start]
        can_merge = (
            span.entity == "PHONE"
            and bool(_PHONE_GAP_PATTERN.fullmatch(gap))
        ) or (
            span.entity == "ADDRESS"
            and bool(_ADDRESS_GAP_PATTERN.fullmatch(gap))
        )
        if can_merge:
            merged[-1] = _extend_span(text, previous, span)
            continue

        merged.append(span)

    return merged


def integrate_rule_spans(
    model_spans: Sequence[EntitySpan],
    rule_spans: Sequence[EntitySpan],
) -> list[EntitySpan]:
    """
    Combine NER spans with rule spans.

    Same-type partial overlaps are replaced by the rule span. Exact duplicate
    spans keep the model confidence. Unrelated model entities such as PERSON
    are kept even when a structured rule is nearby.
    """
    kept_model: list[EntitySpan] = []
    for model in model_spans:
        drop = False
        for rule in rule_spans:
            if not _spans_overlap(model, rule):
                continue
            if _same_span(model, rule):
                drop = False
                break
            if model.entity == rule.entity:
                drop = True
                break
            if (
                model.entity in _NATIONAL_ID_LABELS
                and rule.entity in _NATIONAL_ID_LABELS
            ):
                drop = model.entity != "ID_CARD_NUMBER"
                break
            if (
                model.entity in _STRUCTURED_RULE_LABELS
                and rule.entity in _STRUCTURED_RULE_LABELS
            ):
                drop = True
                break
        if not drop:
            kept_model.append(model)

    kept_keys = {(span.start, span.end, span.entity) for span in kept_model}
    extra_rules = [
        rule
        for rule in rule_spans
        if (rule.start, rule.end, rule.entity) not in kept_keys
    ]
    combined = list(kept_model) + extra_rules
    return canonicalize_egyptian_national_ids(
        sorted(combined, key=lambda span: (span.start, span.end, span.entity))
    )


def _digit_payload(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


def canonicalize_egyptian_national_ids(
    spans: Sequence[EntitySpan],
) -> list[EntitySpan]:
    """
    Keep one canonical ID_CARD_NUMBER span when the same 14-digit Egyptian
    national ID is also tagged NATIONAL_ID. Other labels and other spans stay.
    """
    national_spans = [
        span
        for span in spans
        if span.entity in _NATIONAL_ID_LABELS and len(_digit_payload(span.text)) == 14
    ]
    if not national_spans:
        return list(spans)

    drop: set[tuple[int, int, str]] = set()
    for left in national_spans:
        for right in national_spans:
            if left is right or not _spans_overlap(left, right):
                continue
            if _digit_payload(left.text) != _digit_payload(right.text):
                continue
            if left.entity == "NATIONAL_ID" and right.entity == "ID_CARD_NUMBER":
                drop.add((left.start, left.end, left.entity))
            elif (
                left.entity == "NATIONAL_ID"
                and right.entity == "NATIONAL_ID"
                and (left.end - left.start) < (right.end - right.start)
            ):
                drop.add((left.start, left.end, left.entity))

    return [
        span
        for span in spans
        if (span.start, span.end, span.entity) not in drop
    ]

def merge_adjacent_person_spans(
    text: str,
    spans: list[EntitySpan],
) -> list[EntitySpan]:
    """
    Merge conservatively split PERSON spans when:
    - Both spans are PERSON.
    - Only whitespace separates them.
    - One span contains one token and the other contains multiple tokens.

    Original-text offsets are preserved.
    """
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[EntitySpan] = []

    for span in ordered:
        if not (0 <= span.start < span.end <= len(text)):
            merged.append(span)
            continue

        if text[span.start:span.end] != span.text:
            merged.append(span)
            continue

        if merged:
            previous = merged[-1]

            if (
                previous.entity == "PERSON"
                and span.entity == "PERSON"
                and previous.end <= span.start
            ):
                gap = text[previous.end:span.start]
                previous_tokens = previous.text.split()
                current_tokens = span.text.split()

                # Avoid merging two single-token names automatically.
                one_side_is_fragment = (
                    (len(previous_tokens) == 1 and len(current_tokens) > 1)
                    or (len(previous_tokens) > 1 and len(current_tokens) == 1)
                )

                if gap and gap.isspace() and one_side_is_fragment:
                    merged[-1] = EntitySpan(
                        text=text[previous.start:span.end],
                        entity="PERSON",
                        start=previous.start,
                        end=span.end,
                        confidence=min(
                            previous.confidence,
                            span.confidence,
                        ),
                    )
                    continue

        merged.append(span)

    return merged
