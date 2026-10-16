
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.sensitive.labels import DEFAULT_PRETRAINED_MODEL
from app.sensitive.model import (
    INERModel,
    RawEntityPrediction,
    SensitiveDataModelConfig,
    XLMRobertaNERModel,
    get_shared_ner_model,
)
from app.sensitive.offsets import (
    EntitySpan,
    integrate_rule_spans,
    merge_adjacent_person_spans,
    merge_adjacent_spans,
    polish_entity_span,
    reconstruct_entities,
)
from app.sensitive.rules import (
    adjust_numeric_identifier_label,
    find_rule_entities,
    is_generic_clinical_address,
    is_spurious_identifier,
    trim_person_clinical_tail,
)
from app.sensitive.schemas import (
    SensitiveDataDetectionResult,
    SensitiveEntity,
)


logger = logging.getLogger("ragshield.sensitive.detector")


class SensitiveDataDetectorError(RuntimeError):
    """Raised when sensitive-data detection fails."""


def normalize_entity_label(label: str) -> str:
    """
    Normalize equivalent entity labels to a single canonical label.

    The model's internal label IDs and configuration remain unchanged.
    Normalization affects the detector's processed output only.
    """
    label_aliases = {
        "ID_CARD_NUMBER": "NATIONAL_ID",
    }

    return label_aliases.get(label, label)


@dataclass(frozen=True)
class SensitiveDataDetectorConfig:
    model_name_or_path: str = DEFAULT_PRETRAINED_MODEL
    device: str = "auto"
    confidence_threshold: float = 0.5
    max_length: int = 512
    reuse_shared_model: bool = True

    def to_model_config(self) -> SensitiveDataModelConfig:
        return SensitiveDataModelConfig(
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
            max_length=self.max_length,
        )


class SensitiveDataDetector:
    def __init__(
        self,
        config: SensitiveDataDetectorConfig | None = None,
        *,
        model: INERModel | None = None,
    ) -> None:
        self._config = config or SensitiveDataDetectorConfig()

        if model is not None:
            self._model = model

        elif self._config.reuse_shared_model:
            self._model = get_shared_ner_model(
                self._config.model_name_or_path,
                self._config.device,
                self._config.confidence_threshold,
                self._config.max_length,
            )

        else:
            self._model = XLMRobertaNERModel(
                config=self._config.to_model_config(),
            )

    @property
    def model_name(self) -> str:
        return self._model.model_name

    def detect(self, text: str) -> SensitiveDataDetectionResult:
        """Detect sensitive entities and return offsets in the original text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text:
            return SensitiveDataDetectionResult(
                text=text,
                entities=(),
            )

        try:
            predictions = self._predict_entities(text)

        except Exception as error:
            raise SensitiveDataDetectorError(
                "Sensitive entity detection failed."
            ) from error

        entities = tuple(
            self._to_sensitive_entity(text, prediction)
            for prediction in predictions
        )

        logger.info(
            "%s",
            {
                "event": "sensitive_detection_complete",
                "model_name": self.model_name,
                "text_length": len(text),
                "entity_count": len(entities),
            },
        )

        return SensitiveDataDetectionResult(
            text=text,
            entities=entities,
        )

    def _predict_entities(
        self,
        text: str,
    ) -> list[RawEntityPrediction]:
        run_token_classification = getattr(
            self._model,
            "run_token_classification",
            None,
        )

        if callable(run_token_classification):
            annotations = run_token_classification(text)

            spans = reconstruct_entities(
                text,
                annotations,
                confidence_threshold=self._config.confidence_threshold,
            )

            predictions = [
                RawEntityPrediction(
                    text=span.text,
                    label=span.entity,
                    start=span.start,
                    end=span.end,
                    score=span.confidence,
                )
                for span in spans
            ]

        else:
            predictions = list(self._model.predict(text))

        model_spans: list[EntitySpan] = []

        for prediction in predictions:
            if (
                prediction.label == "ADDRESS"
                and is_generic_clinical_address(prediction.text)
            ):
                continue

            if is_spurious_identifier(
                text,
                prediction.start,
                prediction.end,
                prediction.label,
            ):
                continue

            model_spans.append(
                EntitySpan(
                    text=prediction.text,
                    entity=prediction.label,
                    start=prediction.start,
                    end=prediction.end,
                    confidence=prediction.score,
                )
            )

        model_spans = merge_adjacent_spans(text, model_spans)

        rule_spans = [
            EntitySpan(
                text=rule.text,
                entity=rule.label,
                start=rule.start,
                end=rule.end,
                confidence=rule.score,
            )
            for rule in find_rule_entities(text)
        ]

        combined = integrate_rule_spans(model_spans, rule_spans)

        # Merge conservatively fragmented PERSON entities.
        combined = merge_adjacent_person_spans(text, combined)

        polished: list[RawEntityPrediction] = []

        for span in combined:
            cleaned = polish_entity_span(text, span)

            if cleaned is None:
                continue

            if is_spurious_identifier(
                text,
                cleaned.start,
                cleaned.end,
                cleaned.entity,
            ):
                continue

            # Trim a clinical-role word accidentally included at the
            # end of a PERSON span. Keep offsets relative to original text.
            corrected_offsets = trim_person_clinical_tail(
                text,
                cleaned.start,
                cleaned.end,
                cleaned.entity,
            )

            if corrected_offsets is None:
                continue

            corrected_start, corrected_end = corrected_offsets
            corrected_text = text[corrected_start:corrected_end]

            adjusted = adjust_numeric_identifier_label(
                text,
                corrected_start,
                corrected_end,
                cleaned.entity,
            )

            if adjusted is None:
                continue

            # Normalize equivalent labels after all label adjustments.
            normalized_label = normalize_entity_label(adjusted)

            polished.append(
                RawEntityPrediction(
                    text=corrected_text,
                    label=normalized_label,
                    start=corrected_start,
                    end=corrected_end,
                    score=cleaned.confidence,
                )
            )

        return polished

    def _to_sensitive_entity(
        self,
        text: str,
        prediction: RawEntityPrediction,
    ) -> SensitiveEntity:
        if (
            prediction.start < 0
            or prediction.end > len(text)
            or prediction.start >= prediction.end
            or text[prediction.start:prediction.end] != prediction.text
        ):
            raise SensitiveDataDetectorError(
                "Detected entity offsets do not match the original input text."
            )

        # Defensive normalization at the public output boundary.
        normalized_label = normalize_entity_label(prediction.label)

        return SensitiveEntity(
            text=prediction.text,
            entity=normalized_label,
            start=prediction.start,
            end=prediction.end,
            confidence=prediction.score,
        )
