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
from app.sensitive.offsets import reconstruct_entities
from app.sensitive.rules import find_rule_entities
from app.sensitive.schemas import (
    SensitiveDataDetectionResult,
    SensitiveEntity,
)

logger = logging.getLogger("ragshield.sensitive.detector")


class SensitiveDataDetectorError(RuntimeError):
    """Raised when sensitive-data detection fails."""


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

       # Detect sensitive entities in text.
       # Returns character offsets relative to the exact text argument.

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

        # Apply rule-based detection after model inference.
        rule_entities = find_rule_entities(text)

        for rule in rule_entities:
            # Remove overlapping model predictions so that a
            # rule-confirmed entity is returned as one complete span.
            predictions = [
                prediction
                for prediction in predictions
                if not (
                    prediction.start < rule.end
                    and rule.start < prediction.end
                )
            ]

            predictions.append(
                RawEntityPrediction(
                    text=rule.text,
                    label=rule.label,
                    start=rule.start,
                    end=rule.end,
                    score=rule.score,
                )
            )

        return sorted(
            predictions,
            key=lambda prediction: (
                prediction.start,
                prediction.end,
            ),
        )

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

        return SensitiveEntity(
            text=prediction.text,
            entity=prediction.label,
            start=prediction.start,
            end=prediction.end,
            confidence=prediction.score,
        )
