from __future__ import annotations

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from app.sensitive.labels import normalize_entity_label


class SensitiveEntity(BaseModel):
    """One detected sensitive span in the original input text."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    text: str = Field(min_length=1)
    entity: str = Field(
        min_length=1,
        validation_alias=AliasChoices("entity", "entity_type", "label"),
    )
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("confidence", "score"),
    )

    @computed_field
    @property
    def entity_type(self) -> str:
        return self.entity

    @computed_field
    @property
    def label(self) -> str:
        return self.entity

    @computed_field
    @property
    def score(self) -> float:
        return self.confidence

    @field_validator("entity")
    @classmethod
    def validate_entity(cls, value: str) -> str:
        return normalize_entity_label(value)

    @field_validator("end")
    @classmethod
    def validate_span(cls, end: int, info) -> int:
        start = info.data.get("start")
        if isinstance(start, int) and end <= start:
            raise ValueError("end must be greater than start.")
        return end


class SensitiveDataDetectionResult(BaseModel):
    """Detection output for a single text chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    entities: tuple[SensitiveEntity, ...] = Field(default_factory=tuple)

    @property
    def has_sensitive_data(self) -> bool:
        return bool(self.entities)

    @model_validator(mode="after")
    def validate_entity_offsets(self) -> SensitiveDataDetectionResult:
        text_length = len(self.text)
        for entity in self.entities:
            if entity.end > text_length:
                raise ValueError("Entity offsets exceed the original input text.")
            if self.text[entity.start:entity.end] != entity.text:
                raise ValueError("Entity text does not match the original input offsets.")
        return self
