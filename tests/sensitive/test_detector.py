from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.sensitive import (
    SensitiveDataDetector,
    SensitiveDataDetectorConfig,
    SensitiveEntity,
)
from app.sensitive.labels import (
    CANONICAL_BIO_LABELS,
    DEFAULT_PRETRAINED_MODEL,
    ENTITY_LABELS,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    normalize_entity_label,
)
from app.sensitive.model import (
    RawEntityPrediction,
    SensitiveDataModelConfig,
    SensitiveDataModelError,
    XLMRobertaNERModel,
    get_shared_ner_model,
    locate_model_artifacts,
)
from app.sensitive.offsets import TokenAnnotation, align_entity_span, reconstruct_entities


class FakeNERModel:
    def __init__(self, predictions: dict[str, list[RawEntityPrediction]]) -> None:
        self._predictions = predictions
        self._model_name = "fake-ner-model"

    @property
    def model_name(self) -> str:
        return self._model_name

    def predict(self, text: str) -> list[RawEntityPrediction]:
        return list(self._predictions.get(text, []))


class FakeTokenModel:
    def __init__(self, annotations: list[TokenAnnotation], model_name: str = "fake-token-model") -> None:
        self._annotations = annotations
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def run_token_classification(self, text: str) -> list[TokenAnnotation]:
        return list(self._annotations)

    def predict(self, text: str) -> list[RawEntityPrediction]:
        raise AssertionError("Detector should use run_token_classification, not predict().")


def test_normalize_entity_label_strips_bio_prefix() -> None:
    assert normalize_entity_label("B-PERSON") == "PERSON"
    assert normalize_entity_label("I-NATIONAL_ID") == "NATIONAL_ID"
    assert normalize_entity_label("EMAIL") == "EMAIL"
    assert normalize_entity_label("O") == "O"


def test_canonical_label_mapping_has_35_ids() -> None:
    assert NUM_LABELS == 35
    assert len(CANONICAL_BIO_LABELS) == 35
    assert len(LABEL2ID) == 35
    assert len(ID2LABEL) == 35
    assert len(ENTITY_LABELS) == 17
    assert LABEL2ID["O"] == 0
    assert ID2LABEL[0] == "O"
    assert LABEL2ID["B-NATIONAL_ID"] == 1
    assert LABEL2ID["I-NATIONAL_ID"] == 2
    assert LABEL2ID["B-PERSON"] == 23
    assert LABEL2ID["I-IP_ADDRESS"] == 34
    assert ENTITY_LABELS[0] == "NATIONAL_ID"
    assert ENTITY_LABELS[-1] == "IP_ADDRESS"
    for index, label in enumerate(CANONICAL_BIO_LABELS):
        assert LABEL2ID[label] == index
        assert ID2LABEL[index] == label


def test_align_entity_span_recovers_leading_whitespace() -> None:
    text = "Dr. Omar lives in Cairo."
    start, end = align_entity_span(
        text,
        start=4,
        end=8,
        entity_text="Omar",
    )
    assert text[start:end] == "Omar"
    assert start == 4
    assert end == 8


def test_detector_returns_exact_offsets_with_fake_backend() -> None:
    chunk = "Patient Omar visited 15 Tahrir Street."
    fake_model = FakeNERModel(
        {
            chunk: [
                RawEntityPrediction(
                    text="Omar",
                    label="PERSON",
                    start=8,
                    end=12,
                    score=0.98,
                ),
                RawEntityPrediction(
                    text="15 Tahrir Street",
                    label="ADDRESS",
                    start=21,
                    end=37,
                    score=0.91,
                ),
            ]
        }
    )

    detector = SensitiveDataDetector(model=fake_model)
    result = detector.detect(chunk)

    assert result.has_sensitive_data
    assert len(result.entities) == 2

    for entity in result.entities:
        assert chunk[entity.start:entity.end] == entity.text

    assert result.entities[0] == SensitiveEntity(
        text="Omar",
        label="PERSON",
        start=8,
        end=12,
        score=0.98,
    )
    assert result.entities[1].label == "ADDRESS"
    assert result.entities[1].text == "15 Tahrir Street"


def test_detector_empty_input_returns_no_entities() -> None:
    detector = SensitiveDataDetector(model=FakeNERModel({}))
    result = detector.detect("")
    assert result.text == ""
    assert result.entities == ()
    assert not result.has_sensitive_data


def test_detector_no_entities() -> None:
    text = "No sensitive information is included in this sentence."
    detector = SensitiveDataDetector(model=FakeNERModel({text: []}))
    result = detector.detect(text)
    assert result.text == text
    assert result.entities == ()


def test_get_shared_ner_model_is_cached(monkeypatch) -> None:
    get_shared_ner_model.cache_clear()
    constructed: list[object] = []

    class StubXLMRobertaNERModel:
        def __init__(self, config=None) -> None:
            constructed.append(self)
            self._model_name = "stub-ner-model"

        @property
        def model_name(self) -> str:
            return self._model_name

        def predict(self, text: str) -> list[RawEntityPrediction]:
            return []

    monkeypatch.setattr(
        "app.sensitive.model.XLMRobertaNERModel",
        StubXLMRobertaNERModel,
    )

    first = get_shared_ner_model("stub-model", "cpu", 0.5, 128)
    second = get_shared_ner_model("stub-model", "cpu", 0.5, 128)

    assert first is second
    assert len(constructed) == 1


def test_detector_reuses_shared_model_instance(monkeypatch) -> None:
    shared_model = FakeNERModel({})

    monkeypatch.setattr(
        "app.sensitive.detector.get_shared_ner_model",
        lambda *_args, **_kwargs: shared_model,
    )

    config = SensitiveDataDetectorConfig(reuse_shared_model=True)
    first = SensitiveDataDetector(config=config)
    second = SensitiveDataDetector(config=config)

    assert first._model is second._model


def test_missing_model_directory_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SensitiveDataModelError, match="was not found"):
        XLMRobertaNERModel(
            config=SensitiveDataModelConfig(model_name_or_path=str(missing), device="cpu")
        )


def test_incomplete_model_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(SensitiveDataModelError, match="incomplete"):
        XLMRobertaNERModel(
            config=SensitiveDataModelConfig(model_name_or_path=str(tmp_path), device="cpu")
        )


def test_incompatible_config_json_raises(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"not-a-real-model")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        '{"id2label": {"0": "O", "1": "B-PERSON"}, "label2id": {"O": 0, "B-PERSON": 1}}',
        encoding="utf-8",
    )
    with pytest.raises(SensitiveDataModelError, match="does not match the canonical"):
        XLMRobertaNERModel(
            config=SensitiveDataModelConfig(model_name_or_path=str(tmp_path), device="cpu")
        )


def test_offline_loading_uses_local_files_only(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"not-a-real-model")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    id2label = {str(index): label for index, label in ID2LABEL.items()}
    label2id = dict(LABEL2ID)
    (tmp_path / "config.json").write_text(
        __import__("json").dumps({"id2label": id2label, "label2id": label2id, "model_type": "xlm-roberta"}),
        encoding="utf-8",
    )

    seen: dict[str, bool] = {}

    def fake_from_pretrained(cls_or_path, *args, **kwargs):
        seen["local_files_only"] = bool(kwargs.get("local_files_only"))
        raise RuntimeError("stopped-before-network")

    monkeypatch.setattr("app.sensitive.model.AutoConfig.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("app.sensitive.model.AutoTokenizer.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        "app.sensitive.model.AutoModelForTokenClassification.from_pretrained",
        fake_from_pretrained,
    )

    with pytest.raises(SensitiveDataModelError, match="Offline loading"):
        XLMRobertaNERModel(
            config=SensitiveDataModelConfig(model_name_or_path=str(tmp_path), device="cpu")
        )
    assert seen.get("local_files_only") is True


def test_default_model_path_is_local_bilingual_checkpoint() -> None:
    normalized = DEFAULT_PRETRAINED_MODEL.replace("\\", "/")
    assert normalized.endswith("models/sensitive/bilingual_pii_xlmr_V3OFcombined")
    assert "Davlan/" not in DEFAULT_PRETRAINED_MODEL


def test_split_checkpoint_layout_is_discovered(tmp_path: Path) -> None:
    root = tmp_path / "bilingual_pii_xlmr_V3OFcombined"
    weights_dir = root / "checkpoint-31512"
    tokenizer_dir = root / "checkpoint-21008"
    weights_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    (weights_dir / "model.safetensors").write_bytes(b"not-a-real-model")
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "config.json").write_text(
        __import__("json").dumps(
            {
                "id2label": {str(index): label for index, label in ID2LABEL.items()},
                "label2id": dict(LABEL2ID),
            }
        ),
        encoding="utf-8",
    )

    artifacts = locate_model_artifacts(root)
    assert artifacts.root_dir == root
    assert artifacts.weights_dir == weights_dir
    assert artifacts.tokenizer_dir == tokenizer_dir
    assert artifacts.config_dir == root


def test_mocked_token_inference_english_offsets() -> None:
    text = "Contact Omar at omar@example.com"
    annotations = [
        TokenAnnotation(0, 0, 0, "O", 1.0, True),
        TokenAnnotation(1, 0, 7, "O", 0.99, False),
        TokenAnnotation(2, 8, 12, "B-PERSON", 0.97, False),
        TokenAnnotation(3, 13, 15, "O", 0.99, False),
        TokenAnnotation(4, 16, 32, "B-EMAIL", 0.96, False),
        TokenAnnotation(5, 32, 32, "O", 1.0, True),
    ]
    detector = SensitiveDataDetector(model=FakeTokenModel(annotations))
    result = detector.detect(text)
    assert [entity.entity for entity in result.entities] == ["PERSON", "EMAIL"]
    for entity in result.entities:
        assert text[entity.start:entity.end] == entity.text
    assert result.entities[0].text == "Omar"
    assert result.entities[1].text == "omar@example.com"


def test_mocked_token_inference_arabic_and_mixed_offsets() -> None:
    text = "اسمي منار / Manar and email manar@gmail.com"
    manar_ar = "منار"
    start_ar = text.index(manar_ar)
    end_ar = start_ar + len(manar_ar)
    start_en = text.index("Manar")
    email = "manar@gmail.com"
    start_email = text.index(email)
    annotations = [
        TokenAnnotation(0, 0, 0, "O", 1.0, True),
        TokenAnnotation(1, 0, start_ar, "O", 0.9, False),
        TokenAnnotation(2, start_ar, end_ar, "B-PERSON", 0.95, False),
        TokenAnnotation(3, start_en, start_en + 5, "B-PERSON", 0.94, False),
        TokenAnnotation(4, start_email, start_email + len(email), "B-EMAIL", 0.99, False),
    ]
    detector = SensitiveDataDetector(model=FakeTokenModel(annotations))
    result = detector.detect(text)
    extracted = {entity.text: entity.entity for entity in result.entities}
    assert extracted[manar_ar] == "PERSON"
    assert extracted["Manar"] == "PERSON"
    assert extracted[email] == "EMAIL"
    for entity in result.entities:
        assert text[entity.start:entity.end] == entity.text


@pytest.mark.skipif(
    os.environ.get("RUN_SENSITIVE_NER") != "1",
    reason="Set RUN_SENSITIVE_NER=1 to run the local fine-tuned bilingual NER smoke test.",
)
class TestLiveSensitiveDataDetector:
    @pytest.fixture(scope="class")
    def detector(self) -> SensitiveDataDetector:
        get_shared_ner_model.cache_clear()
        return SensitiveDataDetector(
            config=SensitiveDataDetectorConfig(
                model_name_or_path=DEFAULT_PRETRAINED_MODEL,
                device="cpu",
                confidence_threshold=0.5,
                max_length=256,
            )
        )

    def test_detects_english_entities(self, detector: SensitiveDataDetector) -> None:
        text = "Hello, Omar works at 15 Tahrir Street and email omar@example.com."
        result = detector.detect(text)
        assert result.has_sensitive_data
        for entity in result.entities:
            assert text[entity.start:entity.end] == entity.text
            assert 0.0 <= entity.score <= 1.0
        labels = {entity.label for entity in result.entities}
        assert "PERSON" in labels or "EMAIL" in labels or "ADDRESS" in labels

    def test_detects_arabic_entities(self, detector: SensitiveDataDetector) -> None:
        text = "المريض اسمه أحمد محمد ورقم تليفونه 01012345678."
        result = detector.detect(text)
        assert result.has_sensitive_data
        for entity in result.entities:
            assert text[entity.start:entity.end] == entity.text

    def test_detects_mixed_arabic_english_entities(
        self,
        detector: SensitiveDataDetector,
    ) -> None:
        text = "أنا اسمي منار عبدالباقي وإيميلي manar@gmail.com."
        result = detector.detect(text)
        assert result.has_sensitive_data
        for entity in result.entities:
            assert text[entity.start:entity.end] == entity.text
