from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "Security_layer"
sys.path.insert(0, str(APP_ROOT))

from app.sensitive.labels import DEFAULT_PRETRAINED_MODEL, NUM_LABELS
from app.sensitive.model import SensitiveDataModelConfig, XLMRobertaNERModel


@pytest.mark.skipif(
    os.environ.get("RUN_SENSITIVE_NER") != "1",
    reason="Set RUN_SENSITIVE_NER=1 to load the local V3 bilingual checkpoint.",
)
def test_local_v3_checkpoint_loads_offline() -> None:
    model_path = Path(DEFAULT_PRETRAINED_MODEL)
    assert model_path.name == "bilingual_pii_xlmr_V3OFcombined"
    assert (model_path / "config.json").is_file()
    assert (model_path / "checkpoint-31512" / "model.safetensors").is_file()
    assert (model_path / "checkpoint-21008" / "tokenizer.json").is_file()

    model = XLMRobertaNERModel(
        SensitiveDataModelConfig(
            model_name_or_path=str(model_path),
            device="cpu",
            confidence_threshold=0.5,
            max_length=512,
        )
    )
    assert Path(model.model_name).name == "bilingual_pii_xlmr_V3OFcombined"
    assert len(model.id2label) == NUM_LABELS
    text = "My email is demo.user@example.com"
    predictions = model.predict(text)
    for prediction in predictions:
        assert text[prediction.start:prediction.end] == prediction.text
        assert 0.0 <= prediction.score <= 1.0
