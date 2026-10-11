
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_ROOT = PROJECT_ROOT / "Security_layer"
sys.path.insert(0, str(APP_ROOT))

from app.sensitive.model import (
    XLMRobertaNERModel,
    SensitiveDataModelConfig,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sensitive"
    / "bilingual_pii_xlmr"
)

print("Project root:", PROJECT_ROOT)
print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_PATH}"
    )

config = SensitiveDataModelConfig(
    model_name_or_path=str(MODEL_PATH),
    device="cpu",
    confidence_threshold=0.5,
    max_length=512,
)

print("\nLoading local PII model...")

model = XLMRobertaNERModel(config)

print("Model loaded successfully!")

tests = [
    "My phone number is 01000000000",
    "رقم الهاتف: 01000000000",
    "My email is demo.user@example.com",
    "رقم البطاقة القومية: 12345678901234",
]

for text in tests:
    print(f"\nInput: {text}")
    try:
        print(model.predict(text))
    except Exception as exc:
        print(f"Prediction failed: {type(exc).__name__}: {exc}")
