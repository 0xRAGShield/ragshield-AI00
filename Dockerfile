FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/Security_layer \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-sensitive.txt /app/requirements-sensitive.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       --extra-index-url https://pypi.org/simple \
       -r requirements-sensitive.txt

COPY Security_layer/app/sensitive/ /app/Security_layer/app/sensitive/

COPY models/sensitive/bilingual_pii_xlmr_V3OFcombined/config.json /app/models/sensitive/bilingual_pii_xlmr_V3OFcombined/config.json
COPY models/sensitive/bilingual_pii_xlmr_V3OFcombined/label_mapping.json /app/models/sensitive/bilingual_pii_xlmr_V3OFcombined/label_mapping.json
COPY models/sensitive/bilingual_pii_xlmr_V3OFcombined/checkpoint-31512/model.safetensors /app/models/sensitive/bilingual_pii_xlmr_V3OFcombined/checkpoint-31512/model.safetensors
COPY models/sensitive/bilingual_pii_xlmr_V3OFcombined/checkpoint-21008/tokenizer.json /app/models/sensitive/bilingual_pii_xlmr_V3OFcombined/checkpoint-21008/tokenizer.json
COPY models/sensitive/bilingual_pii_xlmr_V3OFcombined/checkpoint-21008/tokenizer_config.json /app/models/sensitive/bilingual_pii_xlmr_V3OFcombined/checkpoint-21008/tokenizer_config.json

CMD ["python", "-c", "from app.sensitive.model import XLMRobertaNERModel; from app.sensitive.detector import SensitiveDataDetector; m=XLMRobertaNERModel(); d=SensitiveDataDetector(model=m); print('MODEL_LOADED:', m.model_name); print('DEVICE:', m.device); print('LABELS:', len(m.id2label)); r=d.detect('My name is Ahmed. Contact me at ahmed.test@example.com'); print('ENTITIES:', [{'text': e.text, 'label': e.entity, 'start': e.start, 'end': e.end} for e in r.entities]); print('DOCKER_SMOKE_TEST_PASSED')"]
