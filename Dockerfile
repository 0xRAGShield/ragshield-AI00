# RAGShield application image (CUDA-enabled for Qwen3-VL + bitsandbytes).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface 
    

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Keep the CUDA-enabled torch from the base image.
# Upgrade transformers for Qwen3-VL.
RUN awk '!/^torch==/ && !/^transformers==/' /app/requirements.txt \
        > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt \
    && pip install --no-cache-dir torchvision "transformers>=4.57.0" \
    && pip install --no-cache-dir huggingface_hub

# Application code.
COPY RAG-GEN/ /app/RAG-GEN/

# Create the local model directory expected by settings.py.
RUN mkdir -p /app/RAG-GEN/models/qwen3-vl-8b

# Download Qwen3-VL locally during image build.
# This matches:
# LLMSettings.local_weights_dir = "models/qwen3-vl-8b"
RUN huggingface-cli download \
        Qwen/Qwen3-VL-8B-Instruct \
        --local-dir /app/RAG-GEN/models/qwen3-vl-8b

# Download BGE-M3 into the Hugging Face cache.
RUN huggingface-cli download \
        BAAI/bge-m3

# Download CrossEncoder into the Hugging Face cache.
RUN huggingface-cli download \
        cross-encoder/ms-marco-MiniLM-L-6-v2

WORKDIR /app/RAG-GEN

EXPOSE 8000

CMD ["uvicorn", "RAG_GEN:app", "--host", "0.0.0.0", "--port", "8000"]