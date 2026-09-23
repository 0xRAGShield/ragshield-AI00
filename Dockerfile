# RAGShield application image (CUDA-enabled for Qwen3-VL + bitsandbytes).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Keep the CUDA-enabled torch from the base image; upgrade transformers for Qwen3-VL.
RUN awk '!/^torch==/ && !/^transformers==/' /app/requirements.txt \
        > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt \
    && pip install --no-cache-dir torchvision "transformers>=4.57.0"

COPY RAG-GEN/ /app/RAG-GEN/

# Local model weights (must exist on the build host; not downloaded at runtime).
#COPY RAG-GEN/models/qwen3-vl-8b/ /app/RAG-GEN/models/qwen3-vl-8b/
COPY RAG-GEN/models/bge-m3/ /tmp/model-cache/bge-m3/
COPY RAG-GEN/models/cross-encoder-ms-marco-MiniLM-L-6-v2/ /tmp/model-cache/cross-encoder/

RUN python - <<'PY'
import hashlib
import shutil
from pathlib import Path

HF_HOME = Path("/app/.cache/huggingface/hub")


def install_model(repo_id: str, source: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"Missing local model directory: {source}")

    snapshot_id = hashlib.sha256(
        repo_id.encode("utf-8")
    ).hexdigest()[:40]

    cache_name = "models--" + repo_id.replace("/", "--")
    cache_dir = HF_HOME / cache_name
    snapshot_dir = cache_dir / "snapshots" / snapshot_id
    refs_dir = cache_dir / "refs"

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        destination = snapshot_dir / item.name

        if item.is_dir():
            shutil.copytree(
                item,
                destination,
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(item, destination)

    (refs_dir / "main").write_text(
        snapshot_id,
        encoding="utf-8",
    )


install_model("BAAI/bge-m3", Path("/tmp/model-cache/bge-m3"))
install_model(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    Path("/tmp/model-cache/cross-encoder"),
)
PY

WORKDIR /app/RAG-GEN

EXPOSE 8000

CMD ["uvicorn", "RAG_GEN:app", "--host", "0.0.0.0", "--port", "8000"]
