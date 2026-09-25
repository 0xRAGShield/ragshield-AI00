# RAGShield

RAGShield is a FastAPI-based retrieval-augmented generation (RAG) service. It embeds queries with **BGE-M3**, searches a **Qdrant** vector store, re-ranks results with a **Cross-Encoder**, and generates answers with a locally hosted **Qwen3-VL-8B-Instruct** model.

## Workflow

```
POST /query
  → embed query (BGE-M3)
  → retrieve from Qdrant (collection: ragshield_test)
  → re-rank (cross-encoder/ms-marco-MiniLM-L-6-v2)
  → build context + prompt
  → generate answer (Qwen3-VL)
```

**Ingestion (offline):** documents under `data/` are parsed, chunked, embedded, and upserted into Qdrant via `RAG-GEN/app/ingestion/ingest.py`.

**Query (online):** `RAG-GEN/RAG_GEN.py` exposes `GET /health` and `POST /query`.

## Project layout

| Path                                 | Purpose                                                       |
| ------------------------------------ | ------------------------------------------------------------- |
| `RAG-GEN/RAG_GEN.py`               | FastAPI entry point                                           |
| `RAG-GEN/app/core/settings.py`     | Application configuration (env overrides)                     |
| `RAG-GEN/app/pipeline/pipeline.py` | End-to-end RAG orchestration                                  |
| `RAG-GEN/app/embeddings/`          | BGE-M3 embedding wrapper                                      |
| `RAG-GEN/app/vector_store/`        | Qdrant client adapter                                         |
| `RAG-GEN/app/retrieval/`           | Vector search + cross-encoder re-ranking                      |
| `RAG-GEN/app/context/`             | Token-budget context assembly                                 |
| `RAG-GEN/app/generation/`          | Qwen3-VL runtime and prompt builder                           |
| `RAG-GEN/app/ingestion/`           | Document loaders, chunking, indexing                          |
| `RAG-GEN/app/arabic/`              | Arabic / Egyptian text normalization                          |
| `RAG-GEN/models/`                  | Local model weights (gitignored)                              |
| `data/documents/`                  | Source documents for ingestion (PDF, DOCX, TXT)               |
| `data/images/`                     | Image assets                                                  |
| `qdrant_backup/`                   | Qdrant collection snapshot for first-time restore             |
| `qdrant-init.py`                   | Docker sidecar: waits for Qdrant, restores snapshot if needed |
| `Dockerfile`                       | CUDA-enabled application image                                |
| `docker-compose.yml`               | Qdrant + init + RAGShield services                            |
| `requirements.txt`                 | Python dependencies                                           |
| `tests/`                           | Unit and integration tests                                    |

## Prerequisites

### Software

- Python 3.11+
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose (for containerized runs)
- CUDA-capable GPU recommended for Qwen3-VL (4-bit default; 8-bit supported)
- **Qdrant** (included in Docker Compose, or run separately on port `6333`)

### Model weights (required)

Download and place weights locally before building or running. Models are **not** fetched at runtime in Docker (`HF_HUB_OFFLINE=1`).

| Model      | Hugging Face ID                          | Local path                                               |
| ---------- | ---------------------------------------- | -------------------------------------------------------- |
| Embeddings | `BAAI/bge-m3`                          | `RAG-GEN/models/bge-m3/`                               |
| Re-ranker  | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `RAG-GEN/models/cross-encoder-ms-marco-MiniLM-L-6-v2/` |
| LLM        | `Qwen/Qwen3-VL-8B-Instruct`            | `RAG-GEN/models/qwen3-vl-8b/`                          |

### Data & snapshots

| File                                               | Purpose                                                     |
| -------------------------------------------------- | ----------------------------------------------------------- |
| `qdrant_backup/ragshield_test.snapshot`          | Pre-built Qdrant collection (restored on first Docker boot) |
| `qdrant_backup/ragshield_test.snapshot.checksum` | Optional integrity check for snapshot restore               |
| `data/documents/`, `data/images/`              | Sample corpus for the ingestion pipeline                    |

### Python dependencies

```bash
pip install -r requirements.txt
```

Key packages: `fastapi`, `uvicorn`, `qdrant-client`, `sentence-transformers`, `torch`, `transformers`, `bitsandbytes`, `accelerate`, `pydantic`, `pypdf`, `python-docx`, `Pillow`.

> The Docker image installs `transformers>=4.57.0` (required for `Qwen3VLForConditionalGeneration`) while keeping the CUDA build of `torch` from the base image.

## Environment variables

`RAG-GEN/app/core/settings.py` reads these overrides (see `RAG-GEN/.env.example` for a fuller template):

| Variable                      | Default                       | Description                                        |
| ----------------------------- | ----------------------------- | -------------------------------------------------- |
| `RETRIEVAL_QDRANT_HOST`     | `localhost`                 | Qdrant hostname (`qdrant` in Docker Compose)     |
| `RETRIEVAL_QDRANT_PORT`     | `6333`                      | Qdrant HTTP port                                   |
| `RETRIEVAL_COLLECTION_NAME` | `ragshield_test`            | Qdrant collection name                             |
| `LLM_MODEL_PATH`            | `models/qwen3-vl-8b`        | Path to Qwen3-VL weights (relative to`RAG-GEN/`) |
| `LLM_MODEL_ID`              | `Qwen/Qwen3-VL-8B-Instruct` | Model identifier                                   |
| `LLM_DEVICE`                | `auto`                      | `auto`, `cuda`, or `cpu`                     |
| `LLM_QUANTIZATION`          | `4bit`                      | `4bit`, `8bit`, or `none`                    |

**External service:** Qdrant must be reachable at the configured host/port.

## Local setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place model weights

Ensure the three model directories exist under `RAG-GEN/models/` (see table above).

### 3. Start Qdrant and restore the collection

```bash
docker compose up qdrant qdrant-init
```

`qdrant-init.py` connects to `http://qdrant:6333` and restores `ragshield_test` from the snapshot only when the collection does not already exist.

### 4. Start the API

```bash
cd RAG-GEN
python RAG_GEN.py
```

API: `http://127.0.0.1:8000`

### 5. (Optional) Ingest documents

With Qdrant running on `localhost:6333`:

```bash
cd RAG-GEN
python app/ingestion/ingest.py
```

Reads from `data/` at the repository root and indexes into the `ragshield_test` collection.

## Run with Docker

### 1. Pull the RAGShield image

```powershell
docker pull manarabdelbaky16/ragshield-ai00:latest
```

### 2. Required project files

Clone or copy the repository so these files are present:

- `docker-compose.yml`
- `Dockerfile`
- `RAG-GEN/qdrant_backup`
- `qdrant_init.py `

### 3. Start the project

```powershell
docker compose up -d
```

The RAGShield Docker image already contains all required model weights (BGE-M3, Cross-Encoder, and Qwen3-VL-8B-Instruct). Teammates do **not** need to download models separately.

Docker Compose starts the **RAGShield API** and **Qdrant** services and runs Qdrant initialization/restore on first boot.

Qdrant data is stored in the persistent Docker volume:

```text
ragshield_qdrant_storage
```

API: `http://localhost:8000`

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /query`

```json
{
  "query": "What are common diabetes symptoms?",
  "top_k": 8,
  "top_n": 5,
  "max_new_tokens": 512
}
```

Response fields: `answer`, `trace_id`, `degraded`, `retrieved_candidates`, `reranked_candidates`, `evidence_items`.

## Tests

```bash
pytest tests/ -v
```

Integration test against live models and Qdrant:

```bash
RUN_REAL_RAG=1 pytest tests/integration/ -v
```
