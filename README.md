
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

**For local Python runs:** download and place the model weights under `RAG-GEN/models/`.

**For Docker runs:** model weights are **not stored in the Git repository**. The Dockerfile automatically downloads all three models during the first Docker image build.

No manual model download is required for Docker.

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

### 1. Clone the Repository

```powershell
git clone https://github.com/0xRAGShield/ragshield-AI00.git
cd ragshield-AI00
```

Switch to the required branch if necessary:


```powershell
git checkout basic
```


### 2. Required project files

Clone or copy the repository so these files are present:

- `docker-compose.yml`
- `Dockerfile`
- `RAG-GEN/qdrant_backup`
- `qdrant-init.py `

### 3. Start the project

```powershell
docker compose up --build
```


The first build will:

1. Install the application dependencies.
2. Build the CUDA-enabled RAGShield image.
3. Download BGE-M3.
4. Download the Cross-Encoder.
5. Download Qwen3-VL-8B-Instruct.
6. Start Qdrant.
7. Restore the Qdrant snapshot if required.
8. Start the RAGShield API.

No Docker Hub image is required.

No model weights need to be downloaded manually.

Qdrant data is stored in the persistent Docker volume:

```text
ragshield_qdrant_storage
```

API: `http://localhost:8000`

After the first successful build, the project can be started with:


```powershell
docker compose up -d
```


### 4. Check the running containers


```powershell
docker ps
```


The main running containers are:

ragshield
ragshield-qdrant

The ragshield-qdrant-init container may appear as Exited after successfully completing the initialization/restore process.

### 5. Check RAGShield startup

Qwen3-VL-8B may take some time to load during startup.



```powershell
docker logs ragshield --tail 30
```

Wait until the application reports:

Application startup complete.

Then verify the API:


```powershell
Invoke-RestMethod http://localhost:8000/health
```

### 6. Stop the project



```powershell
docker compose down
```

The persistent Qdrant volume remains available for the next startup.


To remove the containers **and** the persistent Qdrant volume:


```powershell
docker compose down -v
```


<pre class="overflow-visible! px-0!" data-start="8355" data-end="8395"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div></div></div></div></div></div></pre>

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
