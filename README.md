
# RAGShield

RAGShield is a FastAPI-based retrieval-augmented generation (RAG) service for healthcare documents and medical images. It embeds queries with **BGE-M3**, searches a **Qdrant** vector store, re-ranks results with a **Cross-Encoder**, and generates answers with a locally hosted **Qwen3-VL-8B-Instruct** model.

Image files are indexed from **filename, path, and explicit metadata** (not from pixels). The API can return clickable media URLs for retrieved images. There is **no OCR** and **no visual embedding** of image contents.

## Workflow

```
POST /query
  → embed query (BGE-M3)
  → retrieve from Qdrant (collection: ragshield_test)
  → re-rank (cross-encoder/ms-marco-MiniLM-L-6-v2)
  → collect image references from reranked evidence
  → build context + prompt
  → generate answer (Qwen3-VL)
  → if generation fails, return degraded=true with retrieved images
```

**Ingestion:** files under `DATA_DIR` (default `data/`) are discovered incrementally at API startup and by `RAG-GEN/app/ingestion/ingest.py`. Only new or modified files are parsed, chunked, embedded, and upserted. Unchanged files are skipped. Missing files are not deleted from Qdrant.

**Query:** `RAG-GEN/RAG_GEN.py` exposes `GET /health`, `POST /query`, and `GET /media/{path}`.

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
| `LLM_MODEL_PATH`            | `models/qwen3-vl-8b`        | Path to Qwen3-VL weights (relative to `RAG-GEN/`) |
| `LLM_MODEL_ID`              | `Qwen/Qwen3-VL-8B-Instruct` | Model identifier                                   |
| `LLM_DEVICE`                | `auto`                      | `auto`, `cuda`, or `cpu`                     |
| `LLM_QUANTIZATION`          | `4bit`                      | `4bit`, `8bit`, or `none`                    |
| `DATA_DIR`                  | repo `data/`                | Corpus root (`documents/` and `images/`)         |
| `PUBLIC_BASE_URL`           | _(empty)_                   | Public origin used to build clickable image URLs (`http://` or `https://` only). Do not trust client `Host` headers. |
| `INGEST_ON_STARTUP`         | `true`                      | Scan `DATA_DIR` and ingest new/changed files when the API starts |
| `INGEST_STATE_DIR`          | `RAG-GEN/ingestion_state/`  | Manifest and ingest lock files                     |
| `INGEST_REFRESH_IMAGES`     | `false`                     | When running `ingest.py`, re-index existing image files so new `image_description` text is embedded. **Not** applied on API startup. |

**External service:** Qdrant must be reachable at the configured host/port.

Set `PUBLIC_BASE_URL` to the URL that **clients** use to reach the API, for example:

- Local process: `http://127.0.0.1:8000`
- Docker published port: `http://localhost:8000`
- Colab or a tunnel: the **public proxy URL**, not the VM loopback address
- Reverse proxy: `https://your.example.com`

`127.0.0.1` is only reachable on the machine that runs the server. It is not a public internet URL.

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


## Image ingestion and semantic retrieval

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`.

Each image is usually one child chunk when the description fits in `child_chunk_size` (400 characters). The embedding is built from that `image_description` string, which already includes the filename plus type aliases. The description is derived from **filename tokens, path segments, and explicit metadata** (`image_description`, `description`, `caption`, `alt_text`, `patient_name` if already present).

The pipeline does **not** read pixels, run OCR, or generate visual embeddings. A name such as `Maram` is taken from a filename like `maram_abdelbaky_prescription.jpeg` and is treated as a **filename label**, not a verified patient identity.

Example stored description (shape, not a guaranteed exact string):

```text
Prescription image labeled Maram Abdelbaky (from filename, not verified identity).
Document type aliases: prescription, medical prescription, روشتة.
Searchable aliases: Maram Abdelbaky prescription, Maram prescription, prescription for Maram Abdelbaky, prescription for Maram, روشتة Maram Abdelbaky, روشتة Maram.
Filename: maram_abdelbaky_prescription.jpeg.
```

That text is what BGE-M3 embeds and the reranker scores. Queries that share those tokens can omit the raw filename:

- `Show me Maram's prescription` (Latin name + English type)
- `عايزه روشتة Maram` (Arabic type alias + Latin filename token)

Queries that use an Arabic **name** not present in the filename, such as `عايزة الروشتة بتاعت مرام` or `عايزة صورة الروشتة الخاصة بمرام`, share only `روشتة` with the indexed text. Matching `مرام` to `Maram` is not implemented.

Relevance still depends on retrieval and reranking. A generic query such as “روشتة” can match several prescription files; more specific tokens (a name label in the filename) rank the corresponding image higher. Unrelated images should not share a unique name token.

Qdrant payload metadata includes `file_name`, `file_type`, `image_path`, `image_url`, `image_description`, `relative_path`, `content_sha256`, `source_id`, and `chunk_id`. `source_id` / `chunk_id` conventions are unchanged.

### Updating existing Qdrant points

API startup **does not** rewrite descriptions for unchanged image files and **does not** migrate the live collection.

To embed the new descriptions for images that are already indexed:

```powershell
cd RAG-GEN
$env:INGEST_REFRESH_IMAGES="true"
python app/ingestion/ingest.py
```

```bash
cd RAG-GEN
INGEST_REFRESH_IMAGES=true python app/ingestion/ingest.py
```

That command re-processes image files only (text documents with unchanged hashes stay skipped). Alternatively, change an image file so its SHA-256 changes, then restart the API or run ingest without the refresh flag.

Until that refresh (or a content change) runs, older image points remain searchable only by previously indexed filename/path text.

Refresh re-queues only files whose suffix is an image extension **and** that were classified as unchanged. It does not re-embed unchanged PDFs/TXT files. It does not re-queue **unverified** files (legacy points that cannot be matched safely). Same-path image points keep the same `source_id`/`chunk_id`, so Qdrant upsert overwrites the existing point instead of inserting a duplicate.

## Media URLs

`GET /media/{path}` serves image files that resolve **inside** `DATA_DIR` and have a supported image extension. Path traversal and non-image files return 404.

`POST /query` returns image objects with:

| Field | Meaning |
| --- | --- |
| `url` | Clickable URL: `PUBLIC_BASE_URL` + `/media/...` when the base URL is set; otherwise the relative `/media/...` path |
| `relative_url` | Always `/media/...` when the file is under `DATA_DIR` |
| `absolute_url` | Fully qualified URL when `PUBLIC_BASE_URL` is set; otherwise `null` |
| `description` | Indexed `image_description` when present on the retrieved record |
| `media_type`, `source`, `document_id`, `chunk_id`, `source_id` | Existing provenance fields |

Filesystem paths are not returned in these URL fields.

## Degraded generation

If retrieval and reranking succeed but the LLM/VLM fails (for example CUDA out-of-memory), the API does **not** turn that into an HTTP 500. The response includes:

- `degraded: true`
- a fallback `answer` stating that generation failed
- any `images` collected from reranked evidence, with media URLs

Failures before generation (retrieval, rerank, context, prompt) still follow the existing HTTP 500 path.

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /query`

```json
{
  "query": "Show me Maram's prescription",
  "top_k": 8,
  "top_n": 5,
  "max_new_tokens": 512
}
```

```json
{
  "answer": "...",
  "trace_id": "...",
  "degraded": false,
  "retrieved_candidates": 5,
  "reranked_candidates": 3,
  "evidence_items": 3,
  "images": [
    {
      "url": "http://127.0.0.1:8000/media/images/maram_abdelbaky_prescription.jpeg",
      "relative_url": "/media/images/maram_abdelbaky_prescription.jpeg",
      "absolute_url": "http://127.0.0.1:8000/media/images/maram_abdelbaky_prescription.jpeg",
      "source": "maram_abdelbaky_prescription.jpeg",
      "media_type": "image/jpeg",
      "image_type": null,
      "document_id": "...",
      "source_id": "...",
      "chunk_id": "...",
      "description": "Prescription image labeled Maram Abdelbaky (from filename, not verified identity). ..."
    }
  ]
}
```

The `url` example assumes `PUBLIC_BASE_URL=http://127.0.0.1:8000`. Without that setting, `url` is the relative `/media/...` path and `absolute_url` is `null`.

### `GET /media/{path}`

Example: `GET http://127.0.0.1:8000/media/images/maram_abdelbaky_prescription.jpeg` returns the image bytes when that file exists under `DATA_DIR`.

## Tests

From the repository root (`ragshield-AI00`), with `PYTHONPATH` including `RAG-GEN`:

```bash
pytest tests/ -v
```

Incremental ingestion mocks (no live Qdrant or embedding model):

```bash
pytest tests/ingestions/test_incremental.py -v
```

Integration test against live models and Qdrant:

```bash
RUN_REAL_RAG=1 pytest tests/integration/ -v
```

## Limitations

- Image search quality depends on filenames, folders, and any explicit metadata. Opaque names (`img001.png`) will not match “Maram’s prescription”.
- Indexed aliases for `maram_abdelbaky_prescription.jpeg` include Latin tokens (`Maram`, `prescription`) and the Arabic **type** word `روشتة`. The Arabic **name** `مرام` is **not** written into the index unless it already appears in the filename or explicit metadata. A query such as `عايزة الروشتة بتاعت مرام` is therefore guaranteed only to share `روشتة` with the indexed text; matching `مرام` to `Maram` is not implemented as transliteration and must not be assumed.
- Existing Qdrant points do not gain `image_description` until image refresh or a content-changing re-ingest.
- `PUBLIC_BASE_URL` must be set by the operator. The API does not invent a public hostname from request headers.
- Google Colab: configure `PUBLIC_BASE_URL` to the URL Colab (or your tunnel) exposes to the browser. `http://127.0.0.1:8000` inside the Colab VM is not automatically reachable from your laptop.
- Incremental ingest does not remove Qdrant points when a source file is deleted from disk.
