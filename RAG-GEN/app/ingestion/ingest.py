from pathlib import Path

from app.core.settings import get_settings
from app.embeddings.embedding_model import BGE_M3_Embedding
from app.ingestion.chunker import ParentChildChunker
from app.ingestion.indexer import Indexer
from app.ingestion.loaders import FileParser
from app.ingestion.pipeline import IngestionPipeline
from app.vector_store.vector_store import QdrantStore


settings = get_settings()
DATA_PATH = Path(settings.data_dir)

embedder = BGE_M3_Embedding()

store = QdrantStore(
    collection_name=settings.retrieval.collection_name,
    mode=settings.retrieval.qdrant_mode,
    host=settings.retrieval.qdrant_host,
    port=settings.retrieval.qdrant_port,
    local_path=settings.retrieval.qdrant_local_path,
    vector_size=embedder.dimension,
)

indexer = Indexer(
    embedder=embedder,
    store=store,
)

pipeline = IngestionPipeline(
    parser=FileParser(),
    chunker=ParentChildChunker(),
    indexer=indexer,
    state_dir=settings.ingest_state_dir,
)

result = pipeline.run(DATA_PATH)

print(f"Files discovered: {result.files_discovered}")
print(f"New files: {result.files_new}")
print(f"Modified files: {result.files_modified}")
print(f"Unchanged files: {result.files_unchanged}")
print(f"Unverified files skipped: {result.files_unverified}")
print(f"Files processed: {result.files_processed}")
print(f"Files failed: {result.files_failed}")
print(f"Documents cleaned: {result.documents_cleaned}")
print(f"Chunks created: {result.chunks_created}")
print(f"Chunks indexed: {result.chunks_indexed}")
