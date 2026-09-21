from app.ingestion.loaders import FileParser
from app.ingestion.chunker import ParentChildChunker
from app.ingestion.indexer import Indexer
from app.embeddings.embedding_model import BGE_M3_Embedding
from app.vector_store.vector_store import QdrantStore
from app.ingestion.pipeline import IngestionPipeline
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data"


embedder = BGE_M3_Embedding()

store = QdrantStore(
    collection_name="ragshield_test"
)

indexer = Indexer(
    embedder=embedder,
    store=store,
)

parser = FileParser()
chunker = ParentChildChunker()

pipeline = IngestionPipeline(
    parser=parser,
    chunker=chunker,
    indexer=indexer,
)

result = pipeline.run(DATA_PATH)

print(f"Files processed: {result.files_processed}")
print(f"Documents cleaned: {result.documents_cleaned}")
print(f"Chunks created: {result.chunks_created}")
print(f"Chunks indexed: {result.chunks_indexed}")