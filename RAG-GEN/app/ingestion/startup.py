from __future__ import annotations

import logging

from app.core.settings import AppSettings
from app.embeddings.embedding_model import BGE_M3_Embedding
from app.ingestion.chunker import ParentChildChunker
from app.ingestion.indexer import Indexer
from app.ingestion.loaders import FileParser
from app.ingestion.pipeline import IngestionPipeline, IngestionResult
from app.vector_store.vector_store import QdrantStore


logger = logging.getLogger("ragshield.ingestion")


def run_startup_ingestion(
    *,
    embedder: BGE_M3_Embedding,
    store: QdrantStore,
    settings: AppSettings,
) -> IngestionResult | None:
    """
    Scan DATA_DIR and index only new or modified files.

    Failures are logged and do not prevent API startup so the
    existing corpus remains queryable.
    """

    if not settings.ingest_on_startup:
        logger.info("Startup ingestion is disabled.")
        return None

    pipeline = IngestionPipeline(
        parser=FileParser(),
        chunker=ParentChildChunker(),
        indexer=Indexer(
            embedder=embedder,
            store=store,
        ),
        state_dir=settings.ingest_state_dir,
    )

    try:
        result = pipeline.run(settings.data_dir)

    except Exception:
        logger.exception(
            "Incremental ingestion failed during API startup."
        )
        return None

    logger.info(
        "Startup ingestion summary",
        extra={
            "files_discovered": result.files_discovered,
            "files_new": result.files_new,
            "files_modified": result.files_modified,
            "files_unchanged": result.files_unchanged,
            "files_unverified": result.files_unverified,
            "files_indexed": result.files_indexed,
            "files_failed": result.files_failed,
            "chunks_indexed": result.chunks_indexed,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
        },
    )

    return result
