from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.settings import get_settings
from app.ingestion.chunker import ParentChildChunker
from app.ingestion.incremental import IncrementalIngestor
from app.ingestion.indexer import Indexer
from app.ingestion.loaders import FileParser


@dataclass(frozen=True)
class IngestionResult:
    files_processed: int
    documents_cleaned: int
    chunks_created: int
    chunks_indexed: int
    files_discovered: int = 0
    files_new: int = 0
    files_modified: int = 0
    files_unchanged: int = 0
    files_unverified: int = 0
    files_failed: int = 0
    files_indexed: int = 0
    skipped: bool = False
    skip_reason: str | None = None


class IngestionPipeline:

    def __init__(
        self,
        *,
        parser: FileParser,
        chunker: ParentChildChunker,
        indexer: Indexer,
        state_dir: str | Path | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._indexer = indexer
        self._state_dir = state_dir

    def run(self, data_path: str | Path) -> IngestionResult:
        state_dir = self._state_dir

        if state_dir is None:
            state_dir = get_settings().ingest_state_dir

        result = IncrementalIngestor(
            parser=self._parser,
            chunker=self._chunker,
            indexer=self._indexer,
            store=self._indexer.store,
            state_dir=state_dir,
        ).run(data_path)

        return IngestionResult(
            files_processed=result.files_processed,
            documents_cleaned=result.documents_cleaned,
            chunks_created=result.chunks_created,
            chunks_indexed=result.chunks_indexed,
            files_discovered=result.files_discovered,
            files_new=result.files_new,
            files_modified=result.files_modified,
            files_unchanged=result.files_unchanged,
            files_unverified=result.files_unverified,
            files_failed=result.files_failed,
            files_indexed=result.files_indexed,
            skipped=result.skipped,
            skip_reason=result.skip_reason,
        )
