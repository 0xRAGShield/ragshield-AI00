
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.models.evidence_models import SourceDocument
from app.ingestion.loaders import FileParser
from app.ingestion.cleaner import clean_document
from app.ingestion.chunker import ParentChildChunker
from app.ingestion.indexer import Indexer


@dataclass(frozen=True)
class IngestionResult:
    files_processed: int
    documents_cleaned: int
    chunks_created: int
    chunks_indexed: int


class IngestionPipeline:

    def __init__(
        self,
        *,
        parser: FileParser,
        chunker: ParentChildChunker,
        indexer: Indexer,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._indexer = indexer

    def run(self, data_path: str | Path) -> IngestionResult:
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Data path does not exist: {path}"
            )

        if not path.is_dir():
            raise ValueError(
                f"Data path must be a directory: {path}"
            )

        documents = self._parser.load_directory(path)

        if not documents:
            return IngestionResult(
                files_processed=0,
                documents_cleaned=0,
                chunks_created=0,
                chunks_indexed=0,
            )

        source_documents = [
            document
            for document in documents
            if isinstance(document, SourceDocument)
        ]

        cleaned_documents = [
            clean_document(document)
            for document in source_documents
        ]

        chunks = self._chunker.chunk_documents(
            cleaned_documents
        )

        indexed_chunks = self._indexer.index(chunks)

        return IngestionResult(
            files_processed=len(documents),
            documents_cleaned=len(cleaned_documents),
            chunks_created=len(chunks),
            chunks_indexed=indexed_chunks,
        )

