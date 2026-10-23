from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.ingestion.chunker import ParentChildChunker
from app.ingestion.cleaner import clean_document
from app.ingestion.indexer import Indexer
from app.ingestion.loaders import FileParser
from app.models.evidence_models import (
    EvidenceChunk,
    ImageDocument,
    SourceDocument,
)
from app.models.image_refs import source_document_from_image
from app.vector_store.vector_store import QdrantStore


logger = logging.getLogger("ragshield.ingestion")

FileStatus = Literal["new", "modified", "unchanged", "unverified"]

_MANIFEST_VERSION = 1
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class IncrementalIngestionResult:
    files_discovered: int = 0
    files_new: int = 0
    files_modified: int = 0
    files_unchanged: int = 0
    files_unverified: int = 0
    files_failed: int = 0
    files_indexed: int = 0
    files_processed: int = 0
    documents_cleaned: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class _FileRecord:
    path: Path
    relative_path: str
    source_id: str
    content_sha256: str


@dataclass
class _CorpusCatalog:
    hashes_by_source: dict[str, set[str]] = field(default_factory=dict)
    relative_paths_by_source: dict[str, set[str]] = field(default_factory=dict)
    source_ids_by_relative_path: dict[str, set[str]] = field(
        default_factory=dict
    )
    hashes_by_relative_path: dict[str, set[str]] = field(
        default_factory=dict
    )
    source_ids_by_hash: dict[str, set[str]] = field(default_factory=dict)
    unhashed_source_ids: set[str] = field(default_factory=set)
    point_count: int = 0


class IncrementalIngestor:
    """
    Scan DATA_DIR and index only new or modified supported files.

    Reuses parse → clean → chunk → embed → upsert. Does not delete
    vectors for files that are missing from disk.
    """

    def __init__(
        self,
        *,
        parser: FileParser,
        chunker: ParentChildChunker,
        indexer: Indexer,
        store: QdrantStore,
        state_dir: str | Path,
        refresh_image_descriptions: bool = False,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._indexer = indexer
        self._store = store
        self._state_dir = Path(state_dir)
        self._refresh_image_descriptions = refresh_image_descriptions
        self._manifest_path = (
            self._state_dir
            / f"{store.collection_name}.manifest.json"
        )
        self._lock_path = (
            self._state_dir
            / f"{store.collection_name}.ingest.lock"
        )

    def run(
        self,
        data_path: str | Path,
    ) -> IncrementalIngestionResult:
        data_dir = Path(data_path)

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data path does not exist: {data_dir}"
            )

        if not data_dir.is_dir():
            raise ValueError(
                f"Data path must be a directory: {data_dir}"
            )

        thread_lock = _thread_lock_for(self._lock_path)

        if not thread_lock.acquire(blocking=False):
            logger.warning(
                "Skipping incremental ingestion; another job is running."
            )
            return IncrementalIngestionResult(
                skipped=True,
                skip_reason="in_process_lock",
            )

        lock_handle = None

        try:
            acquired, lock_handle = _acquire_file_lock(self._lock_path)

            if not acquired:
                logger.warning(
                    "Skipping incremental ingestion; lock file is held."
                )
                return IncrementalIngestionResult(
                    skipped=True,
                    skip_reason="file_lock",
                )

            return self._run_locked(data_dir)

        finally:
            _release_file_lock(lock_handle)
            thread_lock.release()

    def _run_locked(
        self,
        data_dir: Path,
    ) -> IncrementalIngestionResult:
        files = self._parser.list_supported_files(data_dir)
        manifest = _load_manifest(self._manifest_path)
        catalog = self._load_corpus_catalog()

        records: list[_FileRecord] = []
        hash_failures = 0

        for file_path in files:
            try:
                records.append(
                    self._inspect_file(file_path, data_dir)
                )
            except (OSError, ValueError) as error:
                hash_failures += 1
                logger.warning(
                    "Failed to fingerprint ingestion file",
                    extra={
                        "relative_path": _relative_path(
                            file_path,
                            data_dir,
                        ),
                        "error_type": type(error).__name__,
                    },
                )

        classified: dict[FileStatus, list[_FileRecord]] = {
            "new": [],
            "modified": [],
            "unchanged": [],
            "unverified": [],
        }

        for record in records:
            status = _classify_file(
                record,
                manifest=manifest,
                catalog=catalog,
            )
            classified[status].append(record)

        if self._refresh_image_descriptions:
            kept_unchanged: list[_FileRecord] = []
            refreshed = 0

            for record in classified["unchanged"]:
                if (
                    record.path.suffix.lower()
                    in self._parser.IMAGE_EXTENSIONS
                ):
                    classified["modified"].append(record)
                    refreshed += 1
                else:
                    kept_unchanged.append(record)

            classified["unchanged"] = kept_unchanged

            if refreshed:
                logger.info(
                    "Queued unchanged image files for description refresh",
                    extra={"images_queued": refreshed},
                )

        logger.info(
            "Incremental ingestion scan complete",
            extra={
                "files_discovered": len(files),
                "files_new": len(classified["new"]),
                "files_modified": len(classified["modified"]),
                "files_unchanged": len(classified["unchanged"]),
                "files_unverified": len(classified["unverified"]),
                "fingerprint_failures": hash_failures,
                "collection": self._store.collection_name,
            },
        )

        if classified["unverified"]:
            logger.warning(
                "Skipping files that cannot be matched to existing "
                "vectors without sufficient provenance; they were "
                "not treated as new.",
                extra={
                    "files_unverified": len(classified["unverified"]),
                },
            )

        pending = classified["new"] + classified["modified"]

        indexed = 0
        failed = hash_failures
        documents_cleaned = 0
        chunks_created = 0
        chunks_indexed = 0

        if not pending:
            logger.info(
                "No new or modified files; skipping embedding."
            )
            _bootstrap_unchanged_manifest(
                manifest,
                classified["unchanged"],
            )
            _save_manifest(self._manifest_path, manifest)
            return IncrementalIngestionResult(
                files_discovered=len(files),
                files_new=len(classified["new"]),
                files_modified=len(classified["modified"]),
                files_unchanged=len(classified["unchanged"]),
                files_unverified=len(classified["unverified"]),
                files_failed=failed,
                files_indexed=0,
                files_processed=0,
            )

        for record in pending:
            try:
                cleaned, chunks = self._index_record(
                    record,
                    data_dir=data_dir,
                    manifest=manifest,
                    catalog=catalog,
                )
                documents_cleaned += cleaned
                chunks_created += chunks
                chunks_indexed += chunks
                indexed += 1

            except Exception:
                failed += 1
                logger.exception(
                    "Failed to ingest file",
                    extra={
                        "relative_path": record.relative_path,
                    },
                )

        _bootstrap_unchanged_manifest(
            manifest,
            classified["unchanged"],
        )
        _save_manifest(self._manifest_path, manifest)

        logger.info(
            "Incremental ingestion finished",
            extra={
                "files_indexed": indexed,
                "files_failed": failed,
                "chunks_indexed": chunks_indexed,
                "embeddings_generated": chunks_indexed,
            },
        )

        return IncrementalIngestionResult(
            files_discovered=len(files),
            files_new=len(classified["new"]),
            files_modified=len(classified["modified"]),
            files_unchanged=len(classified["unchanged"]),
            files_unverified=len(classified["unverified"]),
            files_failed=failed,
            files_indexed=indexed,
            files_processed=indexed,
            documents_cleaned=documents_cleaned,
            chunks_created=chunks_created,
            chunks_indexed=chunks_indexed,
        )

    def _inspect_file(
        self,
        file_path: Path,
        data_dir: Path,
    ) -> _FileRecord:
        return _FileRecord(
            path=file_path,
            relative_path=_relative_path(file_path, data_dir),
            source_id=self._parser.source_id_for(file_path),
            content_sha256=self._parser.content_sha256(file_path),
        )

    def _load_corpus_catalog(self) -> _CorpusCatalog:
        catalog = _CorpusCatalog()

        payloads = self._store.iter_payloads()
        catalog.point_count = len(payloads)

        for payload in payloads:
            source_id = _optional_str(payload.get("source_id"))

            if not source_id:
                continue

            metadata = payload.get("metadata") or {}

            if not isinstance(metadata, dict):
                metadata = {}

            digest = _optional_str(metadata.get("content_sha256"))
            relative_path = _optional_str(metadata.get("relative_path"))

            catalog.hashes_by_source.setdefault(source_id, set())

            if digest:
                catalog.hashes_by_source[source_id].add(digest)
                catalog.source_ids_by_hash.setdefault(digest, set()).add(
                    source_id
                )
            else:
                catalog.unhashed_source_ids.add(source_id)

            if relative_path:
                catalog.relative_paths_by_source.setdefault(
                    source_id,
                    set(),
                ).add(relative_path)
                catalog.source_ids_by_relative_path.setdefault(
                    relative_path,
                    set(),
                ).add(source_id)

                if digest:
                    catalog.hashes_by_relative_path.setdefault(
                        relative_path,
                        set(),
                    ).add(digest)

        return catalog

    def _index_record(
        self,
        record: _FileRecord,
        *,
        data_dir: Path,
        manifest: dict[str, Any],
        catalog: _CorpusCatalog,
    ) -> tuple[int, int]:
        document = self._parser.load_file(record.path)

        if isinstance(document, ImageDocument):
            source_document = source_document_from_image(
                document,
                data_dir=data_dir,
            )
        elif isinstance(document, SourceDocument):
            source_document = document
        else:
            raise TypeError(
                "Unsupported ingestion document type."
            )

        metadata = dict(source_document.metadata or {})
        metadata["relative_path"] = record.relative_path
        metadata["content_sha256"] = record.content_sha256
        source_document = source_document.model_copy(
            update={"metadata": metadata}
        )

        cleaned = clean_document(source_document)
        chunks = self._chunker.chunk_document(cleaned)
        vectors = self._indexer.embed(chunks)

        self._replace_source_vectors(
            chunks=chunks,
            vectors=vectors,
            stale_source_ids=_stale_source_ids(
                record,
                manifest=manifest,
                catalog=catalog,
            ),
        )
        indexed = len(chunks)

        self._commit_manifest_entry(
            manifest,
            record=record,
            chunk_count=indexed,
        )

        return (1, indexed)

    def _replace_source_vectors(
        self,
        *,
        chunks: list[EvidenceChunk],
        vectors: list,
        stale_source_ids: list[str],
    ) -> None:
        """
        Store replacement vectors before removing obsolete chunks.

        Overlapping chunk IDs are overwritten in place. Chunks that
        no longer exist are deleted only after upsert succeeds.
        """

        new_chunk_ids = {
            chunk.chunk_id
            for chunk in chunks
        }
        stale_chunk_ids: list[str] = []

        for source_id in dict.fromkeys(stale_source_ids):
            for chunk_id in self._store.chunk_ids_for_source(source_id):
                if chunk_id not in new_chunk_ids:
                    stale_chunk_ids.append(chunk_id)

        self._store.upsert(
            chunks=chunks,
            vectors=vectors,
        )

        if stale_chunk_ids:
            self._store.delete_by_chunk_ids(stale_chunk_ids)

    def _commit_manifest_entry(
        self,
        manifest: dict[str, Any],
        *,
        record: _FileRecord,
        chunk_count: int,
    ) -> None:
        files = manifest.setdefault("files", {})
        previous = files.get(record.relative_path)
        files[record.relative_path] = {
            "relative_path": record.relative_path,
            "content_sha256": record.content_sha256,
            "source_id": record.source_id,
            "status": "indexed",
            "indexed_at": _utc_now(),
            "chunk_count": chunk_count,
        }

        try:
            _save_manifest(self._manifest_path, manifest)
        except Exception:
            if previous is None:
                files.pop(record.relative_path, None)
            else:
                files[record.relative_path] = previous

            raise


def _classify_file(
    record: _FileRecord,
    *,
    manifest: dict[str, Any],
    catalog: _CorpusCatalog,
) -> FileStatus:
    files = manifest.get("files") or {}
    entry = files.get(record.relative_path) or {}
    entry_status = _optional_str(entry.get("status"))
    entry_hash = _optional_str(entry.get("content_sha256"))

    if entry_status == "indexed" and entry_hash == record.content_sha256:
        return "unchanged"

    if entry_status == "indexed" and entry_hash != record.content_sha256:
        return "modified"

    relative_hashes = catalog.hashes_by_relative_path.get(
        record.relative_path
    )

    if relative_hashes:
        if record.content_sha256 in relative_hashes:
            return "unchanged"

        return "modified"

    if record.relative_path in catalog.source_ids_by_relative_path:
        return "unchanged"

    source_hashes = catalog.hashes_by_source.get(record.source_id)

    if source_hashes is not None:
        if (
            record.content_sha256 in source_hashes
            or record.source_id in catalog.unhashed_source_ids
        ):
            return "unchanged"

        if source_hashes:
            return "modified"

    if record.content_sha256 in catalog.source_ids_by_hash:
        return "unchanged"

    if (
        catalog.point_count > 0
        and catalog.unhashed_source_ids
        and record.source_id not in catalog.hashes_by_source
        and record.source_id not in catalog.unhashed_source_ids
        and record.relative_path not in catalog.source_ids_by_relative_path
    ):
        return "unverified"

    return "new"


def _stale_source_ids(
    record: _FileRecord,
    *,
    manifest: dict[str, Any],
    catalog: _CorpusCatalog,
) -> list[str]:
    stale = [record.source_id]
    previous = (manifest.get("files") or {}).get(record.relative_path) or {}
    previous_source = _optional_str(previous.get("source_id"))

    if previous_source:
        stale.append(previous_source)

    stale.extend(
        catalog.source_ids_by_relative_path.get(record.relative_path, ())
    )

    return list(dict.fromkeys(stale))


def _bootstrap_unchanged_manifest(
    manifest: dict[str, Any],
    unchanged: list[_FileRecord],
) -> None:
    files = manifest.setdefault("files", {})

    for record in unchanged:
        current = files.get(record.relative_path) or {}

        if (
            current.get("status") == "indexed"
            and current.get("content_sha256") == record.content_sha256
        ):
            continue

        files[record.relative_path] = {
            "relative_path": record.relative_path,
            "content_sha256": record.content_sha256,
            "source_id": record.source_id,
            "status": "indexed",
            "indexed_at": current.get("indexed_at") or _utc_now(),
            "chunk_count": current.get("chunk_count"),
            "bootstrapped": True,
        }


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": _MANIFEST_VERSION,
            "files": {},
        }

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning(
            "Ingestion manifest could not be read; starting from "
            "Qdrant provenance only.",
            extra={"error_type": type(error).__name__},
        )
        return {
            "version": _MANIFEST_VERSION,
            "files": {},
        }

    if not isinstance(loaded, dict):
        return {
            "version": _MANIFEST_VERSION,
            "files": {},
        }

    files = loaded.get("files")

    if not isinstance(files, dict):
        loaded["files"] = {}

    loaded["version"] = _MANIFEST_VERSION
    return loaded


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _relative_path(file_path: Path, data_dir: Path) -> str:
    try:
        return file_path.resolve().relative_to(
            data_dir.resolve()
        ).as_posix()
    except (OSError, ValueError):
        try:
            return file_path.absolute().relative_to(
                data_dir.absolute()
            ).as_posix()
        except ValueError:
            return file_path.as_posix()


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _thread_lock_for(lock_path: Path) -> threading.Lock:
    key = str(lock_path)

    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)

        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock

        return lock


def _acquire_file_lock(path: Path) -> tuple[bool, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")

    try:
        handle.seek(0)

        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
            handle.seek(0)

        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        handle.seek(0)
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
        return True, handle

    except OSError:
        handle.close()
        return False, None


def _release_file_lock(handle: Any) -> None:
    if handle is None:
        return

    try:
        handle.seek(0)

        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        try:
            handle.close()
        except OSError:
            pass
