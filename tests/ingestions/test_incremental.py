from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.settings import AppSettings
from app.ingestion.chunker import ChunkingLimits, ParentChildChunker
from app.ingestion.incremental import IncrementalIngestor
from app.ingestion import incremental as incremental_mod
from app.ingestion.indexer import Indexer
from app.ingestion.loaders import FileParser
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.startup import run_startup_ingestion


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding failed")

        self.calls.append(list(texts))
        return [
            [float(len(text)), 1.0]
            for text in texts
        ]


class FakeVectorStore:
    def __init__(self, collection_name: str = "ragshield_test") -> None:
        self.collection_name = collection_name
        self.points: dict[str, dict] = {}
        self.upsert_calls = 0
        self.delete_chunk_calls: list[list[str]] = []
        self.fail_upsert = False
        self.fail_delete_chunks = False

    def iter_payloads(self) -> list[dict]:
        return [
            dict(point["payload"])
            for point in self.points.values()
        ]

    def chunk_ids_for_source(self, source_id: str) -> list[str]:
        return [
            chunk_id
            for chunk_id, point in self.points.items()
            if point["payload"].get("source_id") == source_id
        ]

    def upsert(self, chunks, vectors) -> None:
        if self.fail_upsert:
            raise RuntimeError("upsert failed")

        self.upsert_calls += 1

        for chunk, vector in zip(chunks, vectors):
            self.points[chunk.chunk_id] = {
                "vector": list(vector),
                "payload": {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "text": chunk.text,
                    "metadata": dict(chunk.metadata or {}),
                },
            }

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        if self.fail_delete_chunks:
            raise RuntimeError("delete failed")

        self.delete_chunk_calls.append(list(chunk_ids))

        for chunk_id in chunk_ids:
            self.points.pop(chunk_id, None)

    def delete_by_source_ids(self, source_ids: list[str]) -> None:
        raise AssertionError(
            "delete_by_source_ids must not be used for replacement"
        )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_ingestor(
    tmp_path: Path,
    *,
    embedder: FakeEmbedder | None = None,
    store: FakeVectorStore | None = None,
    chunker: ParentChildChunker | None = None,
) -> tuple[IncrementalIngestor, FakeEmbedder, FakeVectorStore, Path, Path]:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()

    embedder = embedder or FakeEmbedder()
    store = store or FakeVectorStore()
    indexer = Indexer(embedder=embedder, store=store)
    ingestor = IncrementalIngestor(
        parser=FileParser(),
        chunker=chunker or ParentChildChunker(),
        indexer=indexer,
        store=store,
        state_dir=state_dir,
    )
    return ingestor, embedder, store, data_dir, state_dir


def _manifest(state_dir: Path, collection: str = "ragshield_test") -> dict:
    path = state_dir / f"{collection}.manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_new_file_is_parsed_chunked_embedded_and_upserted(tmp_path):
    ingestor, embedder, store, data_dir, state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "notes.txt", "Patient follow-up notes.")

    result = ingestor.run(data_dir)

    assert result.files_discovered == 1
    assert result.files_new == 1
    assert result.files_indexed == 1
    assert result.files_failed == 0
    assert embedder.calls
    assert "Patient follow-up notes." in embedder.calls[0][0]
    assert store.upsert_calls == 1
    assert len(store.points) == 1

    point = next(iter(store.points.values()))
    assert point["vector"]
    assert point["payload"]["source_id"]
    assert point["payload"]["chunk_id"].startswith(
        point["payload"]["source_id"]
    )
    assert point["payload"]["metadata"]["relative_path"] == "notes.txt"
    assert point["payload"]["metadata"]["content_sha256"]

    manifest = _manifest(state_dir)
    assert manifest["files"]["notes.txt"]["status"] == "indexed"
    assert manifest["files"]["notes.txt"]["content_sha256"]


def test_second_run_skips_unchanged_file_without_embedding(tmp_path):
    ingestor, embedder, store, data_dir, _state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "notes.txt", "Unchanged medical note.")

    first = ingestor.run(data_dir)
    second = ingestor.run(data_dir)

    assert first.files_indexed == 1
    assert second.files_unchanged == 1
    assert second.files_indexed == 0
    assert second.files_new == 0
    assert second.files_modified == 0
    assert len(embedder.calls) == 1
    assert store.upsert_calls == 1


def test_modified_file_replaces_only_that_source(tmp_path):
    chunker = ParentChildChunker(
        ChunkingLimits(
            parent_chunk_size=80,
            child_chunk_size=30,
        )
    )
    ingestor, embedder, store, data_dir, _state_dir = _build_ingestor(
        tmp_path,
        chunker=chunker,
    )
    _write_text(
        data_dir / "report.txt",
        "AAAA " * 20,
    )
    _write_text(
        data_dir / "other.txt",
        "Unrelated source stays the same.",
    )

    first = ingestor.run(data_dir)
    other_ids = [
        chunk_id
        for chunk_id, point in store.points.items()
        if point["payload"]["metadata"]["relative_path"] == "other.txt"
    ]
    report_ids_before = [
        chunk_id
        for chunk_id, point in store.points.items()
        if point["payload"]["metadata"]["relative_path"] == "report.txt"
    ]
    embed_calls_after_first = len(embedder.calls)

    _write_text(data_dir / "report.txt", "BB")
    second = ingestor.run(data_dir)

    assert first.files_indexed == 2
    assert second.files_modified == 1
    assert second.files_unchanged == 1
    assert len(embedder.calls) == embed_calls_after_first + 1
    assert store.upsert_calls == 3

    for chunk_id in other_ids:
        assert chunk_id in store.points

    report_ids_after = [
        chunk_id
        for chunk_id, point in store.points.items()
        if point["payload"]["metadata"]["relative_path"] == "report.txt"
    ]
    assert report_ids_after
    assert other_ids
    assert len(report_ids_after) <= len(report_ids_before)


def test_modified_file_drops_obsolete_chunks_after_upsert(tmp_path):
    chunker = ParentChildChunker(
        ChunkingLimits(
            parent_chunk_size=80,
            child_chunk_size=30,
        )
    )
    ingestor, _embedder, store, data_dir, _state_dir = _build_ingestor(
        tmp_path,
        chunker=chunker,
    )
    _write_text(data_dir / "report.txt", "AAAA " * 20)
    ingestor.run(data_dir)
    before = set(store.points)

    _write_text(data_dir / "report.txt", "BB")
    ingestor.run(data_dir)
    after = set(store.points)

    assert after
    assert after != before
    assert store.delete_chunk_calls
    leftover = before - after
    flattened = {
        chunk_id
        for call in store.delete_chunk_calls
        for chunk_id in call
    }
    assert leftover <= flattened


def test_source_and_chunk_ids_are_stable_across_retries(tmp_path):
    ingestor, _embedder, store, data_dir, _state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "notes.txt", "Stable identity text.")

    ingestor.run(data_dir)
    first_ids = dict(store.points)
    store.upsert_calls = 0

    (data_dir / "notes.txt").write_text(
        "Stable identity text.",
        encoding="utf-8",
    )
    ingestor.run(data_dir)

    assert store.points.keys() == first_ids.keys()
    assert store.upsert_calls == 0


def test_parse_failure_is_not_recorded_in_manifest(tmp_path):
    ingestor, embedder, store, data_dir, state_dir = _build_ingestor(tmp_path)
    (data_dir / "bad.txt").write_bytes(b"\xff\xfe\x00")

    result = ingestor.run(data_dir)

    assert result.files_failed == 1
    assert result.files_indexed == 0
    assert embedder.calls == []
    assert store.upsert_calls == 0
    assert not (state_dir / "ragshield_test.manifest.json").exists() or (
        "bad.txt" not in _manifest(state_dir).get("files", {})
    )


def test_embedding_failure_keeps_old_vectors_and_manifest(tmp_path):
    ingestor, embedder, store, data_dir, state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "notes.txt", "Original text.")
    ingestor.run(data_dir)
    original_points = dict(store.points)
    original_hash = _manifest(state_dir)["files"]["notes.txt"]["content_sha256"]

    _write_text(data_dir / "notes.txt", "Changed text.")
    embedder.fail = True
    result = ingestor.run(data_dir)

    assert result.files_failed == 1
    assert result.files_indexed == 0
    assert store.points == original_points
    assert _manifest(state_dir)["files"]["notes.txt"]["content_sha256"] == (
        original_hash
    )


def test_upsert_failure_does_not_delete_existing_vectors(tmp_path):
    ingestor, _embedder, store, data_dir, state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "notes.txt", "Original text.")
    ingestor.run(data_dir)
    original_points = dict(store.points)

    _write_text(data_dir / "notes.txt", "Changed text.")
    store.fail_upsert = True
    result = ingestor.run(data_dir)

    assert result.files_failed == 1
    assert store.points.keys() == original_points.keys()
    assert store.delete_chunk_calls == []
    assert _manifest(state_dir)["files"]["notes.txt"]["status"] == "indexed"
    assert "Changed text." not in next(iter(store.points.values()))["payload"]["text"]


def test_stale_delete_failure_is_not_marked_indexed_for_new_hash(tmp_path):
    chunker = ParentChildChunker(
        ChunkingLimits(
            parent_chunk_size=80,
            child_chunk_size=30,
        )
    )
    ingestor, _embedder, store, data_dir, state_dir = _build_ingestor(
        tmp_path,
        chunker=chunker,
    )
    _write_text(data_dir / "report.txt", "AAAA " * 20)
    ingestor.run(data_dir)
    original_hash = _manifest(state_dir)["files"]["report.txt"]["content_sha256"]

    _write_text(data_dir / "report.txt", "BB")
    store.fail_delete_chunks = True
    result = ingestor.run(data_dir)

    assert result.files_failed == 1
    assert store.upsert_calls >= 2
    assert _manifest(state_dir)["files"]["report.txt"]["content_sha256"] == (
        original_hash
    )


def test_manifest_write_failure_reverts_in_memory_entry(tmp_path, monkeypatch):
    ingestor, _embedder, store, data_dir, state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "notes.txt", "New file.")

    def fail_save(_path, _manifest):
        raise OSError("manifest write failed")

    monkeypatch.setattr(incremental_mod, "_save_manifest", fail_save)
    result = ingestor.run(data_dir)

    assert result.files_failed == 1
    assert store.upsert_calls == 1
    assert not (state_dir / "ragshield_test.manifest.json").exists()


def test_retry_after_upsert_failure_is_idempotent(tmp_path):
    ingestor, embedder, store, data_dir, _state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "notes.txt", "Retry me.")
    store.fail_upsert = True
    failed = ingestor.run(data_dir)
    store.fail_upsert = False
    success = ingestor.run(data_dir)

    assert failed.files_failed == 1
    assert success.files_indexed == 1
    assert len(store.points) == 1
    point = next(iter(store.points.values()))
    assert point["payload"]["chunk_id"] == (
        f"{point['payload']['source_id']}:chunk:000000"
    )
    assert len(embedder.calls) == 2


def test_legacy_hash_match_is_not_treated_as_new(tmp_path):
    ingestor, embedder, store, data_dir, _state_dir = _build_ingestor(tmp_path)
    path = data_dir / "legacy.txt"
    _write_text(path, "Legacy corpus text.")
    digest = FileParser().content_sha256(path)

    store.points["foreign:chunk:000000"] = {
        "vector": [1.0, 1.0],
        "payload": {
            "chunk_id": "foreign:chunk:000000",
            "source_id": "windows-path-source",
            "text": "Legacy corpus text.",
            "metadata": {"content_sha256": digest},
        },
    }

    result = ingestor.run(data_dir)

    assert result.files_unchanged == 1
    assert result.files_new == 0
    assert embedder.calls == []
    assert store.upsert_calls == 0


def test_unhashed_unmatched_legacy_points_are_unverified(tmp_path):
    ingestor, embedder, store, data_dir, _state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "maybe.txt", "Cannot verify this file.")
    store.points["unknown:chunk:000000"] = {
        "vector": [1.0, 1.0],
        "payload": {
            "chunk_id": "unknown:chunk:000000",
            "source_id": "old-absolute-path-id",
            "text": "something else",
            "metadata": {},
        },
    }

    result = ingestor.run(data_dir)

    assert result.files_unverified == 1
    assert result.files_new == 0
    assert embedder.calls == []
    assert store.upsert_calls == 0


def test_relative_path_match_survives_source_id_change(tmp_path):
    ingestor, embedder, store, data_dir, _state_dir = _build_ingestor(tmp_path)
    path = data_dir / "scan.txt"
    _write_text(path, "Same relative path.")
    digest = FileParser().content_sha256(path)
    store.points["docker:chunk:000000"] = {
        "vector": [1.0, 1.0],
        "payload": {
            "chunk_id": "docker:chunk:000000",
            "source_id": "linux-abs-path-id",
            "text": "Same relative path.",
            "metadata": {
                "relative_path": "scan.txt",
                "content_sha256": digest,
            },
        },
    }

    result = ingestor.run(data_dir)

    assert result.files_unchanged == 1
    assert embedder.calls == []


def test_image_file_indexes_path_metadata_not_pixels(tmp_path):
    ingestor, embedder, store, data_dir, _state_dir = _build_ingestor(tmp_path)
    image_path = data_dir / "images" / "xray.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"\x89PNG\r\nnot-a-real-pixel-tensor")

    result = ingestor.run(data_dir)

    assert result.files_indexed == 1
    assert store.upsert_calls == 1
    payload = next(iter(store.points.values()))["payload"]
    assert payload["metadata"]["file_type"] == "image"
    assert payload["metadata"]["relative_path"] == "images/xray.png"
    assert payload["metadata"]["images"]
    assert "xray" in embedder.calls[0][0].lower()
    assert "not-a-real-pixel-tensor" not in embedder.calls[0][0]


def test_startup_flag_disables_ingestion(tmp_path):
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    settings = AppSettings(
        data_dir=str(tmp_path / "data"),
        ingest_state_dir=str(tmp_path / "state"),
        ingest_on_startup=False,
    )
    (tmp_path / "data").mkdir()

    result = run_startup_ingestion(
        embedder=embedder,
        store=store,
        settings=settings,
    )

    assert result is None
    assert embedder.calls == []
    assert store.upsert_calls == 0


def test_startup_runs_incremental_pipeline(tmp_path):
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    data_dir = tmp_path / "data"
    _write_text(data_dir / "notes.txt", "Startup file.")
    settings = AppSettings(
        data_dir=str(data_dir),
        ingest_state_dir=str(tmp_path / "state"),
        ingest_on_startup=True,
    )

    result = run_startup_ingestion(
        embedder=embedder,
        store=store,
        settings=settings,
    )

    assert result is not None
    assert result.files_indexed == 1
    assert store.upsert_calls == 1


def test_pipeline_wrapper_uses_incremental_ingestor(tmp_path):
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    data_dir = tmp_path / "data"
    _write_text(data_dir / "notes.txt", "Via pipeline.")
    pipeline = IngestionPipeline(
        parser=FileParser(),
        chunker=ParentChildChunker(),
        indexer=Indexer(embedder=embedder, store=store),
        state_dir=tmp_path / "state",
    )

    result = pipeline.run(data_dir)

    assert result.files_new == 1
    assert result.chunks_indexed == 1


def test_in_process_lock_skips_second_writer(tmp_path):
    ingestor, _embedder, _store, data_dir, _state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "notes.txt", "Locked.")
    lock = incremental_mod._thread_lock_for(ingestor._lock_path)
    assert lock.acquire(blocking=False)

    try:
        result = ingestor.run(data_dir)
    finally:
        lock.release()

    assert result.skipped is True
    assert result.skip_reason == "in_process_lock"


def test_file_lock_skips_when_held(tmp_path, monkeypatch):
    ingestor, embedder, store, data_dir, _state_dir = _build_ingestor(tmp_path)
    _write_text(data_dir / "notes.txt", "Externally locked.")
    monkeypatch.setattr(
        incremental_mod,
        "_acquire_file_lock",
        lambda _path: (False, None),
    )

    result = ingestor.run(data_dir)

    assert result.skipped is True
    assert result.skip_reason == "file_lock"
    assert embedder.calls == []
    assert store.upsert_calls == 0
