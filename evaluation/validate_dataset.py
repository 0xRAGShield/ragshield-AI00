"""Validate evaluation JSONL against the loader schema. Does not call the pipeline."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    HUMAN_RATING_SCALE,
    parse_human_rating,
    expected_abstention_label,
    expected_route_label,
    file_sha256,
    gold_chunk_ids,
    gold_source_ids,
    human_annotation_status,
    load_config,
    load_jsonl,
    merge_runtime_flags,
    parse_runner_args,
    repo_root,
    resolve_path,
    utc_now,
    write_json,
    write_markdown,
)

_CHUNK_ID_RE = re.compile(r"^[0-9a-f]{64}:chunk:\d{6}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FAITHFULNESS_LABELS = {
    "supported",
    "unsupported",
    "contradicted",
    "indeterminate",
}


def _validate_human(row: dict) -> list[str]:
    issues = []
    for name in HUMAN_RATING_SCALE:
        try:
            parse_human_rating(
                name, row.get(name), query_id=str(row.get("query_id") or "")
            )
        except ValueError as exc:
            issues.append(str(exc))
    return issues


def _iter_chunk_like_ids(row: dict) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for key in ("relevant_chunk_ids", "expected_evidence_ids"):
        for item in row.get(key) or []:
            found.append((key, str(item)))
    for judgment in row.get("relevance_judgments") or []:
        if isinstance(judgment, dict) and judgment.get("chunk_id"):
            found.append(("relevance_judgments.chunk_id", str(judgment["chunk_id"])))
    for claim in row.get("claim_annotations") or []:
        if not isinstance(claim, dict):
            continue
        for eid in claim.get("evidence_ids") or []:
            found.append(("claim_annotations.evidence_ids", str(eid)))
    return found


def _manifest_ids() -> dict[str, set[str]]:
    path = repo_root() / "RAG-GEN" / "ingestion_state" / "ragshield_test.manifest.json"
    source_ids: set[str] = set()
    content_hashes: set[str] = set()
    if not path.is_file():
        return {"source_ids": source_ids, "content_sha256": content_hashes, "path": set()}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"source_ids": source_ids, "content_sha256": content_hashes, "path": set()}
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        return {"source_ids": source_ids, "content_sha256": content_hashes, "path": set()}
    for record in files.values():
        if not isinstance(record, dict):
            continue
        sid = record.get("source_id")
        if isinstance(sid, str) and sid.strip():
            source_ids.add(sid.strip())
        digest = record.get("content_sha256")
        if isinstance(digest, str) and digest.strip():
            content_hashes.add(digest.strip())
    return {"source_ids": source_ids, "content_sha256": content_hashes, "path": set()}


def build_report(rows: list[dict], dataset_path: Path) -> dict:
    ids = [row["query_id"] for row in rows]
    dupes = [qid for qid, n in Counter(ids).items() if n > 1]
    missing_chunk_gold = []
    known_empty = []
    missing_ref = []
    missing_claims = []
    missing_route = []
    missing_abstention = []
    human_issues = []
    claims_missing_faithfulness_label = []
    malformed_chunk_ids = []
    manifest = _manifest_ids()
    gold_source_not_in_manifest = []
    document_id_not_in_manifest_hash = []
    for row in rows:
        qid = row["query_id"]
        gold = gold_chunk_ids(row)
        if gold is None:
            missing_chunk_gold.append(qid)
        elif gold == []:
            known_empty.append(qid)
        ref = row.get("reference_answer")
        if not (isinstance(ref, str) and ref.strip()):
            missing_ref.append(qid)
        claims = row.get("claim_annotations") or []
        if not claims:
            missing_claims.append(qid)
        else:
            unlabeled = []
            for item in claims:
                if not isinstance(item, dict):
                    unlabeled.append("<non-object>")
                    continue
                claim_type = item.get("claim_type")
                if claim_type is not None and not (
                    isinstance(claim_type, str) and claim_type.strip()
                ):
                    unlabeled.append("invalid_claim_type")
                text = str(item.get("claim") or item.get("text") or "").strip()
                label = str(item.get("label") or "").strip().casefold()
                if text and label not in _FAITHFULNESS_LABELS:
                    unlabeled.append(text[:40])
            if unlabeled:
                claims_missing_faithfulness_label.append(qid)
        if expected_route_label(row) is None:
            missing_route.append(qid)
        if expected_abstention_label(row) is None:
            missing_abstention.append(qid)
        human_issues.extend(
            f"{qid}: {msg}" for msg in _validate_human(row)
        )
        for field, cid in _iter_chunk_like_ids(row):
            if not _CHUNK_ID_RE.fullmatch(cid):
                malformed_chunk_ids.append({"query_id": qid, "field": field, "value": cid})
        sources = gold_source_ids(row) or []
        for sid in sources:
            if manifest["source_ids"] and sid not in manifest["source_ids"]:
                gold_source_not_in_manifest.append({"query_id": qid, "source_id": sid})
        for did in row.get("relevant_document_ids") or []:
            text = str(did)
            if (
                manifest["content_sha256"]
                and _SHA256_RE.fullmatch(text)
                and text not in manifest["content_sha256"]
            ):
                document_id_not_in_manifest_hash.append(
                    {"query_id": qid, "relevant_document_id": text}
                )
    return {
        "generated_at_utc": utc_now(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "n_rows": len(rows),
        "duplicate_query_ids": dupes,
        "n_with_binary_chunk_gold": len(rows) - len(missing_chunk_gold) - len(known_empty),
        "unlabelled_chunk_gold_query_ids": missing_chunk_gold,
        "explicit_no_relevant_results_query_ids": known_empty,
        "missing_reference_answer_query_ids": missing_ref,
        "missing_claim_annotations_query_ids": missing_claims,
        "claim_annotations_missing_faithfulness_label_query_ids": (
            claims_missing_faithfulness_label
        ),
        "malformed_chunk_like_ids": malformed_chunk_ids,
        "gold_source_id_not_in_current_manifest": gold_source_not_in_manifest,
        "document_id_not_in_current_manifest_content_sha256": (
            document_id_not_in_manifest_hash
        ),
        "missing_route_label_query_ids": missing_route,
        "missing_abstention_label_query_ids": missing_abstention,
        "human_rating_issues": human_issues,
        "human_annotation_status": human_annotation_status(rows),
        "notes": [
            "Unlabelled fields are omitted from metric means; they are not scored as 0.",
            "This script does not contact Qdrant; chunk IDs are not existence-checked here.",
            "Do not invent relevant_chunk_ids or reference_answer values.",
            "claim_annotations need label in {supported, unsupported, contradicted, indeterminate} for hallucination scoring; support_level is not used as that label.",
            "chunk_id format expected by ingest is {source_id}:chunk:{index:06d} where source_id is a 64-char hex digest.",
        ],
    }


def main() -> int:
    args = parse_runner_args("Validate RAGShield evaluation JSONL")
    config = merge_runtime_flags(load_config(args.config), args)
    dataset_path = resolve_path(args.dataset or config["dataset_path"])
    rows = load_jsonl(dataset_path)
    report = build_report(rows, dataset_path)
    out_dir = resolve_path(config["output_dir"]) / "validation"
    write_json(out_dir / "dataset_validation.json", report)
    lines = [
        "# Dataset validation",
        "",
        f"- path: `{report['dataset_path']}`",
        f"- sha256: `{report['dataset_sha256']}`",
        f"- n_rows: {report['n_rows']}",
        f"- duplicate_query_ids: {report['duplicate_query_ids']}",
        f"- binary chunk gold: {report['n_with_binary_chunk_gold']}",
        f"- unlabelled retrieval: {report['unlabelled_chunk_gold_query_ids']}",
        f"- missing reference_answer: {report['missing_reference_answer_query_ids']}",
        f"- missing claim_annotations: {report['missing_claim_annotations_query_ids']}",
        f"- human: {json.dumps(report['human_annotation_status'])}",
        "",
        "Chunk IDs were not looked up in Qdrant.",
        "",
    ]
    write_markdown(out_dir / "dataset_validation.md", "\n".join(lines))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    blocking = (
        report["duplicate_query_ids"]
        or report["human_rating_issues"]
        or report["claim_annotations_missing_faithfulness_label_query_ids"]
        or report["malformed_chunk_like_ids"]
    )
    if blocking:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
