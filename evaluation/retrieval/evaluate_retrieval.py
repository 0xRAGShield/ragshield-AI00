from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  
    apply_eval_env,
    build_retriever_only,
    candidate_payload,
    configure_logging,
    ensure_rag_gen_on_path,
    file_sha256,
    gold_chunk_ids,
    gold_source_ids,
    limit_rows,
    load_config,
    load_jsonl,
    merge_runtime_flags,
    parse_runner_args,
    redact_mapping,
    resolve_path,
    utc_now,
    write_json,
    write_jsonl,
    write_markdown,
)
from metrics import mean_ignore_none, ranking_bundle  # noqa: E402


def _aggregate(rows: list[dict], k_values: list[int], prefix: str) -> dict:
    summary = {}
    for k in k_values:
        for name in ("hit", "recall", "precision", "mrr", "ndcg"):
            key = f"{name}@{k}"
            summary[f"{prefix}_{key}"] = mean_ignore_none(
                [row.get("metrics", {}).get(key) for row in rows]
            )
            summary[f"{prefix}_{key}_n"] = sum(
                1 for row in rows if row.get("metrics", {}).get(key) is not None
            )
    return summary


async def _run_live(config: dict, rows: list[dict], logger) -> list[dict]:
    ensure_rag_gen_on_path()
    from app.models.rag_models import RetrievalQuery

    retriever, settings = build_retriever_only()
    k_values = list(config.get("k_values") or [5])
    top_k = config.get("top_k")
    redact = bool(config.get("redact_evidence_text", True))
    results: list[dict] = []
    try:
        for row in rows:
            record: dict = {
                "query_id": row["query_id"],
                "query": row["query"],
                "language": row.get("language"),
                "error": None,
            }
            try:
                result = await retriever.retrieve(
                    RetrievalQuery(text=row["query"], top_k=top_k)
                )
                ranked_chunks = [c.chunk_id for c in result.candidates]
                ranked_sources = [c.source_id for c in result.candidates]
                gold_chunks = gold_chunk_ids(row)
                gold_sources = gold_source_ids(row)
                metrics = {}
                for k in k_values:
                    metrics.update(
                        ranking_bundle(
                            ranked_chunks,
                            gold_chunks,
                            k,
                            row.get("relevance_judgments"),
                        )
                    )
                    source_metrics = ranking_bundle(ranked_sources, gold_sources, k)
                    metrics[f"source_hit@{k}"] = source_metrics[f"hit@{k}"]
                    metrics[f"source_recall@{k}"] = source_metrics[f"recall@{k}"]
                record.update(
                    {
                        "trace_id": result.metrics.trace_id,
                        "requested_top_k": result.metrics.requested_top_k,
                        "returned_candidates": len(result.candidates),
                        "no_result": len(result.candidates) == 0,
                        "duplicate_chunk_ids": len(ranked_chunks) != len(set(ranked_chunks)),
                        "is_degraded": result.is_degraded or result.metrics.is_degraded,
                        "embedding_latency_ms": result.metrics.embedding_latency_ms,
                        "search_latency_ms": result.metrics.search_latency_ms,
                        "total_latency_ms": result.metrics.total_latency_ms,
                        "candidates": [
                            candidate_payload(c, redact=redact) for c in result.candidates
                        ],
                        "metrics": metrics,
                    }
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                logger.exception("retrieval failed query_id=%s", row["query_id"])
            results.append(record)
    finally:
        retriever.close()
    return results


def _dry_run(rows: list[dict], k_values: list[int]) -> list[dict]:
    return [
        {
            "query_id": row["query_id"],
            "query": row["query"],
            "dry_run": True,
            "would_call": "Retriever.retrieve(RetrievalQuery)",
            "k_values": k_values,
            "has_chunk_gold": gold_chunk_ids(row) is not None,
            "has_source_gold": gold_source_ids(row) is not None,
            "metrics": None,
            "note": "No live retrieval. Labels empty → metrics skipped, not scored as 0.",
        }
        for row in rows
    ]


def _report(meta: dict, aggregates: dict) -> str:
    lines = [
        "# Retrieval evaluation report",
        "",
        "This file is a run artifact. The design session did not execute retrieval.",
        "",
        "## Environment",
        f"- generated_at_utc: {meta.get('generated_at_utc')}",
        f"- execute: {meta.get('execute')}",
        f"- dataset_sha256: {meta.get('dataset_sha256')}",
        f"- n_rows: {meta.get('n_rows')}",
        f"- k_values: {meta.get('k_values')}",
        "",
        "## Aggregates (None means unlabeled or dry-run)",
    ]
    for key, value in aggregates.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Denominators",
            "- Hit@k: 1 if ≥1 gold chunk_id in top k; mean over labeled queries.",
            "- Recall@k: |top-k ∩ gold| / |gold|.",
            "- Precision@k: |top-k ∩ gold| / k (cutoff k, even if fewer hits returned).",
            "- MRR@k: 1/rank of first gold, else 0; unlabeled excluded from the mean.",
            "- nDCG@k: requires gold or relevance_judgments; skipped otherwise.",
            "",
            "## Limitations",
            "- Dense cosine search only (Retriever forbids hybrid/sparse).",
            "- Query embedding uses expand_query_with_document_aliases; original text stays on RetrievalQuery.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    logger = configure_logging()
    args = parse_runner_args("RAGShield Phase 2 retrieval evaluation")
    config = merge_runtime_flags(load_config(args.config), args)
    apply_eval_env(config)

    dataset_path = resolve_path(config["dataset_path"])
    output_dir = resolve_path(config["output_dir"]) / "retrieval"
    rows = limit_rows(load_jsonl(dataset_path), config.get("max_cases"))
    k_values = list(config.get("k_values") or [5])
    execute = bool(config.get("execute"))

    if execute:
        logger.info("LIVE retrieval starting for %s rows", len(rows))
        per_query = asyncio.run(_run_live(config, rows, logger))
    else:
        logger.info("DRY-RUN retrieval for %s rows (pass --execute to call Retriever)", len(rows))
        per_query = _dry_run(rows, k_values)

    labeled_metrics = [row for row in per_query if row.get("metrics")]
    aggregates = {
        "n_rows": len(per_query),
        "n_errors": sum(1 for row in per_query if row.get("error")),
        "n_no_result": sum(1 for row in per_query if row.get("no_result")),
        **_aggregate(labeled_metrics, k_values, "chunk"),
    }
    meta = {
        "suite": "retrieval",
        "generated_at_utc": utc_now(),
        "execute": execute,
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "n_rows": len(rows),
        "k_values": k_values,
        "top_k": config.get("top_k"),
        "config": redact_mapping({k: v for k, v in config.items() if k != "notes"}),
    }
    write_jsonl(output_dir / "per_query.jsonl", per_query)
    write_json(output_dir / "aggregate.json", {"meta": meta, "metrics": aggregates})
    write_markdown(output_dir / "report.md", _report(meta, aggregates))
    logger.info("Wrote %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
