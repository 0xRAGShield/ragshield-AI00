from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import ( 
    apply_eval_env,
    build_reranker_only,
    build_retriever_only,
    candidate_payload,
    configure_logging,
    ensure_rag_gen_on_path,
    file_sha256,
    gold_chunk_ids,
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


def _load_frozen_pool(path: Path) -> dict[str, list[dict]]:
    pool: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        pool[str(obj["query_id"])] = list(obj["candidates"])
    return pool


def _metrics_for(ranked_ids: list[str], gold, k_values, judgments) -> dict:
    bundle = {}
    for k in k_values:
        bundle.update(ranking_bundle(ranked_ids, gold, k, judgments))
    return bundle


async def _run_live(config: dict, rows: list[dict], logger) -> list[dict]:
    ensure_rag_gen_on_path()
    from app.models.rag_models import (
        RetrievalCandidate,
        RetrievalMetrics,
        RetrievalQuery,
        RetrievalResult,
    )

    k_values = list(config.get("k_values") or [5])
    top_k = config.get("top_k")
    top_n = config.get("top_n")
    redact = bool(config.get("redact_evidence_text", True))
    frozen_path = config.get("frozen_candidate_pool_path")
    frozen = _load_frozen_pool(resolve_path(frozen_path)) if frozen_path else None

    retriever = None
    settings = None
    if frozen is None:
        retriever, settings = build_retriever_only()
    reranker, rerank_settings = build_reranker_only()
    settings = settings or rerank_settings
    results: list[dict] = []
    try:
        for row in rows:
            record: dict = {"query_id": row["query_id"], "query": row["query"], "error": None}
            try:
                if frozen is not None:
                    raw = frozen.get(row["query_id"])
                    if raw is None:
                        raise KeyError(
                            f"frozen pool missing query_id={row['query_id']}"
                        )
                    candidates = [
                        RetrievalCandidate(
                            chunk_id=item["chunk_id"],
                            source_id=item["source_id"],
                            text=item["text"],
                            score=float(item["score"]),
                            metadata=dict(item.get("metadata") or {}),
                        )
                        for item in raw
                    ]
                    retrieval = RetrievalResult(
                        query=RetrievalQuery(text=row["query"], top_k=top_k),
                        candidates=candidates,
                        metrics=RetrievalMetrics(
                            requested_top_k=len(candidates),
                            returned_candidates=len(candidates),
                        ),
                    )
                    pool_source = "frozen_jsonl"
                else:
                    retrieval = await retriever.retrieve(
                        RetrievalQuery(text=row["query"], top_k=top_k)
                    )
                    pool_source = "live_retriever"

                pre_ids = [c.chunk_id for c in retrieval.candidates]
                reranked = await reranker.rerank(retrieval, top_n=top_n)
                post_ids = [c.chunk_id for c in reranked.candidates]
                gold = gold_chunk_ids(row)
                judgments = row.get("relevance_judgments")
                pre = _metrics_for(pre_ids, gold, k_values, judgments)
                post = _metrics_for(post_ids, gold, k_values, judgments)
                delta = {}
                for key, post_value in post.items():
                    pre_value = pre.get(key)
                    if pre_value is None or post_value is None:
                        delta[key] = None
                    else:
                        delta[key] = post_value - pre_value
                gold_set = set(gold or [])
                missing_from_pool = sorted(gold_set.difference(pre_ids)) if gold else []
                record.update(
                    {
                        "pool_source": pool_source,
                        "input_count": len(pre_ids),
                        "output_count": len(post_ids),
                        "is_fallback": reranked.is_fallback,
                        "is_degraded": reranked.is_degraded,
                        "scoring_time_ms": reranked.metrics.scoring_time_ms,
                        "execution_time_ms": reranked.metrics.execution_time_ms,
                        "gold_missing_from_input_pool": missing_from_pool,
                        "pre_chunk_ids": pre_ids,
                        "post_chunk_ids": post_ids,
                        "pre_candidates": [
                            candidate_payload(c, redact=redact)
                            for c in retrieval.candidates
                        ],
                        "post_candidates": [
                            candidate_payload(c, redact=redact)
                            for c in reranked.candidates
                        ],
                        "pre_metrics": pre,
                        "post_metrics": post,
                        "delta_metrics": delta,
                    }
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                logger.exception("rerank failed query_id=%s", row["query_id"])
            results.append(record)
    finally:
        reranker.close()
        if retriever is not None:
            retriever.close()
    return results


def _dry_run(rows: list[dict], k_values: list[int]) -> list[dict]:
    return [
        {
            "query_id": row["query_id"],
            "dry_run": True,
            "would_call": "Reranker.rerank(RetrievalResult, top_n=...)",
            "k_values": k_values,
            "note": (
                "Pre/post comparison uses the same candidate pool. "
                "Gold IDs absent from the pool cannot be recovered by reranking."
            ),
        }
        for row in rows
    ]


def main() -> int:
    logger = configure_logging()
    args = parse_runner_args("RAGShield Phase 2 reranker evaluation")
    config = merge_runtime_flags(load_config(args.config), args)
    apply_eval_env(config)
    dataset_path = resolve_path(config["dataset_path"])
    output_dir = resolve_path(config["output_dir"]) / "reranker"
    rows = limit_rows(load_jsonl(dataset_path), config.get("max_cases"))
    k_values = list(config.get("k_values") or [5])
    execute = bool(config.get("execute"))

    if execute:
        per_query = asyncio.run(_run_live(config, rows, logger))
    else:
        logger.info("DRY-RUN reranker (%s rows)", len(rows))
        per_query = _dry_run(rows, k_values)

    def _mean(path_keys: tuple[str, str]) -> dict:
        out = {}
        for k in k_values:
            for name in ("hit", "recall", "precision", "mrr", "ndcg"):
                key = f"{name}@{k}"
                values = [
                    (row.get(path_keys[0]) or {}).get(key)
                    for row in per_query
                ]
                out[f"{path_keys[1]}_{key}"] = mean_ignore_none(values)
        return out

    aggregates = {
        "n_rows": len(per_query),
        "n_errors": sum(1 for row in per_query if row.get("error")),
        "n_fallback": sum(1 for row in per_query if row.get("is_fallback")),
        **_mean(("pre_metrics", "pre")),
        **_mean(("post_metrics", "post")),
        **_mean(("delta_metrics", "delta")),
    }
    meta = {
        "suite": "reranker",
        "generated_at_utc": utc_now(),
        "execute": execute,
        "dataset_sha256": file_sha256(dataset_path),
        "frozen_candidate_pool_path": config.get("frozen_candidate_pool_path"),
        "k_values": k_values,
        "top_n": config.get("top_n"),
        "config": redact_mapping({k: v for k, v in config.items() if k != "notes"}),
        "limitation": (
            "Reranker.rerank only reorders RetrievalResult.candidates; "
            "missing golds in that list stay missing."
        ),
    }
    write_jsonl(output_dir / "per_query.jsonl", per_query)
    write_json(output_dir / "aggregate.json", {"meta": meta, "metrics": aggregates})
    write_markdown(
        output_dir / "report.md",
        "# Reranker evaluation report\n\n"
        f"execute={execute}\n\n"
        "Compare pre_metrics vs post_metrics on the **same** candidate pool. "
        "delta_mrr@k = post − pre. Unlabeled queries are omitted from means.\n",
    )
    logger.info("Wrote %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
