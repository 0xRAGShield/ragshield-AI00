
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
    build_safe_pipeline,
    configure_logging,
    ensure_rag_gen_on_path,
    file_sha256,
    gold_chunk_ids,
    limit_rows,
    load_config,
    load_jsonl,
    looks_like_abstention,
    merge_runtime_flags,
    parse_runner_args,
    redact_mapping,
    redact_text,
    resolve_path,
    utc_now,
    write_json,
    write_jsonl,
    write_markdown,
)
from metrics import hit_at_k, mean_ignore_none  # noqa: E402


QUERY_RESPONSE_FIELDS = (
    "answer",
    "trace_id",
    "degraded",
    "retrieved_candidates",
    "reranked_candidates",
    "evidence_items",
    "images",
)


def _schema_ok(payload: dict) -> bool:
    return all(field in payload for field in QUERY_RESPONSE_FIELDS)


async def _run_live(config: dict, rows: list[dict], logger) -> list[dict]:
    ensure_rag_gen_on_path()
    from routers.query_router import classify_query

    pipeline, settings = build_safe_pipeline()
    redact = bool(config.get("redact_evidence_text", True))
    track = config.get("evaluation_track") or "full_pipeline"
    results: list[dict] = []
    try:
        await pipeline.startup()
        for row in rows:
            record: dict = {
                "query_id": row["query_id"],
                "query": row["query"],
                "task": row.get("task"),
                "answerable": row.get("answerable"),
                "evaluation_track": track,
                "error": None,
                "e2e_success": False,
            }
            query_text = row["query"]
            try:
                route = classify_query(query_text)
                record["predicted_route"] = route
                record["route_expected"] = row.get("route_expected")
                record["route_correct"] = (
                    None
                    if row.get("route_expected") not in {"CHAT", "RAG"}
                    else route == row.get("route_expected")
                )
                if track == "api_faithful" and route == "CHAT":
                    record.update(
                        {
                            "skipped_pipeline": True,
                            "e2e_success": True,
                            "http_equivalent": {
                                "answer": "<canned chat; not executed in dry terms>",
                                "trace_id": None,
                                "degraded": False,
                                "retrieved_candidates": 0,
                                "reranked_candidates": 0,
                                "evidence_items": 0,
                                "images": [],
                            },
                            "note": (
                                "API-faithful CHAT short-circuit. "
                                "generate_chat_response is not invoked here; "
                                "mark human_chat_ok separately if needed."
                            ),
                        }
                    )
                    results.append(record)
                    continue

                result = await pipeline.run(
                    query_text,
                    top_k=config.get("top_k"),
                    top_n=config.get("top_n"),
                )
                gen = result.generation
                answer = None if gen is None else gen.text
                http_equiv = {
                    "answer": redact_text(answer, redact) if answer else (
                        "Answer generation failed, but retrieved sources are included. "
                        "Any matching medical images are listed in the images field."
                        if gen is None
                        else ""
                    ),
                    "trace_id": result.trace_id,
                    "degraded": bool(result.is_degraded or gen is None),
                    "retrieved_candidates": len(result.retrieval.candidates),
                    "reranked_candidates": len(result.reranking.candidates),
                    "evidence_items": len(result.context.evidence),
                    "images": [
                        {"url": img.url, "source_id": img.source_id}
                        for img in result.images
                    ],
                }
                gold = gold_chunk_ids(row)
                context_ids = [b.identity.chunk_id for b in result.context.evidence]
                unanswerable = row.get("answerable") is False
                abstained = looks_like_abstention(answer or "")
                quality_ok = None
                if row.get("reference_answer") and answer:
                    quality_ok = None  # human / generation suite; not e2e success
                expected_degraded = bool(row.get("task") == "expect_degraded")
                control_ok = _schema_ok(http_equiv) and record.get("error") is None
                if unanswerable:
                    answerability_ok = abstained or (gen is None)
                elif row.get("answerable") is True:
                    answerability_ok = bool(answer) and gen is not None
                else:
                    answerability_ok = None
                record.update(
                    {
                        "skipped_pipeline": False,
                        "trace_id": result.trace_id,
                        "http_equivalent": http_equiv,
                        "schema_valid": _schema_ok(http_equiv),
                        "is_degraded": result.is_degraded,
                        "generation_is_none": gen is None,
                        "retrieved_chunk_ids": [
                            c.chunk_id for c in result.retrieval.candidates
                        ],
                        "reranked_chunk_ids": [
                            c.chunk_id for c in result.reranking.candidates
                        ],
                        "context_chunk_ids": context_ids,
                        "image_source_ids": [img.source_id for img in result.images],
                        "prompt_safety_mode": result.prompt.safety_mode,
                        "retrieval_latency_ms": result.retrieval.metrics.total_latency_ms,
                        "rerank_latency_ms": result.reranking.metrics.execution_time_ms,
                        "context_latency_ms": result.context.metrics.context_build_time_ms,
                        "generation_latency_ms": None if gen is None else gen.latency_ms,
                        "gold_hit_retrieval": hit_at_k(
                            [c.chunk_id for c in result.retrieval.candidates],
                            gold,
                            result.retrieval.metrics.requested_top_k or 5,
                        ),
                        "gold_in_context": (
                            None
                            if not gold
                            else float(any(cid in set(context_ids) for cid in gold))
                        ),
                        "answerability_ok": answerability_ok,
                        "expected_degraded": expected_degraded,
                        "e2e_success": bool(control_ok),
                        "answer_quality_not_used_for_e2e_success": quality_ok,
                        "settings_collection": settings.retrieval.collection_name,
                    }
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                record["e2e_success"] = False
                if row.get("task") == "malformed_expect_error":
                    record["e2e_success"] = True
                    record["expected_error_observed"] = True
                logger.exception("e2e failed query_id=%s", row["query_id"])
            results.append(record)
    finally:
        await pipeline.shutdown()
    return results


def main() -> int:
    logger = configure_logging()
    args = parse_runner_args("RAGShield Phase 2 end-to-end evaluation")
    config = merge_runtime_flags(load_config(args.config), args)
    apply_eval_env(config)
    dataset_path = resolve_path(config["dataset_path"])
    output_dir = resolve_path(config["output_dir"]) / "end_to_end"
    rows = limit_rows(load_jsonl(dataset_path), config.get("max_cases"))
    execute = bool(config.get("execute"))

    if execute:
        per_query = asyncio.run(_run_live(config, rows, logger))
    else:
        logger.info("DRY-RUN end-to-end (%s rows)", len(rows))
        per_query = [
            {
                "query_id": row["query_id"],
                "task": row.get("task"),
                "dry_run": True,
                "would_call": "classify_query + RAGPipeline.run",
                "evaluation_track": config.get("evaluation_track"),
                "e2e_success": None,
                "note": "e2e_success is schema/control success, not answer quality.",
            }
            for row in rows
        ]

    scored = [row for row in per_query if row.get("e2e_success") is not None]
    aggregates = {
        "n_rows": len(per_query),
        "n_errors": sum(1 for row in per_query if row.get("error")),
        "e2e_success_rate": mean_ignore_none(
            [float(row["e2e_success"]) for row in scored]
        ),
        "n_degraded": sum(1 for row in per_query if row.get("is_degraded")),
        "mean_retrieval_latency_ms": mean_ignore_none(
            [row.get("retrieval_latency_ms") for row in per_query]
        ),
        "mean_rerank_latency_ms": mean_ignore_none(
            [row.get("rerank_latency_ms") for row in per_query]
        ),
        "mean_generation_latency_ms": mean_ignore_none(
            [row.get("generation_latency_ms") for row in per_query]
        ),
        "routing_accuracy": mean_ignore_none(
            [
                float(row["route_correct"])
                for row in per_query
                if row.get("route_correct") is not None
            ]
        ),
    }
    meta = {
        "suite": "end_to_end",
        "generated_at_utc": utc_now(),
        "execute": execute,
        "dataset_sha256": file_sha256(dataset_path),
        "entry_point": "RAGPipeline.run (in-process); HTTP not used",
        "docker_api": "RAG_GEN:app POST /query applies classify_query",
        "config": redact_mapping({k: v for k, v in config.items() if k != "notes"}),
    }
    write_jsonl(output_dir / "per_query.jsonl", per_query)
    write_json(output_dir / "aggregate.json", {"meta": meta, "metrics": aggregates})
    write_markdown(
        output_dir / "report.md",
        "# End-to-end evaluation report\n\n"
        "e2e_success = pipeline completed with QueryResponse-shaped fields "
        "(or an expected malformed error). It is **not** answer correctness.\n",
    )
    logger.info("Wrote %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
