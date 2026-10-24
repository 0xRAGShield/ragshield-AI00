from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  
    build_safe_pipeline,
    configure_logging,
    file_sha256,
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
from metrics import mean_ignore_none  # noqa: E402


def _token_f1(pred: str | None, ref: str | None) -> float | None:
    if not pred or not ref:
        return None
    pred_tokens = pred.casefold().split()
    ref_tokens = ref.casefold().split()
    if not pred_tokens or not ref_tokens:
        return None
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    overlap = len(pred_set & ref_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _citation_overlap(answer: str | None, evidence_ids: list[str]) -> dict:
    if not answer:
        return {"cited_chunk_ids_found": [], "unsupported_citation_like_ids": None}
    found = [cid for cid in evidence_ids if cid and cid in answer]
    return {"cited_chunk_ids_found": found, "n_gold_ids_mentioned": len(found)}


async def _run_live(config: dict, rows: list[dict], logger) -> list[dict]:
    pipeline, settings = build_safe_pipeline()
    redact = bool(config.get("redact_evidence_text", True))
    results: list[dict] = []
    try:
        await pipeline.startup()
        for row in rows:
            record: dict = {
                "query_id": row["query_id"],
                "query": row["query"],
                "answerable": row.get("answerable"),
                "error": None,
                "human_correctness": None,
                "human_faithfulness": None,
                "human_relevance": None,
                "human_completeness": None,
                "human_abstention": None,
                "human_medical_safety": None,
            }
            try:
                result = await pipeline.run(
                    row["query"],
                    top_k=config.get("top_k"),
                    top_n=config.get("top_n"),
                )
                gen = result.generation
                answer = None if gen is None else gen.text
                evidence_ids = [block.identity.chunk_id for block in result.context.evidence]
                abstained = looks_like_abstention(answer or "")
                expected_unanswerable = row.get("answerable") is False
                record.update(
                    {
                        "trace_id": result.trace_id,
                        "is_degraded": result.is_degraded,
                        "generation_is_none": gen is None,
                        "answer": redact_text(answer, redact),
                        "answer_raw_kept": not redact,
                        "reference_answer_present": bool(row.get("reference_answer")),
                        "token_f1_vs_reference": _token_f1(
                            answer, row.get("reference_answer")
                        ),
                        "automated_abstention_heuristic": abstained,
                        "abstention_match": (
                            None
                            if row.get("answerable") is None
                            else (abstained == expected_unanswerable)
                        ),
                        "evidence_chunk_ids": evidence_ids,
                        "image_source_ids": [img.source_id for img in result.images],
                        "prompt_safety_mode": result.prompt.safety_mode,
                        "prompt_is_empty_evidence": result.prompt.is_empty_evidence,
                        "generation_latency_ms": None if gen is None else gen.latency_ms,
                        "generation_model_id": None if gen is None else gen.model_id,
                        "generation_quantization": None if gen is None else gen.quantization,
                        "llm_settings_model_id": settings.llm.model_id,
                        "citations": _citation_overlap(answer, evidence_ids),
                        "optional_llm_judge_requested": bool(
                            config.get("optional_llm_judge")
                        ),
                        "optional_llm_judge_ran": False,
                        "note": (
                            "token_f1 is a weak proxy. Human faithfulness/correctness "
                            "are the primary generation metrics and remain null here."
                        ),
                    }
                )
                if not redact:
                    record["answer"] = answer
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                logger.exception("generation failed query_id=%s", row["query_id"])
            results.append(record)
    finally:
        await pipeline.shutdown()
    return results


def main() -> int:
    logger = configure_logging()
    args = parse_runner_args("RAGShield Phase 2 generation evaluation")
    config = merge_runtime_flags(load_config(args.config), args)
    apply_eval_env(config)
    dataset_path = resolve_path(config["dataset_path"])
    output_dir = resolve_path(config["output_dir"]) / "generation"
    rows = limit_rows(load_jsonl(dataset_path), config.get("max_cases"))
    execute = bool(config.get("execute"))

    if config.get("optional_llm_judge"):
        logger.info(
            "optional_llm_judge=true but no judge module exists in this repo; skipped."
        )

    if execute:
        per_query = asyncio.run(_run_live(config, rows, logger))
    else:
        logger.info("DRY-RUN generation (%s rows)", len(rows))
        per_query = [
            {
                "query_id": row["query_id"],
                "dry_run": True,
                "would_call": "RAGPipeline.run",
                "has_reference_answer": bool(row.get("reference_answer")),
                "human_correctness": None,
                "human_faithfulness": None,
                "human_relevance": None,
                "human_completeness": None,
                "human_abstention": None,
                "human_medical_safety": None,
            }
            for row in rows
        ]

    aggregates = {
        "n_rows": len(per_query),
        "n_errors": sum(1 for row in per_query if row.get("error")),
        "n_degraded": sum(1 for row in per_query if row.get("is_degraded")),
        "n_generation_none": sum(1 for row in per_query if row.get("generation_is_none")),
        "mean_token_f1": mean_ignore_none(
            [row.get("token_f1_vs_reference") for row in per_query]
        ),
        "mean_abstention_match": mean_ignore_none(
            [
                float(row["abstention_match"])
                for row in per_query
                if row.get("abstention_match") is not None
            ]
        ),
        "human_metrics_filled": False,
    }
    meta = {
        "suite": "generation",
        "generated_at_utc": utc_now(),
        "execute": execute,
        "dataset_sha256": file_sha256(dataset_path),
        "config": redact_mapping({k: v for k, v in config.items() if k != "notes"}),
        "primary_metrics": [
            "human_correctness",
            "human_faithfulness",
            "human_relevance",
            "human_completeness",
            "human_abstention",
            "human_medical_safety",
        ],
        "secondary_automated": ["token_f1_vs_reference", "abstention_match"],
    }
    write_jsonl(output_dir / "per_query.jsonl", per_query)
    write_json(output_dir / "aggregate.json", {"meta": meta, "metrics": aggregates})
    write_markdown(
        output_dir / "report.md",
        "# Generation evaluation report\n\n"
        "Primary scores are **human** columns. token_f1 is optional and misleading "
        "for valid Arabic paraphrase. No LLM judge is bundled.\n",
    )
    logger.info("Wrote %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
