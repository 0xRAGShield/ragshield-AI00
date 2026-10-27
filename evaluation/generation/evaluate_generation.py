from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    HUMAN_RATING_FIELDS,
    HUMAN_RATING_PRECEDENCE,
    HUMAN_RATING_SCALE,
    apply_eval_env,
    attach_human_ratings,
    build_safe_pipeline,
    configure_logging,
    expected_abstention_label,
    file_sha256,
    human_ratings_from_row,
    limit_rows,
    load_config,
    load_human_rating_index,
    load_jsonl,
    looks_like_abstention,
    merge_runtime_flags,
    parse_runner_args,
    redact_mapping,
    redact_text,
    resolve_human_review_path,
    resolve_path,
    utc_now,
    write_json,
    write_jsonl,
    write_markdown,
)
from metrics import (  # noqa: E402
    EXCLUSION_NO_ABSTENTION_LABEL,
    EXCLUSION_NO_REFERENCE,
    exact_match,
    human_rating_aggregates,
    key_fact_coverage,
    query_answer_overlap,
    token_overlap,
)
from reporting import coverage_from_per_query, generation_report_markdown  # noqa: E402


def _citation_overlap(answer: str | None, evidence_ids: list[str]) -> dict:
    if not answer:
        return {"cited_chunk_ids_found": [], "unsupported_citation_like_ids": None}
    found = [cid for cid in evidence_ids if cid and cid in answer]
    return {"cited_chunk_ids_found": found, "n_gold_ids_mentioned": len(found)}


def seed_generation_record(row: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "query_id": row["query_id"],
        "query": row.get("query"),
        "answerable": row.get("answerable"),
        "error": None,
        **human_ratings_from_row(row),
    }
    if dry_run:
        record.update(
            {
                "dry_run": True,
                "would_call": "RAGPipeline.run",
                "has_reference_answer": bool(row.get("reference_answer")),
            }
        )
    return record


def finalize_generation_records(
    records: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset_by_id = {str(row["query_id"]): row for row in dataset_rows}
    review_path = resolve_human_review_path(config, output_dir)
    review_by_id = load_human_rating_index(review_path, source="review")
    existing_path = output_dir / "per_query.jsonl"
    existing_by_id = (
        load_human_rating_index(existing_path, source="existing_per_query")
        if existing_path.is_file()
        else {}
    )
    return attach_human_ratings(
        records,
        dataset_by_id=dataset_by_id,
        review_by_id=review_by_id,
        existing_by_id=existing_by_id,
    )


async def _run_live(config: dict, rows: list[dict], logger) -> list[dict]:
    pipeline, settings = build_safe_pipeline()
    redact = bool(config.get("redact_answer_text", True))
    results: list[dict] = []
    try:
        await pipeline.startup()
        for row in rows:
            record = seed_generation_record(row, dry_run=False)
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
                expect_abstain = expected_abstention_label(row)
                overlap = token_overlap(answer, row.get("reference_answer"))
                record.update(
                    {
                        "trace_id": result.trace_id,
                        "is_degraded": result.is_degraded,
                        "generation_is_none": gen is None,
                        "answer": redact_text(answer, redact),
                        "answer_raw_kept": not redact,
                        "reference_answer_present": bool(row.get("reference_answer")),
                        "lexical_exact_match": exact_match(
                            answer, row.get("reference_answer")
                        ),
                        "token_precision_vs_reference": overlap["token_precision"],
                        "token_recall_vs_reference": overlap["token_recall"],
                        "token_f1_vs_reference": overlap["token_f1"],
                        "lexical_overlap_not_clinical_correctness": True,
                        "relevancy_lexical_diagnostic": query_answer_overlap(
                            row["query"], answer
                        ),
                        "completeness_key_fact_lexical": key_fact_coverage(
                            answer, row.get("key_facts")
                        ),
                        "automated_abstention_heuristic": abstained,
                        "expected_abstention": expect_abstain,
                        "abstention_match": (
                            None
                            if expect_abstain is None
                            else (abstained == expect_abstain)
                        ),
                        "evidence_chunk_ids": evidence_ids,
                        "evidence_text": [
                            {
                                "chunk_id": block.identity.chunk_id,
                                "source_id": block.identity.source_id,
                                "text": redact_text(
                                    block.content,
                                    bool(config.get("redact_evidence_text", True)),
                                ),
                            }
                            for block in result.context.evidence
                        ],
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
                            "token_f1 is a weak lexical proxy, not clinical correctness. "
                            "Human 0-2 ratings are joined by query_id from the dataset; "
                            "an optional review file may override non-null fields only."
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
    output_dir = resolve_path(config["output_dir"])
    output_dir = (
        (output_dir / "dry_run" / "generation")
        if not bool(config.get("execute"))
        else (output_dir / "generation")
    )
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
        per_query = [seed_generation_record(row, dry_run=True) for row in rows]

    per_query = finalize_generation_records(
        per_query, rows, output_dir=output_dir, config=config
    )

    f1 = coverage_from_per_query(
        per_query, "token_f1_vs_reference", exclude_reason=EXCLUSION_NO_REFERENCE
    )
    em = coverage_from_per_query(
        per_query, "lexical_exact_match", exclude_reason=EXCLUSION_NO_REFERENCE
    )
    abst = coverage_from_per_query(
        per_query,
        "abstention_match",
        exclude_reason=EXCLUSION_NO_ABSTENTION_LABEL,
    )
    rel = coverage_from_per_query(
        per_query,
        "relevancy_lexical_diagnostic",
        exclude_reason="missing_query_or_answer",
    )
    human_agg = human_rating_aggregates(per_query)
    aggregates = {
        "n_rows": len(per_query),
        "n_errors": sum(1 for row in per_query if row.get("error")),
        "n_degraded": sum(1 for row in per_query if row.get("is_degraded")),
        "n_generation_none": sum(1 for row in per_query if row.get("generation_is_none")),
        "mean_token_f1": f1["mean"],
        "mean_token_f1_n": f1["n_eligible"],
        "mean_token_f1_n_excluded": f1["n_excluded"],
        "mean_token_f1_excluded_query_ids": f1["excluded_query_ids"],
        "mean_lexical_exact_match": em["mean"],
        "mean_lexical_exact_match_n": em["n_eligible"],
        "mean_abstention_match": abst["mean"],
        "mean_abstention_match_n": abst["n_eligible"],
        "mean_abstention_match_n_excluded": abst["n_excluded"],
        "mean_relevancy_lexical_diagnostic": rel["mean"],
        "mean_relevancy_lexical_diagnostic_n": rel["n_eligible"],
        **human_agg,
    }
    meta = {
        "suite": "generation",
        "generated_at_utc": utc_now(),
        "execute": execute,
        "dataset_sha256": file_sha256(dataset_path),
        "human_review_path": str(resolve_human_review_path(config, output_dir) or ""),
        "human_rating_precedence": list(HUMAN_RATING_PRECEDENCE),
        "config": redact_mapping({k: v for k, v in config.items() if k != "notes"}),
        "primary_metrics": list(HUMAN_RATING_FIELDS),
        "secondary_automated": [
            "token_f1_vs_reference",
            "lexical_exact_match",
            "abstention_match",
            "relevancy_lexical_diagnostic",
        ],
        "limitations": [
            "token_f1 and exact_match are lexical, not clinical correctness.",
            "relevancy_lexical_diagnostic is query-token overlap, not RAGAS answer relevancy.",
            "completeness_key_fact_lexical requires key_facts; otherwise ineligible.",
            "No LLM judge is implemented.",
            "Human ratings are not inferred from generated text; null stays unannotated.",
            "normalized_pct is (mean / 2) * 100 and is not an accuracy percentage.",
        ],
        "human_rating_scale": HUMAN_RATING_SCALE,
    }
    write_jsonl(output_dir / "per_query.jsonl", per_query)
    write_json(output_dir / "aggregate.json", {"meta": meta, "metrics": aggregates})
    write_markdown(
        output_dir / "report.md",
        generation_report_markdown(aggregates, execute=execute),
    )
    logger.info("Wrote %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
