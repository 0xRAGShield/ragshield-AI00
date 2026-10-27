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
    file_sha256,
    limit_rows,
    load_config,
    load_jsonl,
    merge_runtime_flags,
    parse_runner_args,
    redact_mapping,
    redact_text,
    resolve_path,
    split_claims,
    utc_now,
    write_json,
    write_jsonl,
    write_markdown,
)
from metrics import hallucination_bundle, mean_ignore_none  # noqa: E402
from reporting import coverage_from_per_query  # noqa: E402

VALID_LABELS = {
    "supported",
    "unsupported",
    "contradicted",
    "indeterminate",
}


def _faithfulness_label(item: dict) -> str | None:
    """Return a scoring label only when the dataset set `label` explicitly.

    `support_level` (direct, needs_verification, …) describes gold-claim
    evidence, not whether a generated answer hallucinated. Do not map it.
    """
    raw = item.get("label")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    label = str(raw).strip().casefold()
    if label not in VALID_LABELS:
        return None
    return label


def _normalize_annotations(raw) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("claim") or "").strip()
        label = _faithfulness_label(item)
        if text and label is not None:
            out.append({"text": text, "label": label})
    return out


def _counts(annotations: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in VALID_LABELS}
    for item in annotations:
        counts[item["label"]] += 1
    return counts


def _lexical_overlap_supported(claim: str, evidence_blobs: list[str]) -> bool | None:
    """Weak optional heuristic. Never used as a contradiction label.

    Returns True only if every token longer than 3 chars in the claim appears
    in concatenated evidence. Otherwise None (indeterminate), not False.
    """
    tokens = [tok for tok in claim.casefold().split() if len(tok) > 3]
    if not tokens:
        return None
    blob = "\n".join(evidence_blobs).casefold()
    if all(tok in blob for tok in tokens):
        return True
    return None


async def _run_live(config: dict, rows: list[dict], logger) -> list[dict]:
    pipeline, _settings = build_safe_pipeline()
    redact = bool(config.get("redact_evidence_text", True))
    results: list[dict] = []
    try:
        await pipeline.startup()
        for row in rows:
            record: dict = {"query_id": row["query_id"], "query": row["query"], "error": None}
            try:
                result = await pipeline.run(
                    row["query"],
                    top_k=config.get("top_k"),
                    top_n=config.get("top_n"),
                )
                evidence_texts = [block.content for block in result.context.evidence]
                evidence_ids = [block.identity.chunk_id for block in result.context.evidence]
                answer = None if result.generation is None else result.generation.text
                provided = _normalize_annotations(row.get("claim_annotations"))
                if provided:
                    annotations = provided
                    annotation_source = "dataset"
                else:
                    annotations = [
                        {"text": claim, "label": "indeterminate"}
                        for claim in split_claims(answer or "")
                    ]
                    annotation_source = "sentence_split_unlabeled"
                    if config.get("optional_llm_judge"):
                        # No judge implementation in-repo; keep indeterminate.
                        annotation_source = "sentence_split_unlabeled_judge_unavailable"
                counts = _counts(annotations)
                bundle = hallucination_bundle(counts)
                heuristic_flags = []
                for item in annotations:
                    flag = _lexical_overlap_supported(item["text"], evidence_texts)
                    heuristic_flags.append(flag)
                record.update(
                    {
                        "trace_id": result.trace_id,
                        "is_degraded": result.is_degraded,
                        "generation_is_none": result.generation is None,
                        "answer": redact_text(answer, redact),
                        "evidence_chunk_ids": evidence_ids,
                        "evidence_text": [
                            redact_text(text, redact) for text in evidence_texts
                        ],
                        "annotation_source": annotation_source,
                        "claims": annotations,
                        "claim_counts": counts,
                        "metrics": bundle,
                        "lexical_support_heuristic": heuristic_flags,
                        "note": (
                            "Heuristic True means high lexical overlap; False is never "
                            "emitted. Missing context ≠ contradicted."
                        ),
                    }
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                logger.exception("hallucination eval failed query_id=%s", row["query_id"])
            results.append(record)
    finally:
        await pipeline.shutdown()
    return results


def main() -> int:
    logger = configure_logging()
    args = parse_runner_args("RAGShield Phase 2 hallucination evaluation")
    config = merge_runtime_flags(load_config(args.config), args)
    apply_eval_env(config)
    dataset_path = resolve_path(config["dataset_path"])
    output_dir = resolve_path(config["output_dir"])
    output_dir = (
        (output_dir / "dry_run" / "hallucination")
        if not bool(config.get("execute"))
        else (output_dir / "hallucination")
    )
    rows = limit_rows(load_jsonl(dataset_path), config.get("max_cases"))
    execute = bool(config.get("execute"))

    if execute:
        per_query = asyncio.run(_run_live(config, rows, logger))
    else:
        logger.info("DRY-RUN hallucination (%s rows)", len(rows))
        per_query = []
        for row in rows:
            provided = _normalize_annotations(row.get("claim_annotations"))
            counts = _counts(provided)
            per_query.append(
                {
                    "query_id": row["query_id"],
                    "dry_run": True,
                    "would_call": "RAGPipeline.run then score claims vs context.evidence",
                    "dataset_claim_annotations": provided,
                    "claim_counts": counts,
                    "metrics": hallucination_bundle(counts) if provided else None,
                    "note": "Unlabeled claims must stay indeterminate.",
                }
            )

    n_human_claims = sum(
        1
        for row in per_query
        if row.get("annotation_source") == "dataset"
        or row.get("dataset_claim_annotations")
    )
    determined = coverage_from_per_query(
        per_query,
        "answer_level_hallucination_rate",
        exclude_reason="no_determined_claims_all_indeterminate_or_unlabelled",
        nested="metrics",
    )
    support = coverage_from_per_query(
        per_query,
        "claim_support_rate",
        exclude_reason="no_claim_metrics",
        nested="metrics",
    )
    aggregates = {
        "n_rows": len(per_query),
        "n_errors": sum(1 for row in per_query if row.get("error")),
        "n_queries_with_human_claim_labels": n_human_claims,
        "mean_claim_support_rate": support["mean"],
        "mean_claim_support_rate_n": support["n_eligible"],
        "mean_claim_support_rate_n_excluded": support["n_excluded"],
        "mean_unsupported_claim_rate": mean_ignore_none(
            [(row.get("metrics") or {}).get("unsupported_claim_rate") for row in per_query]
        ),
        "mean_contradiction_rate": mean_ignore_none(
            [(row.get("metrics") or {}).get("contradiction_rate") for row in per_query]
        ),
        "mean_indeterminate_claim_rate": mean_ignore_none(
            [
                (row.get("metrics") or {}).get("indeterminate_claim_rate")
                for row in per_query
            ]
        ),
        "mean_answer_level_hallucination_rate": determined["mean"],
        "mean_answer_level_hallucination_rate_n": determined["n_eligible"],
        "mean_answer_level_hallucination_rate_n_excluded": determined["n_excluded"],
        "mean_answer_level_hallucination_rate_excluded_query_ids": determined[
            "excluded_query_ids"
        ],
        "limitation": (
            "Indeterminate-only queries do not prove answers are hallucination-free. "
            "Fill claim_annotations. Auto-split claims stay indeterminate."
        ),
    }
    meta = {
        "suite": "hallucination",
        "generated_at_utc": utc_now(),
        "execute": execute,
        "dataset_sha256": file_sha256(dataset_path),
        "config": redact_mapping({k: v for k, v in config.items() if k != "notes"}),
        "label_set": sorted(VALID_LABELS),
        "denominator": (
            "Rates use all claims including indeterminate. "
            "Answer-level hallucination ignores indeterminate claims."
        ),
        "manual_annotations_override_sentence_split": True,
        "manual_claims_are_not_aligned_to_generated_answer": True,
        "annotation_evidence_ids_unused_in_rates": True,
        "lexical_overlap_heuristic_does_not_set_unsupported": True,
    }
    write_jsonl(output_dir / "per_query.jsonl", per_query)
    write_json(output_dir / "aggregate.json", {"meta": meta, "metrics": aggregates})
    write_markdown(
        output_dir / "report.md",
        "# Hallucination evaluation report\n\n"
        "Do not treat missing context as contradiction. Fill claim_annotations "
        "manually: supported, unsupported, contradicted, indeterminate. "
        "Sentence-split unlabeled claims stay indeterminate; that is not proof "
        "of faithfulness.\n",
    )
    logger.info("Wrote %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
