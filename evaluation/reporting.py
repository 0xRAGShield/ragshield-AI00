
from __future__ import annotations

from typing import Any

from common import HUMAN_RATING_FIELDS, HUMAN_RATING_MAX
from metrics import metric_coverage


def coverage_from_per_query(
    per_query: list[dict[str, Any]],
    metric_key: str,
    *,
    exclude_reason: str,
    nested: str | None = None,
) -> dict[str, Any]:
    values = []
    ids = []
    for row in per_query:
        ids.append(str(row.get("query_id")))
        blob = row.get(nested) if nested else row
        if not isinstance(blob, dict):
            values.append(None)
        else:
            values.append(blob.get(metric_key))
    return metric_coverage(values, exclude_reason=exclude_reason, query_ids=ids)


def suite_meta(
    *,
    suite: str,
    execute: bool,
    dataset_path: str,
    dataset_sha256: str | None,
    n_rows: int,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "suite": suite,
        "execute": execute,
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_sha256,
        "n_rows": n_rows,
        "config": config,
        "metric_notes": {
            "unlabelled_is_not_zero": True,
            "precision_at_k_denominator": "cutoff k",
            "ndcg_binary_matches_hit_eligibility": True,
            "ranking_library": "ranx",
            "ranking_measures": [
                "hit_rate@k",
                "recall@k",
                "precision@k",
                "mrr@k",
                "ndcg_burges@k",
            ],
            "token_overlap_library": "scikit-learn",
            "token_overlap_measures": [
                "precision_score",
                "recall_score",
                "f1_score",
            ],
            "metric_definitions": "evaluation/METRICS.md",
        },
    }
    if extra:
        payload.update(extra)
    return payload


def human_evaluation_markdown_table(aggregates: dict[str, Any]) -> str:
    lines = [
        "## Human evaluation (0–2 ordinal scale)",
        "",
        "These scores are **human ratings**, not automated lexical metrics and "
        "not clinical accuracy. Missing ratings (`null`) are excluded from means "
        "and are not treated as zero. "
        f"`normalized_pct` is (mean / {HUMAN_RATING_MAX}) × 100, a descriptive "
        "rescaling only.",
        "",
        "| Dimension | Mean / 2 | Annotated n | Missing n | Normalized % (descriptive) |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in HUMAN_RATING_FIELDS:
        mean = aggregates.get(f"{field}_mean")
        n_ann = aggregates.get(f"{field}_n")
        n_miss = aggregates.get(f"{field}_n_missing")
        pct = aggregates.get(f"{field}_normalized_pct")
        mean_text = "—" if mean is None else f"{float(mean):.4f}"
        pct_text = "—" if pct is None else f"{float(pct):.1f}"
        lines.append(
            f"| `{field}` | {mean_text} | {n_ann} | {n_miss} | {pct_text} |"
        )
    filled = aggregates.get("human_metrics_filled")
    lines.extend(
        [
            "",
            f"- n_fully_annotated: {aggregates.get('n_fully_annotated')}",
            f"- n_partial: {aggregates.get('n_partial')}",
            f"- n_unannotated: {aggregates.get('n_unannotated')}",
            f"- human_metrics_filled: {filled}",
            "",
        ]
    )
    return "\n".join(lines)


def generation_report_markdown(aggregates: dict[str, Any], *, execute: bool) -> str:
    return (
        "# Generation evaluation report\n\n"
        f"execute={execute}\n\n"
        "Primary scores are **human** 0/1/2 fields (`null` = unannotated). "
        "token_f1 (scikit-learn token-set precision/recall/F1) and exact_match "
        "are lexical overlap only and are **not** clinical correctness.\n\n"
        + human_evaluation_markdown_table(aggregates)
        + "Human ratings are joined by `query_id` from the dataset JSONL. "
        "`per_query_human_review.jsonl` is an optional non-null override when "
        "configured or present; it is not required. An existing "
        "`per_query.jsonl` only fills leftover nulls and cannot replace a "
        "valid dataset score with null. This runner never invents ratings.\n\n"
        "## Automated lexical metrics\n\n"
        f"- mean_token_f1: {aggregates.get('mean_token_f1')}\n"
        f"- mean_lexical_exact_match: {aggregates.get('mean_lexical_exact_match')}\n"
        f"- mean_abstention_match: {aggregates.get('mean_abstention_match')}\n"
        f"- mean_relevancy_lexical_diagnostic: "
        f"{aggregates.get('mean_relevancy_lexical_diagnostic')}\n"
    )
