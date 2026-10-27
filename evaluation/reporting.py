"""Shared aggregate/exclusion payloads for suite reports."""

from __future__ import annotations

from typing import Any

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
        },
    }
    if extra:
        payload.update(extra)
    return payload
