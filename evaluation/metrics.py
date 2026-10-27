from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from common import (
    HUMAN_RATING_FIELDS,
    HUMAN_RATING_MAX,
    HUMAN_RATING_PRECEDENCE,
    HUMAN_RATING_SCALE,
    human_annotation_status,
    parse_human_rating,
)

EXCLUSION_UNLABELED_CHUNKS = "unlabeled_no_relevant_chunk_ids"
EXCLUSION_NO_REFERENCE = "unlabeled_no_reference_answer"
EXCLUSION_NO_CLAIMS = "unlabeled_no_claim_annotations"
EXCLUSION_NO_ROUTE = "unlabeled_no_route_expected"
EXCLUSION_NO_ABSTENTION_LABEL = "unlabeled_no_abstention_expectation"
EXCLUSION_ERROR = "runner_error"
EXCLUSION_DRY_RUN = "dry_run_no_live_outputs"


def mean_ignore_none(values: Sequence[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def metric_coverage(
    values: Sequence[float | None],
    *,
    exclude_reason: str,
    query_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    eligible_ids: list[str] = []
    excluded_ids: list[str] = []
    nums: list[float] = []
    ids = list(query_ids) if query_ids is not None else [str(i) for i in range(len(values))]
    for qid, value in zip(ids, values):
        if value is None:
            excluded_ids.append(qid)
        else:
            eligible_ids.append(qid)
            nums.append(float(value))
    return {
        "mean": (sum(nums) / len(nums)) if nums else None,
        "n_eligible": len(nums),
        "n_excluded": len(excluded_ids),
        "eligible_query_ids": eligible_ids,
        "excluded_query_ids": excluded_ids,
        "excluded_reason": exclude_reason if excluded_ids else None,
    }


def _as_ids(ranked: Sequence[str] | None) -> list[str]:
    if not ranked:
        return []
    return [str(item) for item in ranked]


def _gold_set(gold: Sequence[str] | None) -> set[str] | None:
    if gold is None:
        return None
    return {str(item).strip() for item in gold if str(item).strip()}


def hit_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Sequence[str] | None,
    k: int,
) -> float | None:
    
    gold = _gold_set(gold_ids)
    if gold is None:
        return None
    if k < 1:
        raise ValueError("k must be >= 1")
    if not gold:
        return 0.0
    return float(any(item in gold for item in _as_ids(ranked_ids)[:k]))


def recall_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Sequence[str] | None,
    k: int,
) -> float | None:
    gold = _gold_set(gold_ids)
    if gold is None:
        return None
    if k < 1:
        raise ValueError("k must be >= 1")
    if not gold:
        return 0.0
    return len(gold.intersection(_as_ids(ranked_ids)[:k])) / len(gold)


def precision_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Sequence[str] | None,
    k: int,
) -> float | None:
    
    gold = _gold_set(gold_ids)
    if gold is None:
        return None
    if k < 1:
        raise ValueError("k must be >= 1")
    ranked = _as_ids(ranked_ids)[:k]
    return len(gold.intersection(ranked)) / k


def mrr_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Sequence[str] | None,
    k: int,
) -> float | None:
    gold = _gold_set(gold_ids)
    if gold is None:
        return None
    if k < 1:
        raise ValueError("k must be >= 1")
    if not gold:
        return 0.0
    for rank, item in enumerate(_as_ids(ranked_ids)[:k], start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def _grades_for_ids(
    gold_ids: Sequence[str] | None,
    relevance_judgments: Sequence[Mapping[str, Any]] | None,
) -> dict[str, float]:
    grades: dict[str, float] = {}
    if relevance_judgments:
        for item in relevance_judgments:
            cid = str(item.get("chunk_id") or item.get("id") or "").strip()
            if not cid:
                continue
            if "grade" not in item and "relevance" not in item:
                continue
            raw = item.get("grade", item.get("relevance"))
            grades[cid] = float(raw)
    elif gold_ids:
        for cid in gold_ids:
            text = str(cid).strip()
            if text:
                grades[text] = 1.0
    return grades


def ndcg_at_k(
    ranked_ids: Sequence[str],
    gold_ids: Sequence[str] | None = None,
    k: int = 5,
    relevance_judgments: Sequence[Mapping[str, Any]] | None = None,
) -> float | None:
  
    if k < 1:
        raise ValueError("k must be >= 1")
    grades = _grades_for_ids(gold_ids, relevance_judgments)
    if not grades:
        return None

    def dcg(order: Sequence[str]) -> float:
        total = 0.0
        for rank, item in enumerate(order[:k], start=1):
            rel = grades.get(item, 0.0)
            total += (2.0**rel - 1.0) / math.log2(rank + 1)
        return total

    ideal = sorted(grades, key=lambda cid: grades[cid], reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0:
        return None
    return dcg(_as_ids(ranked_ids)) / ideal_dcg


def ranking_bundle(
    ranked_ids: Sequence[str],
    gold_ids: Sequence[str] | None,
    k: int,
    relevance_judgments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, float | None]:
   
    return {
        f"hit@{k}": hit_at_k(ranked_ids, gold_ids, k),
        f"recall@{k}": recall_at_k(ranked_ids, gold_ids, k),
        f"precision@{k}": precision_at_k(ranked_ids, gold_ids, k),
        f"mrr@{k}": mrr_at_k(ranked_ids, gold_ids, k),
        f"ndcg@{k}": ndcg_at_k(ranked_ids, gold_ids=gold_ids, k=k),
        f"ndcg_graded@{k}": ndcg_at_k(
            ranked_ids,
            gold_ids=None,
            k=k,
            relevance_judgments=relevance_judgments,
        )
        if relevance_judgments
        else None,
    }


def gold_ranks(ranked_ids: Sequence[str], gold_ids: Sequence[str] | None) -> dict[str, int | None]:
    gold = _gold_set(gold_ids) or set()
    order = _as_ids(ranked_ids)
    out: dict[str, int | None] = {}
    for gid in sorted(gold):
        try:
            out[gid] = order.index(gid) + 1
        except ValueError:
            out[gid] = None
    return out


def claim_rate(counts: Mapping[str, int], key: str) -> float | None:
    total = sum(max(0, int(v)) for v in counts.values())
    if total == 0:
        return None
    return int(counts.get(key, 0)) / total


def hallucination_bundle(counts: Mapping[str, int]) -> dict[str, float | None]:
    supported = int(counts.get("supported", 0))
    unsupported = int(counts.get("unsupported", 0))
    contradicted = int(counts.get("contradicted", 0))
    indeterminate = int(counts.get("indeterminate", 0))
    total = supported + unsupported + contradicted + indeterminate
    determined = supported + unsupported + contradicted
    answer_hallucinated = None
    if determined > 0:
        answer_hallucinated = float((unsupported + contradicted) > 0)
    return {
        "claim_support_rate": claim_rate(counts, "supported"),
        "unsupported_claim_rate": claim_rate(counts, "unsupported"),
        "contradiction_rate": claim_rate(counts, "contradicted"),
        "indeterminate_claim_rate": claim_rate(counts, "indeterminate"),
        "answer_level_hallucination_rate": answer_hallucinated,
        "n_claims": float(total) if total else None,
        "n_determined_claims": float(determined) if determined else None,
        "eligible_for_answer_level": determined > 0,
    }


_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    return _WS.sub(" ", (text or "").casefold()).strip()


def exact_match(pred: str | None, ref: str | None) -> float | None:
    if pred is None or ref is None:
        return None
    left = normalize_answer(pred)
    right = normalize_answer(ref)
    if not left or not right:
        return None
    return float(left == right)


def token_overlap(pred: str | None, ref: str | None) -> dict[str, float | None]:
    if pred is None or ref is None:
        return {"token_precision": None, "token_recall": None, "token_f1": None}
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()
    if not pred_tokens or not ref_tokens:
        return {"token_precision": None, "token_recall": None, "token_f1": None}
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    overlap = len(pred_set & ref_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def query_answer_overlap(query: str | None, answer: str | None) -> float | None:
  
    if query is None or answer is None:
        return None
    q_tokens = [t for t in normalize_answer(query).split() if len(t) > 2]
    a_tokens = set(normalize_answer(answer).split())
    if not q_tokens or not a_tokens:
        return None
    return sum(t in a_tokens for t in q_tokens) / len(q_tokens)


def human_dimension_stats(
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, float | int | None]:
    values: list[int] = []
    missing = 0
    for row in records:
        parsed = parse_human_rating(
            field,
            row.get(field),
            query_id=None if row.get("query_id") is None else str(row.get("query_id")),
        )
        if parsed is None:
            missing += 1
        else:
            values.append(parsed)
    mean = (sum(values) / len(values)) if values else None
    normalized_pct = None if mean is None else (mean / HUMAN_RATING_MAX) * 100.0
    return {
        "mean": mean,
        "n": len(values),
        "n_missing": missing,
        "normalized_pct": normalized_pct,
    }


def human_rating_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status = human_annotation_status(list(records))
    by_dimension: dict[str, dict[str, float | int | None]] = {}
    payload: dict[str, Any] = {
        "human_metrics_filled": status["human_metrics_filled"],
        "n_fully_annotated": status["n_fully_annotated"],
        "n_partial": status["n_partial"],
        "n_unannotated": status["n_unannotated"],
        "human_annotation_status": status,
        "human_rating_scale": HUMAN_RATING_SCALE,
        "human_rating_precedence": list(HUMAN_RATING_PRECEDENCE),
        "human_normalized_pct_note": (
            "normalized_pct is (mean / 2) * 100, a descriptive rescaling of the "
            "0-2 human ordinal scale. It is not clinical accuracy and is not an "
            "automated lexical metric."
        ),
    }
    for field in HUMAN_RATING_FIELDS:
        stats = human_dimension_stats(records, field)
        by_dimension[field] = stats
        payload[f"{field}_mean"] = stats["mean"]
        payload[f"{field}_n"] = stats["n"]
        payload[f"{field}_n_missing"] = stats["n_missing"]
        payload[f"{field}_normalized_pct"] = stats["normalized_pct"]
    payload["by_dimension"] = by_dimension
    return payload


def key_fact_coverage(answer: str | None, key_facts: Sequence[str] | None) -> float | None:
    if answer is None or not key_facts:
        return None
    blob = normalize_answer(answer)
    facts = [normalize_answer(str(f)) for f in key_facts if str(f).strip()]
    if not facts or not blob:
        return None
    return sum(fact in blob for fact in facts) / len(facts)
