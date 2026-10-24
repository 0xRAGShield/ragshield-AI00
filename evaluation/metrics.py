
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def mean_ignore_none(values: Sequence[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _as_ids(ranked: Sequence[str] | None) -> list[str]:
    if not ranked:
        return []
    return [str(item) for item in ranked]


def _gold_set(gold: Sequence[str] | None) -> set[str] | None:
    if gold is None:
        return None
    cleaned = [str(item).strip() for item in gold if str(item).strip()]
    if not cleaned:
        return None
    return set(cleaned)


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
        f"ndcg@{k}": ndcg_at_k(
            ranked_ids,
            gold_ids=gold_ids,
            k=k,
            relevance_judgments=relevance_judgments,
        ),
    }


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
    }
