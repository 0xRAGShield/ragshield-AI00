"""Unit tests for evaluation.metrics using synthetic rankings only.

Do not import app.*, Qdrant, or models. These tests were not executed
during scaffolding.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import (
    hallucination_bundle,
    hit_at_k,
    mean_ignore_none,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_unlabeled_returns_none():
    ranked = ["a", "b"]
    assert hit_at_k(ranked, [], 5) is None
    assert hit_at_k(ranked, None, 5) is None
    assert recall_at_k(ranked, [], 1) is None
    assert precision_at_k(ranked, [], 1) is None
    assert mrr_at_k(ranked, [], 1) is None
    assert ndcg_at_k(ranked, gold_ids=[], k=1) is None


def test_hit_recall_precision_mrr():
    ranked = ["u", "g1", "x"]
    gold = ["g1", "g2"]
    assert hit_at_k(ranked, gold, 2) == 1.0
    assert hit_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert mrr_at_k(ranked, gold, 3) == 0.5


def test_ndcg_binary_and_graded():
    ranked = ["b", "a"]
    assert ndcg_at_k(ranked, gold_ids=["a"], k=2) is not None
    graded = ndcg_at_k(
        ranked,
        k=2,
        relevance_judgments=[
            {"chunk_id": "a", "grade": 3},
            {"chunk_id": "b", "grade": 1},
        ],
    )
    assert graded is not None
    assert 0.0 <= graded <= 1.0


def test_mean_ignore_none():
    assert mean_ignore_none([1.0, None, 0.0]) == 0.5
    assert mean_ignore_none([None]) is None


def test_hallucination_denominators():
    counts = {
        "supported": 2,
        "unsupported": 1,
        "contradicted": 0,
        "indeterminate": 1,
    }
    bundle = hallucination_bundle(counts)
    assert bundle["claim_support_rate"] == 0.5
    assert bundle["unsupported_claim_rate"] == 0.25
    assert bundle["indeterminate_claim_rate"] == 0.25
    assert bundle["answer_level_hallucination_rate"] == 1.0
    empty = hallucination_bundle(
        {"supported": 0, "unsupported": 0, "contradicted": 0, "indeterminate": 0}
    )
    assert empty["claim_support_rate"] is None
    assert empty["answer_level_hallucination_rate"] is None
