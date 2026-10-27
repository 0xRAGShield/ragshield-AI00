"""Deterministic synthetic IR/generation scoring. Does not call the RAG pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import gold_chunk_ids, load_jsonl
from metrics import ranking_bundle
from reporting import coverage_from_per_query

SYNTHETIC = Path(__file__).resolve().parents[1] / "datasets" / "synthetic_eval.jsonl"

# Manually assigned rankings (not model output).
PRE = {
    "SYN-HIT": ["synth:chunk:gold", "synth:chunk:other", "synth:chunk:noise"],
    "SYN-MISS": ["synth:chunk:a", "synth:chunk:b", "synth:chunk:c"],
    "SYN-UNLABELED": ["synth:chunk:a", "synth:chunk:b"],
    "SYN-EMPTY": ["synth:chunk:a"],
    "SYN-RERANK": ["synth:chunk:gold", "synth:chunk:other", "synth:chunk:noise"],
}
POST = dict(PRE)
POST["SYN-RERANK"] = ["synth:chunk:other", "synth:chunk:noise", "synth:chunk:gold"]


def _rows():
    return load_jsonl(SYNTHETIC)


def test_pre_post_same_set_different_order():
    assert set(PRE["SYN-RERANK"]) == set(POST["SYN-RERANK"])
    assert PRE["SYN-RERANK"] != POST["SYN-RERANK"]
    gold = ["synth:chunk:gold"]
    pre = ranking_bundle(PRE["SYN-RERANK"], gold, 1)
    post = ranking_bundle(POST["SYN-RERANK"], gold, 1)
    assert pre["hit@1"] == 1.0
    assert post["hit@1"] == 0.0
    assert post["hit@1"] - pre["hit@1"] == -1.0
    post5 = ranking_bundle(POST["SYN-RERANK"], gold, 5)
    assert post5["hit@5"] == 1.0


def test_missing_relevant_from_pool():
    gold = gold_chunk_ids(next(r for r in _rows() if r["query_id"] == "SYN-MISS"))
    ranked = PRE["SYN-MISS"]
    assert gold == ["synth:chunk:absent"]
    assert "synth:chunk:absent" not in ranked
    bundle = ranking_bundle(ranked, gold, 5)
    assert bundle["hit@5"] == 0.0
    assert bundle["recall@5"] == 0.0


def test_aggregate_denominators_exclude_unlabelled():
    per_query = []
    for row in _rows():
        gold = gold_chunk_ids(row)
        metrics = ranking_bundle(PRE[row["query_id"]], gold, 5)
        per_query.append({"query_id": row["query_id"], "metrics": metrics})
    hit = coverage_from_per_query(
        per_query, "hit@5", exclude_reason="unlabeled_no_relevant_chunk_ids", nested="metrics"
    )
    # HIT, MISS, EMPTY (known-empty scores 0), RERANK — unlabeled excluded.
    assert hit["n_eligible"] == 4
    assert hit["n_excluded"] == 1
    assert hit["excluded_query_ids"] == ["SYN-UNLABELED"]
    assert hit["mean"] == 0.5


def test_synthetic_hit_query_perfect_at_k1():
    row = next(r for r in _rows() if r["query_id"] == "SYN-HIT")
    bundle = ranking_bundle(PRE["SYN-HIT"], gold_chunk_ids(row), 1)
    assert bundle["hit@1"] == 1.0
    assert bundle["precision@1"] == 1.0
    assert bundle["mrr@1"] == 1.0
