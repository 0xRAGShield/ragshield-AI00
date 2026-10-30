"""Unit tests for evaluation.metrics using synthetic rankings only."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import expected_abstention_label, gold_chunk_ids, looks_like_abstention
from metrics import (
    exact_match,
    gold_ranks,
    hallucination_bundle,
    hit_at_k,
    key_fact_coverage,
    mean_ignore_none,
    metric_coverage,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    query_answer_overlap,
    ranking_bundle,
    recall_at_k,
    token_overlap,
)


def test_unlabeled_returns_none():
    ranked = ["a", "b"]
    assert hit_at_k(ranked, None, 5) is None
    assert recall_at_k(ranked, None, 1) is None
    assert precision_at_k(ranked, None, 1) is None
    assert mrr_at_k(ranked, None, 1) is None
    assert ndcg_at_k(ranked, gold_ids=None, k=1) is None
    assert gold_chunk_ids({"relevant_chunk_ids": []}) is None


def test_known_empty_gold_is_labelled_zero_hit():
    ranked = ["a", "b"]
    assert gold_chunk_ids({"no_relevant_results": True, "relevant_chunk_ids": []}) == []
    assert hit_at_k(ranked, [], 5) == 0.0
    assert recall_at_k(ranked, [], 5) == 0.0
    assert precision_at_k(ranked, [], 5) == 0.0
    assert mrr_at_k(ranked, [], 5) == 0.0


def test_all_relevant_retrieved():
    ranked = ["g1", "g2", "x"]
    gold = ["g1", "g2"]
    assert hit_at_k(ranked, gold, 3) == 1.0
    assert recall_at_k(ranked, gold, 3) == 1.0
    assert precision_at_k(ranked, gold, 3) == 2 / 3
    assert mrr_at_k(ranked, gold, 3) == 1.0


def test_no_relevant_retrieved():
    ranked = ["a", "b", "c"]
    gold = ["g1"]
    assert hit_at_k(ranked, gold, 3) == 0.0
    assert recall_at_k(ranked, gold, 3) == 0.0
    assert precision_at_k(ranked, gold, 3) == 0.0
    assert mrr_at_k(ranked, gold, 3) == 0.0


def test_relevant_at_rank_1():
    ranked = ["g1", "x"]
    gold = ["g1"]
    assert hit_at_k(ranked, gold, 1) == 1.0
    assert mrr_at_k(ranked, gold, 5) == 1.0


def test_relevant_at_lower_rank():
    ranked = ["u", "g1", "x"]
    gold = ["g1", "g2"]
    assert hit_at_k(ranked, gold, 2) == 1.0
    assert hit_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert mrr_at_k(ranked, gold, 3) == 0.5


def test_precision_uses_cutoff_when_fewer_than_k():
    ranked = ["g1"]
    gold = ["g1"]
    assert precision_at_k(ranked, gold, 5) == 1 / 5
    assert hit_at_k(ranked, gold, 5) == 1.0
    assert recall_at_k(ranked, gold, 5) == 1.0


def test_duplicate_candidate_ids():
    ranked = ["g1", "g1", "x"]
    gold = ["g1"]
    assert hit_at_k(ranked, gold, 2) == 1.0
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert gold_ranks(ranked, gold)["g1"] == 1


def test_ndcg_binary_matches_hit_eligibility():
    ranked = ["b", "a"]
    binary = ranking_bundle(ranked, ["a"], 2, relevance_judgments=None)
    assert binary["hit@2"] == 1.0
    assert binary["ndcg@2"] is not None
    assert binary["ndcg_graded@2"] is None
    unlabeled = ranking_bundle(ranked, None, 2, [{"chunk_id": "a", "grade": 1}])
    assert unlabeled["hit@2"] is None
    assert unlabeled["ndcg@2"] is None
    assert unlabeled["ndcg_graded@2"] is not None


def test_ndcg_grade_zero_only_is_ineligible():
    ranked = ["a"]
    assert ndcg_at_k(ranked, k=1, relevance_judgments=[{"chunk_id": "a", "grade": 0}]) is None


def test_mean_ignore_none_and_coverage():
    assert mean_ignore_none([1.0, None, 0.0]) == 0.5
    assert mean_ignore_none([None]) is None
    cov = metric_coverage(
        [1.0, None, 0.0],
        exclude_reason="unlabeled_no_relevant_chunk_ids",
        query_ids=["D02", "A01", "D03"],
    )
    assert cov["n_eligible"] == 2
    assert cov["n_excluded"] == 1
    assert cov["excluded_query_ids"] == ["A01"]
    assert cov["mean"] == 0.5


def test_hallucination_unsupported_and_indeterminate():
    mixed = hallucination_bundle(
        {"supported": 2, "unsupported": 1, "contradicted": 0, "indeterminate": 1}
    )
    assert mixed["claim_support_rate"] == 0.5
    assert mixed["unsupported_claim_rate"] == 0.25
    assert mixed["indeterminate_claim_rate"] == 0.25
    assert mixed["answer_level_hallucination_rate"] == 1.0
    all_indeterminate = hallucination_bundle(
        {"supported": 0, "unsupported": 0, "contradicted": 0, "indeterminate": 3}
    )
    assert all_indeterminate["claim_support_rate"] == 0.0
    assert all_indeterminate["indeterminate_claim_rate"] == 1.0
    assert all_indeterminate["answer_level_hallucination_rate"] is None
    empty = hallucination_bundle(
        {"supported": 0, "unsupported": 0, "contradicted": 0, "indeterminate": 0}
    )
    assert empty["claim_support_rate"] is None
    contradicted = hallucination_bundle(
        {"supported": 0, "unsupported": 0, "contradicted": 1, "indeterminate": 0}
    )
    assert contradicted["contradiction_rate"] == 1.0
    assert contradicted["answer_level_hallucination_rate"] == 1.0


def test_empty_answers_and_references():
    assert exact_match("", "ref") is None
    assert exact_match("ans", "") is None
    assert exact_match(None, "ref") is None
    assert token_overlap("", "ref")["token_f1"] is None
    assert token_overlap("same words", "same words")["token_f1"] == 1.0
    assert exact_match("  High Blood  ", "high blood") == 1.0
    assert query_answer_overlap("high blood pressure", "pressure is high") is not None
    assert key_fact_coverage("force against vessel walls", ["force against vessel walls"]) == 1.0
    assert key_fact_coverage("unrelated", ["force against vessel walls"]) == 0.0
    assert key_fact_coverage("text", []) is None


def test_normalize_annotations_requires_explicit_label():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hallucination"))
    from evaluate_hallucination import _normalize_annotations

    raw = [
        {"claim": "no label", "support_level": "direct"},
        {"claim": "ok", "label": "supported"},
        {"claim": "bad label", "label": "direct"},
    ]
    out = _normalize_annotations(raw)
    assert out == [{"text": "ok", "label": "supported"}]


def _protocol_ndcg(ranked: list[str], grades: dict[str, float], k: int) -> float:
    """Independent Burges nDCG used as the expected value, not the implementation."""

    def dcg(order: list[str]) -> float:
        total = 0.0
        for rank, item in enumerate(order[:k], start=1):
            total += (2.0 ** grades.get(item, 0.0) - 1.0) / math.log2(rank + 1)
        return total

    ideal = sorted(grades, key=lambda doc: grades[doc], reverse=True)
    ideal_dcg = dcg(ideal)
    assert ideal_dcg > 0
    return dcg(ranked) / ideal_dcg


def test_empty_ranking_is_zero_when_gold_exists():
    gold = ["g1"]
    assert hit_at_k([], gold, 5) == 0.0
    assert recall_at_k([], gold, 5) == 0.0
    assert precision_at_k([], gold, 5) == 0.0
    assert mrr_at_k([], gold, 5) == 0.0
    assert ndcg_at_k([], gold_ids=gold, k=5) == 0.0


def test_fewer_than_k_and_no_relevant_documents():
    assert precision_at_k(["g1"], ["g1"], 5) == 0.2
    assert recall_at_k(["g1"], ["g1", "g2"], 1) == 0.5
    assert hit_at_k(["a", "b"], ["missing"], 3) == 0.0
    assert mrr_at_k(["a", "b"], ["missing"], 3) == 0.0
    assert ndcg_at_k(["a", "b"], gold_ids=["missing"], k=3) == 0.0


def test_duplicate_ids_keep_set_precision_and_repeated_ndcg_gain():
    ranked = ["g1", "g1", "x"]
    gold = ["g1"]
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert recall_at_k(ranked, gold, 2) == 1.0
    assert hit_at_k(ranked, gold, 2) == 1.0
    assert mrr_at_k(ranked, gold, 2) == 1.0
    score = ndcg_at_k(ranked, gold_ids=gold, k=2)
    first_only = _protocol_ndcg(["g1", "__unjudged__"], {"g1": 1.0}, 2)
    assert math.isclose(score, first_only)
    assert 0.0 <= score <= 1.0
    assert score == 1.0

    later_kept = ndcg_at_k(["g1", "g1", "g2"], gold_ids=["g1", "g2"], k=3)
    expected_later = _protocol_ndcg(
        ["g1", "__unjudged__", "g2"], {"g1": 1.0, "g2": 1.0}, 3
    )
    compacted = _protocol_ndcg(["g1", "g2"], {"g1": 1.0, "g2": 1.0}, 3)
    assert math.isclose(later_kept, expected_later)
    assert later_kept < compacted
    assert 0.0 <= later_kept <= 1.0


def test_graded_ndcg_known_values_and_missing_judgments():
    judgments = [
        {"chunk_id": "a", "grade": 1},
        {"chunk_id": "b", "grade": 3},
        {"chunk_id": "c", "grade": 0},
    ]
    grades = {"a": 1.0, "b": 3.0, "c": 0.0}
    ranked = ["a", "b", "c"]
    score = ndcg_at_k(ranked, k=3, relevance_judgments=judgments)
    assert math.isclose(score, _protocol_ndcg(ranked, grades, 3))
    assert math.isclose(score, 0.7098097413968655)

    fractional = [{"chunk_id": "a", "grade": 0.5}, {"chunk_id": "b", "grade": 1}]
    fractional_score = ndcg_at_k(
        ["a", "b"], k=2, relevance_judgments=fractional
    )
    assert math.isclose(
        fractional_score, _protocol_ndcg(["a", "b"], {"a": 0.5, "b": 1.0}, 2)
    )

    unjudged_first = ndcg_at_k(
        ["unjudged", "a"],
        k=2,
        relevance_judgments=[{"chunk_id": "a", "grade": 1}],
    )
    assert math.isclose(
        unjudged_first, _protocol_ndcg(["unjudged", "a"], {"a": 1.0}, 2)
    )
    assert ndcg_at_k(["a"], k=2, relevance_judgments=None, gold_ids=None) is None
    assert ndcg_at_k(
        ["a"],
        k=1,
        relevance_judgments=[{"chunk_id": "a", "grade": 0}, {"chunk_id": "b", "grade": -1}],
    ) is None


def test_token_overlap_matches_sklearn_confusion_counts():
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    overlap = token_overlap("dog dog cat", "dog bird")
    labels = ["bird", "cat", "dog"]
    y_true = [1, 0, 1]
    y_pred = [0, 1, 1]
    assert overlap["token_precision"] == precision_score(y_true, y_pred, zero_division=0)
    assert overlap["token_recall"] == recall_score(y_true, y_pred, zero_division=0)
    assert overlap["token_f1"] == f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    assert (tn, fp, fn, tp) == (0, 1, 1, 1)
    assert overlap["token_precision"] == 0.5
    assert overlap["token_recall"] == 0.5
    assert math.isclose(overlap["token_f1"], 0.5)
    assert token_overlap("alpha beta", "gamma delta")["token_f1"] == 0.0
    assert token_overlap("same same", "same")["token_precision"] == 1.0


def test_k_must_be_positive():
    for fn in (hit_at_k, recall_at_k, precision_at_k, mrr_at_k):
        try:
            fn(["a"], ["a"], 0)
        except ValueError as exc:
            assert "k must be >= 1" in str(exc)
        else:
            raise AssertionError("expected ValueError")
    try:
        ndcg_at_k(["a"], gold_ids=["a"], k=0)
    except ValueError as exc:
        assert "k must be >= 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_abstention_expected_versus_not():
    assert expected_abstention_label({"expected_abstention": True}) is True
    assert expected_abstention_label({"answerable": False}) is True
    assert expected_abstention_label({"answerable": True}) is False
    assert expected_abstention_label({}) is None
    assert looks_like_abstention("There is not enough evidence in the sources.")
    assert not looks_like_abstention("High blood pressure is elevated force.")
