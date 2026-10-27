"""Human-rating join and aggregation for generation evaluation. No models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    HUMAN_RATING_FIELDS,
    attach_human_ratings,
    human_annotation_status,
    load_human_rating_index,
    load_jsonl,
    parse_human_rating,
    write_json,
    write_jsonl,
    write_markdown,
)
from generation.evaluate_generation import (
    finalize_generation_records,
    seed_generation_record,
)
from metrics import human_rating_aggregates
from reporting import generation_report_markdown

FULL = {field: 2 for field in HUMAN_RATING_FIELDS}
EMPTY = {field: None for field in HUMAN_RATING_FIELDS}
TEMPLATE = Path(__file__).resolve().parents[1] / "datasets" / "evaluation_template.jsonl"
EXPECTED_TEMPLATE_RATINGS = {
    "A01": (2, 1, 2, 2, 2, 2),
    "A02": (1, 1, 2, 2, 2, 2),
    "D01": (2, 2, 2, 2, 2, 2),
    "D02": (2, 2, 2, 2, 2, 1),
    "D03": (2, 1, 2, 2, 2, 1),
    "C01": (2, 2, 2, 2, 2, 2),
    "C02": (2, 2, 2, 2, 2, 2),
    "C03": (2, 2, 2, 2, 2, 2),
    "C04": (1, 1, 2, 1, 2, 2),
    "C05": (1, 1, 2, 1, 2, 2),
    "P01": (1, 2, 0, 0, 2, 2),
    "P02": (2, 2, 2, 2, 2, 2),
    "P03": (2, 2, 2, 2, 2, 2),
    "H01": (1, 1, 2, 1, 2, 2),
    "H02": (1, 1, 2, 1, 2, 2),
    "H03": (1, 2, 2, 0, 2, 2),
    "ALL01": (2, 1, 2, 2, 2, 2),
    "ALL02": (0, 0, 1, 0, 0, 2),
    "ALL03": (1, 1, 2, 0, 0, 1),
}


def _row(query_id: str, ratings: dict | None = None, **extra) -> dict:
    payload = {"query_id": query_id, "query": f"q-{query_id}", **EMPTY}
    if ratings:
        payload.update(ratings)
    payload.update(extra)
    return payload


def test_all_six_ratings_populated_for_all_queries():
    records = [
        _row("A01", FULL),
        _row("A02", FULL),
        _row("D01", {**FULL, "human_correctness": 1}),
    ]
    agg = human_rating_aggregates(records)
    assert agg["n_fully_annotated"] == 3
    assert agg["n_partial"] == 0
    assert agg["n_unannotated"] == 0
    assert agg["human_metrics_filled"] is True
    assert agg["human_correctness_n"] == 3
    assert agg["human_correctness_n_missing"] == 0
    assert agg["human_correctness_mean"] == pytest.approx(5 / 3)
    assert agg["human_faithfulness_mean"] == 2.0
    assert agg["human_correctness_normalized_pct"] == pytest.approx((5 / 3) / 2 * 100)


def test_all_ratings_missing():
    records = [_row("A01"), _row("A02")]
    agg = human_rating_aggregates(records)
    assert agg["n_fully_annotated"] == 0
    assert agg["n_unannotated"] == 2
    assert agg["human_metrics_filled"] is False
    assert agg["human_correctness_mean"] is None
    assert agg["human_correctness_n"] == 0
    assert agg["human_correctness_n_missing"] == 2
    assert agg["human_correctness_normalized_pct"] is None


def test_partially_annotated_queries():
    records = [
        _row("A01", {"human_correctness": 2, "human_faithfulness": 1}),
        _row("A02", FULL),
        _row("A03"),
    ]
    status = human_annotation_status(records)
    assert status["n_fully_annotated"] == 1
    assert status["n_partial"] == 1
    assert status["n_unannotated"] == 1
    agg = human_rating_aggregates(records)
    assert agg["human_correctness_n"] == 2
    assert agg["human_correctness_n_missing"] == 1
    assert agg["human_relevance_n"] == 1
    assert agg["human_metrics_filled"] is False


def test_mixed_null_and_valid_ratings_exclude_null_from_mean():
    records = [
        _row("A01", {"human_correctness": 0}),
        _row("A02", {"human_correctness": 2}),
        _row("A03"),
    ]
    agg = human_rating_aggregates(records)
    assert agg["human_correctness_mean"] == 1.0
    assert agg["human_correctness_n"] == 2
    assert agg["human_correctness_n_missing"] == 1
    assert 0 in [row["human_correctness"] for row in records]


@pytest.mark.parametrize("bad", [3, -1, 2.0, "2", True, False, 99])
def test_invalid_rating_values_rejected(bad):
    with pytest.raises(ValueError, match="expected null or integer 0, 1, or 2"):
        parse_human_rating("human_correctness", bad, query_id="A01")


def test_seed_preserves_ratings_in_dry_run_and_live_records():
    row = _row("A01", {**FULL, "human_correctness": 1, "reference_answer": "ref"})
    dry = seed_generation_record(row, dry_run=True)
    live = seed_generation_record(row, dry_run=False)
    assert dry["dry_run"] is True
    assert "dry_run" not in live
    for record in (dry, live):
        assert record["query_id"] == "A01"
        assert record["human_correctness"] == 1
        assert record["human_medical_safety"] == 2
        for field in HUMAN_RATING_FIELDS:
            assert field in record


def test_join_by_query_id_ignores_source_order():
    dataset = [_row("A01", FULL), _row("B02", {"human_correctness": 1})]
    review = {"B02": _row("B02", {**EMPTY, "human_faithfulness": 2})}
    records = [
        seed_generation_record(dataset[1], dry_run=True),
        seed_generation_record(dataset[0], dry_run=True),
    ]
    attached = attach_human_ratings(
        records,
        dataset_by_id={row["query_id"]: row for row in dataset},
        review_by_id=review,
    )
    by_id = {row["query_id"]: row for row in attached}
    assert [row["query_id"] for row in attached] == ["B02", "A01"]
    assert by_id["A01"]["human_correctness"] == 2
    assert by_id["B02"]["human_correctness"] == 1
    assert by_id["B02"]["human_faithfulness"] == 2
    assert by_id["B02"]["human_rating_sources"]["human_faithfulness"] == "review"
    assert by_id["B02"]["human_rating_sources"]["human_correctness"] == "dataset"


def test_review_file_fills_null_dataset_fields(tmp_path: Path):
    dataset_rows = [_row("A01"), _row("A02")]
    review = tmp_path / "per_query_human_review.jsonl"
    review.write_text(
        "# human review\n"
        "{\n"
        '  "query_id": "A02",\n'
        '  "human_correctness": 1,\n'
        '  "human_faithfulness": 2,\n'
        '  "human_relevance": 2,\n'
        '  "human_completeness": 1,\n'
        '  "human_abstention": 2,\n'
        '  "human_medical_safety": 2\n'
        "}\n"
        '{"query_id": "A01", "human_correctness": 2, "human_faithfulness": 2, '
        '"human_relevance": 2, "human_completeness": 2, "human_abstention": 2, '
        '"human_medical_safety": 2}\n',
        encoding="utf-8",
    )
    records = [seed_generation_record(row, dry_run=True) for row in dataset_rows]
    attached = finalize_generation_records(
        records,
        dataset_rows,
        output_dir=tmp_path,
        config={},
    )
    by_id = {row["query_id"]: row for row in attached}
    assert by_id["A01"]["human_correctness"] == 2
    assert by_id["A02"]["human_completeness"] == 1
    assert all(
        by_id["A01"]["human_rating_sources"][field] == "review"
        for field in HUMAN_RATING_FIELDS
    )
    loaded = load_human_rating_index(review, source="review")
    assert set(loaded) == {"A01", "A02"}


def test_fresh_generation_preserves_existing_annotations(tmp_path: Path):
    dataset_rows = [_row("A01"), _row("A02")]
    previous = [
        {
            "query_id": "A01",
            "answer": "old-answer",
            **FULL,
        }
    ]
    write_jsonl(tmp_path / "per_query.jsonl", previous)
    fresh = [
        {**seed_generation_record(dataset_rows[0], dry_run=False), "answer": "new-answer"},
        {**seed_generation_record(dataset_rows[1], dry_run=False), "answer": "also-new"},
    ]
    attached = finalize_generation_records(
        fresh,
        dataset_rows,
        output_dir=tmp_path,
        config={},
    )
    by_id = {row["query_id"]: row for row in attached}
    assert by_id["A01"]["answer"] == "new-answer"
    assert by_id["A01"]["human_correctness"] == 2
    assert by_id["A01"]["human_rating_sources"]["human_correctness"] == "existing_per_query"
    assert by_id["A02"]["human_correctness"] is None
    assert by_id["A02"]["answer"] == "also-new"


def test_review_beats_dataset_and_never_overwrites_with_null():
    dataset = [_row("A01", FULL)]
    review = {"A01": _row("A01", {**EMPTY, "human_correctness": 0})}
    records = [seed_generation_record(dataset[0], dry_run=False)]
    attached = attach_human_ratings(
        records,
        dataset_by_id={"A01": dataset[0]},
        review_by_id=review,
    )[0]
    assert attached["human_correctness"] == 0
    assert attached["human_faithfulness"] == 2
    assert attached["human_rating_sources"]["human_correctness"] == "review"
    assert attached["human_rating_sources"]["human_faithfulness"] == "dataset"


def test_aggregate_and_markdown_output(tmp_path: Path):
    records = [
        _row("A01", FULL, token_f1_vs_reference=0.5, lexical_exact_match=0.0),
        _row("A02", {**FULL, "human_medical_safety": 1}, token_f1_vs_reference=0.25),
    ]
    agg = human_rating_aggregates(records)
    agg["mean_token_f1"] = 0.375
    agg["mean_lexical_exact_match"] = 0.0
    agg["mean_abstention_match"] = 1.0
    agg["mean_relevancy_lexical_diagnostic"] = 0.4
    markdown = generation_report_markdown(agg, execute=True)
    write_markdown(tmp_path / "report.md", markdown)
    write_json(tmp_path / "aggregate.json", {"metrics": agg})
    write_jsonl(tmp_path / "per_query.jsonl", records)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Human evaluation" in text
    assert "not clinical accuracy" in text
    assert "`human_correctness`" in text
    assert "Automated lexical metrics" in text
    assert "mean_token_f1: 0.375" in text
    assert "human_metrics_filled: True" in text
    payload = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["human_medical_safety_mean"] == 1.5
    assert payload["metrics"]["human_medical_safety_n"] == 2
    assert payload["metrics"]["n_fully_annotated"] == 2
    saved = [
        json.loads(line)
        for line in (tmp_path / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert saved[0]["human_correctness"] == 2


def test_actual_dataset_loader_returns_all_six_ratings():
    rows = load_jsonl(TEMPLATE)
    assert [row["query_id"] for row in rows] == list(EXPECTED_TEMPLATE_RATINGS)
    for row in rows:
        got = tuple(row[field] for field in HUMAN_RATING_FIELDS)
        assert got == EXPECTED_TEMPLATE_RATINGS[row["query_id"]]


def test_generation_aggregates_actual_dataset_ratings_by_query_id(tmp_path: Path):
    rows = load_jsonl(TEMPLATE)
    records = [seed_generation_record(row, dry_run=True) for row in reversed(rows)]
    write_jsonl(
        tmp_path / "per_query.jsonl",
        [{"query_id": row["query_id"], **EMPTY} for row in rows],
    )
    attached = finalize_generation_records(
        records, rows, output_dir=tmp_path, config={}
    )
    assert [row["query_id"] for row in attached] == list(
        reversed(EXPECTED_TEMPLATE_RATINGS)
    )
    for record in attached:
        got = tuple(record[field] for field in HUMAN_RATING_FIELDS)
        assert got == EXPECTED_TEMPLATE_RATINGS[record["query_id"]]
        assert all(
            record["human_rating_sources"][field] == "dataset"
            for field in HUMAN_RATING_FIELDS
        )
    agg = human_rating_aggregates(attached)
    assert agg["n_fully_annotated"] == 19
    assert agg["n_partial"] == 0
    assert agg["n_unannotated"] == 0
    assert agg["human_metrics_filled"] is True
    assert agg["human_correctness_mean"] == pytest.approx(28 / 19)
    assert agg["human_faithfulness_mean"] == pytest.approx(27 / 19)
    assert agg["human_relevance_mean"] == pytest.approx(35 / 19)
    assert agg["human_completeness_mean"] == pytest.approx(26 / 19)
    assert agg["human_abstention_mean"] == pytest.approx(34 / 19)
    assert agg["human_medical_safety_mean"] == pytest.approx(35 / 19)
    for field in HUMAN_RATING_FIELDS:
        assert agg[f"{field}_n"] == 19
        assert agg[f"{field}_n_missing"] == 0
        assert agg[f"{field}_normalized_pct"] == pytest.approx(
            agg[f"{field}_mean"] / 2 * 100
        )
    markdown = generation_report_markdown(agg, execute=False)
    assert "Human evaluation" in markdown
    assert "optional non-null override" in markdown
    assert "not required" in markdown


def test_invalid_review_file_fails_join(tmp_path: Path):
    dataset_rows = [_row("A01")]
    review = tmp_path / "per_query_human_review.jsonl"
    review.write_text(
        '{"query_id": "A01", "human_correctness": 9}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="human_correctness=9"):
        finalize_generation_records(
            [seed_generation_record(dataset_rows[0], dry_run=True)],
            dataset_rows,
            output_dir=tmp_path,
            config={},
        )
