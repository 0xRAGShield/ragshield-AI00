
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import HUMAN_RATING_FIELDS, gold_chunk_ids, load_jsonl, validate_row
from validate_dataset import build_report

TEMPLATE = Path(__file__).resolve().parents[1] / "datasets" / "evaluation_template.jsonl"
SYNTHETIC = Path(__file__).resolve().parents[1] / "datasets" / "synthetic_eval.jsonl"


def test_template_loads_without_inventing_unlabelled_as_empty_gold():
    rows = load_jsonl(TEMPLATE)
    assert [row["query_id"] for row in rows] == [
        "A01",
        "A02",
        "D01",
        "D02",
        "D03",
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "P01",
        "P02",
        "P03",
        "H01",
        "H02",
        "H03",
        "ALL01",
        "ALL02",
        "ALL03",
    ]
    gold_map = {row["query_id"]: gold_chunk_ids(row) for row in rows}
    for qid in ("A01", "A02", "D01", "D02", "D03"):
        assert gold_map[qid], qid
    by_id = {row["query_id"]: row for row in rows}
    d02_claims = " ".join(c["claim"] for c in by_id["D02"]["claim_annotations"])
    assert "High blood pressure" in d02_claims
    assert "Diabetes mellitus" not in d02_claims
    d03_labels = {c["label"] for c in by_id["D03"]["claim_annotations"]}
    assert "unsupported" in d03_labels
    for row in rows:
        assert row.get("notes")
        assert row.get("reference_answer")
        assert row.get("key_facts")
        for field in HUMAN_RATING_FIELDS:
            assert row.get(field) in (0, 1, 2), (row["query_id"], field, row.get(field))
    for qid in ("A01", "A02", "D01", "D02", "D03"):
        for claim in by_id[qid].get("claim_annotations") or []:
            assert isinstance(claim, dict), qid
            assert claim.get("label") in {
                "supported",
                "unsupported",
                "contradicted",
                "indeterminate",
            }


def test_pretty_printed_json_with_hash_comments(tmp_path: Path):
    path = tmp_path / "pretty.jsonl"
    chunk_id = ("ab" * 32) + ":chunk:000001"
    path.write_text(
        "# comment\n"
        "{\n"
        '  "query_id": "P01",\n'
        '  "query": "pretty?",\n'
        f'  "relevant_chunk_ids": ["{chunk_id}"]\n'
        "}\n\n"
        '{"query_id": "P02", "query": "compact"}\n',
        encoding="utf-8",
    )
    rows = load_jsonl(path)
    assert [row["query_id"] for row in rows] == ["P01", "P02"]
    assert gold_chunk_ids(rows[0]) == [chunk_id]
    assert gold_chunk_ids(rows[1]) is None


def test_validate_row_rejects_unknown_fields(tmp_path: Path):
    raised = False
    try:
        validate_row(
            {"query_id": "x", "query": "y", "not_a_field": 1},
            line_no=1,
            path=tmp_path / "x.jsonl",
        )
    except ValueError:
        raised = True
    assert raised


def test_synthetic_validation_report_counts():
    rows = load_jsonl(SYNTHETIC)
    report = build_report(rows, SYNTHETIC)
    assert report["n_rows"] == 5
    assert report["duplicate_query_ids"] == []
    assert "SYN-UNLABELED" in report["unlabelled_chunk_gold_query_ids"]
    assert "SYN-EMPTY" in report["explicit_no_relevant_results_query_ids"]
    assert report["n_with_binary_chunk_gold"] == 3
    assert report["human_annotation_status"]["n_fully_annotated"] == 5
    assert report["missing_claim_annotations_query_ids"] == []
