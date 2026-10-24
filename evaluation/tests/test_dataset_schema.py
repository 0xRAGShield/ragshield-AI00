"""Schema tests for the placeholder JSONL. No pipeline execution."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_jsonl, gold_chunk_ids, validate_row


TEMPLATE = Path(__file__).resolve().parents[1] / "datasets" / "evaluation_template.jsonl"


def test_template_loads_and_has_no_fabricated_chunk_gold():
    rows = load_jsonl(TEMPLATE)
    assert len(rows) >= 1
    for row in rows:
        assert gold_chunk_ids(row) is None
        assert row.get("notes")


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
