"""HTTP QueryResponse contract. Live POST is optional and skipped by default."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "RAG-GEN" / "app" / "api" / "main.py"


def test_http_query_response_omits_chunk_ids():
    tree = ast.parse(API.read_text(encoding="utf-8"))
    fields: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "QueryResponse":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.append(item.target.id)
    assert "answer" in fields
    assert "retrieved_candidates" in fields
    assert "chunk_ids" not in fields
    assert "retrieved_chunk_ids" not in fields


@pytest.mark.skipif(
    not os.environ.get("RAGSHIELD_EVAL_API_URL"),
    reason="optional HTTP API test; set RAGSHIELD_EVAL_API_URL to enable",
)
def test_optional_live_query_endpoint():
    import json
    import urllib.request

    url = os.environ["RAGSHIELD_EVAL_API_URL"].rstrip("/") + "/query"
    payload = json.dumps({"query": "ping"}).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    for key in (
        "answer",
        "trace_id",
        "degraded",
        "retrieved_candidates",
        "reranked_candidates",
        "evidence_items",
        "images",
    ):
        assert key in body
