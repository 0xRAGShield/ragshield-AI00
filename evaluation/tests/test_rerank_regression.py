"""Document the D02/D03 rerank demotion as ranking quality, not an eval/ID bug.

Uses the recorded candidate orders from evaluation/reports/runs/reranker/per_query.jsonl
(2026-10-27 execute=true). Does not load CrossEncoder or Qdrant.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import ranking_bundle

D02_GOLD = "e1c59e60738bde64aa366585663c29295c412bcd212905c235a1260d29f75e1e:chunk:000002"
D03_GOLD = D02_GOLD

# Live run orders (identical ID strings pre/post; only permutation).
D02_PRE = [
    "e1c59e60738bde64aa366585663c29295c412bcd212905c235a1260d29f75e1e:chunk:000002",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000022",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000034",
    "e1c59e60738bde64aa366585663c29295c412bcd212905c235a1260d29f75e1e:chunk:000000",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000030",
]
D02_POST = [
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000030",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000034",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000022",
    "e1c59e60738bde64aa366585663c29295c412bcd212905c235a1260d29f75e1e:chunk:000002",
    "e1c59e60738bde64aa366585663c29295c412bcd212905c235a1260d29f75e1e:chunk:000000",
]
D03_PRE = [
    "e1c59e60738bde64aa366585663c29295c412bcd212905c235a1260d29f75e1e:chunk:000002",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000022",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000034",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000006",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000015",
]
D03_POST = [
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000034",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000006",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000022",
    "f2233613a261ebc2acd86374693dfc732d418a6eb65ff57bc30c4c9bd4e4100a:chunk:000015",
    "e1c59e60738bde64aa366585663c29295c412bcd212905c235a1260d29f75e1e:chunk:000002",
]


def test_d02_d03_regression_is_gold_demotion_not_id_mismatch():
    assert set(D02_PRE) == set(D02_POST)
    assert set(D03_PRE) == set(D03_POST)
    assert D02_GOLD in D02_PRE and D02_GOLD in D02_POST
    pre = ranking_bundle(D02_PRE, [D02_GOLD], 3)
    post = ranking_bundle(D02_POST, [D02_GOLD], 3)
    assert pre["hit@3"] == 1.0
    assert post["hit@3"] == 0.0
    d03_pre = ranking_bundle(D03_PRE, [D03_GOLD], 1)
    d03_post = ranking_bundle(D03_POST, [D03_GOLD], 1)
    assert d03_pre["hit@1"] == 1.0
    assert d03_post["hit@1"] == 0.0
    assert ranking_bundle(D03_POST, [D03_GOLD], 5)["hit@5"] == 1.0


def test_ms_marco_negative_scores_still_sort_higher_first():
    """MiniLM logits may be negative; higher_is_better=True must not invert them."""
    scored = [
        ("gold_en_definition", -2.170001983642578),
        ("ar_case_030", -0.5070029497146606),
        ("ar_case_034", -0.9004439115524292),
        ("ar_case_022", -0.9654197692871094),
    ]
    ordered = sorted(scored, key=lambda item: item[1], reverse=True)
    assert [item[0] for item in ordered] == [
        "ar_case_030",
        "ar_case_034",
        "ar_case_022",
        "gold_en_definition",
    ]
