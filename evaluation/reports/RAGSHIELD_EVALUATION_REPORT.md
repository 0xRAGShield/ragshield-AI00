Report — Step 1: Retrieval Evaluation
Evaluation metadata
Field	                 Value
Evaluation suite	Retrieval
Dataset size	        5 queries
Execution mode	Executed (execute=true)
Cutoffs	                \(k = 1, 3, 5\)
top_k	                 5
Errors                 	0
Queries with no results	0
Dataset SHA-256	874428a2fcd6c2dcd335987a2e81ba67bf4632a78b647f019d51a14653d1e6fe

Metrics used
Metric	Purpose
chunk_hit@k	Measures whether at least one relevant chunk appears within the top \(k\) results.
chunk_recall@k	Measures the proportion of known relevant chunks retrieved within the top \(k\).
chunk_precision@k	Measures the proportion of the top \(k\) results that are relevant.
chunk_mrr@k	Measures how highly the first relevant chunk is ranked.
chunk_ndcg@k	Measures ranking quality, potentially accounting for graded relevance.

Actual results
Metric	@1	@3	@5
chunk_hit@k	1.000	1.000	1.000
chunk_recall@k	1.000	1.000	1.000
chunk_precision@k	1.000	0.333	0.200
chunk_mrr@k	1.000	1.000	1.000
chunk_ndcg@k	1.000	1.000	1.000

Interpretation and limitations
The retrieval evaluation reported perfect scores for Hit, Recall, MRR, and nDCG at all three cutoffs. However, these results must be interpreted cautiously:

- Limited evaluation coverage: the aggregate report contains five queries, but most chunk-level metrics were calculated using only n = 2 eligible queries. The nDCG metrics used n = 4.
- Precision decreases as k increases: precision falls from 1.00 at @1 to 0.20 at @5. This means the proportion of relevant results decreases as more chunks are included in the evaluated ranking.
- No execution failures: the report recorded zero errors and zero queries with no results.
- Not yet a general performance guarantee: the perfect scores do not establish that retrieval is consistently accurate across a larger or more diverse medical dataset.
  The different denominators should be investigated to determine why some queries were excluded from individual metric calculation

# RAGShield Phase 2 evaluation report

**Date of this write-up:** 25 October 2026
**Repository context:** RAGShield Phase 2 tree (`ragshield-AI00-Phase2` / Colab path `/content/Phase2_OF_RAG` in run metadata). Git branch name `Phase2_OF_RAG` was stated by the operator; **git revision was not verified** in this reporting session.
**Evaluation approach:** custom Python metrics (`evaluation/metrics.py`). RAGAS, TruLens, and scikit-learn are not used.
**Scope:** one live execution family of the five evaluation suites on a five-query dataset. This document **does not rerun** models or evaluation.

---

## 1. Executive summary

A live run (`execute=true`) completed on **25 October 2026 (UTC)** against collection `ragshield_test` with dataset SHA-256 `874428a2fcd6c2dcd335987a2e81ba67bf4632a78b647f019d51a14653d1e6fe`. All five queries executed without pipeline errors or degraded flags.

**What the numbers support**

- Dense retrieval placed the **two binary-gold chunks** (D02, D03) at rank 1. Hit/Recall/MRR of 1.0 at every k describe **those two queries only**, not the full set of five.
- The **cross-encoder reranker moved those gold chunks from rank 1 to rank 4 (D02) and rank 5 (D03)**. Post-rerank Hit@1 and Hit@3 are 0.0 on the labeled pair; Hit@5 remains 1.0. This is ranking demotion of evidence that retrieval had already found, not a collection miss.
- End-to-end **control-plane success is 1.0** (schema-valid `RAGPipeline.run` outputs, routing CHAT/RAG accuracy 1.0). That is **not** medical correctness.
- Generation produced answers for all five queries (`n_generation_none=0`). Mean token-F1 versus reference is **0.25**. Mean abstention match is **0.40**. Human quality scores were **not filled**.
- Generation latency dominates wall time (~**78 s** mean vs ~91 ms retrieval and ~23 ms rerank).

**What the numbers do not support**

- They do **not** show that the system is hallucination-free. Hallucination aggregates (when taken from the operator’s latest known report) show **100% indeterminate claims** and a **null** answer-level hallucination rate. No support/contradiction decision was made.
- Perfect retrieval scores must not be generalised beyond n = 2 (Hit/Recall/MRR/Precision) or n = 4 (nDCG).
- Token-F1 is lexical overlap, not faithfulness or clinical accuracy.

---

## 2. Evaluation configuration and dataset metadata

### 2.1 Source of truth for the run

Checked-in `evaluation/config.json` currently has `"execute": false` (dry-run default). **The run artifacts override that:** each `aggregate.json` stores `"execute": true` and a copied config with `execute: true`.

| Item                  | Value                                                                            | Source                                                 |
| --------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Dataset path (config) | `evaluation/datasets/evaluation_template.jsonl`                                | run `meta.config`                                    |
| Dataset path (Colab)  | `/content/Phase2_OF_RAG/evaluation/datasets/evaluation_template.jsonl`         | retrieval `meta.dataset_path`                        |
| Dataset SHA-256       | `874428a2fcd6c2dcd335987a2e81ba67bf4632a78b647f019d51a14653d1e6fe`             | all suite `meta.dataset_sha256` (same hash)          |
| `n_rows`            | 5                                                                                | all aggregates                                         |
| Suites enabled        | retrieval, reranker, generation, hallucination, end_to_end                       | `meta.config.suites`                                 |
| Track                 | `full_pipeline`                                                                | `meta.config.evaluation_track`                       |
| `top_k`             | 5                                                                                | config / retrieval                                     |
| `top_n`             | 5                                                                                | config / reranker                                      |
| `k_values`          | [1, 3, 5]                                                                        | config                                                 |
| LLM judge             | `optional_llm_judge=false`; `optional_llm_judge_ran` is always false in code | config + `evaluate_generation.py`                    |
| Human scores          | `human_metrics_filled=false`                                                   | generation aggregate                                   |
| Collection            | `ragshield_test`                                                               | end-to-end per-query `settings_collection`           |
| Redaction             | `redact_evidence_text=true`                                                    | config; candidate `text` redacted in retrieval JSONL |

### 2.2 Run timestamps (UTC)

One sequential family; hashes match. No earlier run directories were found under `evaluation/reports/runs/`.

| Suite      | `generated_at_utc`             | Artifact path                                                             |
| ---------- | -------------------------------- | ------------------------------------------------------------------------- |
| Retrieval  | 2026-10-25T01:55:13.288303+00:00 | `evaluation/reports/runs/retrieval/`                                    |
| Reranker   | 2026-10-25T01:55:35.487707+00:00 | `evaluation/reports/runs/reranker/aggregate.json`                       |
| Generation | 2026-10-25T02:03:47.623143+00:00 | `evaluation/reports/runs/generation/aggregate.json`                     |
| End-to-end | 2026-10-25T02:20:24.981961+00:00 | `evaluation/reports/runs/end to end/` (directory name contains a space) |

**Hallucination:** no `aggregate.json`, `per_query.jsonl`, or `report.md` exists under `evaluation/reports/runs/` (or any `*hallucin*` path). Suite code exists at `evaluation/hallucination/evaluate_hallucination.py`. Hallucination numbers in §4.D are the **operator-supplied latest known values**, labelled **unverified in this tree**.

**Also missing from the tree:** `reranker/per_query.jsonl`, `generation/per_query.jsonl`, and most suite `report.md` files except retrieval.

### 2.3 Dataset composition (current JSONL on disk)

Five queries. Binary retrieval gold (`relevant_chunk_ids` non-empty) exists **only for D02 and D03**. A01/A02 are `topic_only` (grade 1, not in `relevant_chunk_ids`). D01 is `not_sufficient` (grade 0). `route_expected` is `RAG` for all five. `answerable` is false for A01, A02, D01 and true for D02, D03.

The on-disk JSONL currently includes `reference_answer` strings. The live generation aggregate has a non-null `mean_token_f1`, which requires those fields at run time. This reporting session **did not recompute** the SHA-256 of the current file; if the file was edited after the run, hashes could diverge.

### 2.4 Setup limitations

- n = 5 total; n = 2 for binary IR metrics.
- Custom metrics only; no RAGAS/TruLens/sklearn.
- Images were empty in end-to-end records (`images: []`).
- English MiniLM cross-encoder is used for rerank (production `CrossEncoderRerankerModel`).
- Current `config.json` on disk is dry-run; do not confuse it with the executed run.

---

## 3. How denominators work

`evaluation/metrics.py` returns `None` when gold ID lists are empty. Aggregates use `mean_ignore_none`. Therefore:

| Count                  | Meaning in this run                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| `n_rows` = 5         | Dataset size / successful loop iterations                                                                     |
| `n_errors` = 0       | Exceptions in the suite runner                                                                                |
| `chunk_hit@k_n` = 2  | Queries with non-empty `relevant_chunk_ids` (D02, D03)                                                      |
| `chunk_ndcg@k_n` = 4 | Queries with usable grades (A01, A02 grade 1; D02, D03 grade 3). D01 grade 0 yields `ndcg=null` (IDCG is 0) |

Reranker `aggregate.json` **does not store `_n`**. From the same gold rules, Hit/Recall/Precision/MRR means are over **n = 2**; nDCG over **n = 4**.

---

## 4. Stage A — Retrieval

**Implementation:** `Retriever.retrieve` → `ranking_bundle` in `evaluation/metrics.py`, aggregated in `evaluate_retrieval.py`.

### 4.1 Metric definitions (custom Python)

| Metric name in aggregates | Definition                                                 | Cutoff       | Higher/lower |
| ------------------------- | ---------------------------------------------------------- | ------------ | ------------ |
| `chunk_hit@k`           | 1 if any gold `chunk_id` is in the first k retrieved IDs | k ∈ {1,3,5} | higher       |
| `chunk_recall@k`        | \|top-k ∩ gold\| / \|gold\|                               | same         | higher       |
| `chunk_precision@k`     | \|top-k ∩ gold\| / **k** (cutoff, not list length)  | same         | higher       |
| `chunk_mrr@k`           | 1/rank of first gold, else 0                               | same         | higher       |
| `chunk_ndcg@k`          | DCG/IDCG using `relevance_judgments` grades when present | same         | higher       |

Source-level `source_hit@k` / `source_recall@k` are computed per query in `per_query.jsonl` but **not** copied into retrieval `aggregate.json`.

Latencies recorded per query: `embedding_latency_ms`, `search_latency_ms`, `total_latency_ms`. They are **not** averaged in retrieval `aggregate.json`.

### 4.2 Confirmed aggregate results

From `evaluation/reports/runs/retrieval/aggregate.json`.

Operational: `n_rows=5`, `n_errors=0`, `n_no_result=0`.

| k | `chunk_hit@k` | `chunk_recall@k` | `chunk_precision@k` | `chunk_mrr@k` | n (binary) | `chunk_ndcg@k` | n (nDCG) |
| - | --------------: | -----------------: | --------------------: | --------------: | ---------: | ---------------: | -------: |
| 1 |             1.0 |                1.0 |                   1.0 |             1.0 |          2 |              1.0 |        4 |
| 3 |             1.0 |                1.0 |    0.3333333333333333 |             1.0 |          2 |              1.0 |        4 |
| 5 |             1.0 |                1.0 |                   0.2 |             1.0 |          2 |              1.0 |        4 |

Precision falls as 1/k because each labeled query has **one** gold chunk at rank 1: Precision@3 = 1/3, Precision@5 = 1/5. That is arithmetic, not retrieval failure.

### 4.3 Per-query evidence (retrieval `per_query.jsonl`)

| query_id | Binary gold               | Rank of gold / topic chunk                        | Hit@k | nDCG |
| -------- | ------------------------- | ------------------------------------------------- | ----- | ---- |
| A01      | none (`topic_only`)     | judged chunk `…:chunk:000005` at rank 1        | null  | 1.0  |
| A02      | none (`topic_only`)     | judged chunk `…:chunk:000008` at rank 1        | null  | 1.0  |
| D01      | none (`not_sufficient`) | judged `…:chunk:000002` **not** in top 5 | null  | null |
| D02      | `…:chunk:000002`       | rank 1                                            | 1.0   | 1.0  |
| D03      | `…:chunk:000002`       | rank 1                                            | 1.0   | 1.0  |

**Interpretation:** On the two questions the dataset treats as answerable with direct evidence, dense cosine search returned the gold chunk first. Perfect 1.0 scores are **not** a five-query retrieval benchmark. A01/A02 nDCG = 1.0 only means the **topic-only** chunk (grade 1) was rank 1; those chunks are documented as **not** answering the question.

---

## 5. Stage B — Reranker

**Implementation:** same `ranking_bundle` on pre-rerank retrieval IDs and post-rerank IDs; delta = post − pre (`evaluate_reranker.py`). `n_fallback=0`. Frozen candidate pool was not used (`frozen_candidate_pool_path: null`), so the pool is live retrieval.

Reranker `per_query.jsonl` is **not in the repository**. Rank orders below are taken from **end-to-end** `reranked_chunk_ids` on the same dataset hash (later in the same UTC morning). That is a **same-run-family inference**, not a substitute for the missing rerank JSONL.

### 5.1 Confirmed aggregates

From `evaluation/reports/runs/reranker/aggregate.json`. Pre values match retrieval binary metrics for Hit/Recall/Precision/MRR.

| Metric    | k |                Pre |              Post |  Delta (post − pre) |
| --------- | -: | -----------------: | ----------------: | -------------------: |
| hit       | 1 |                1.0 |               0.0 |                −1.0 |
| recall    | 1 |                1.0 |               0.0 |                −1.0 |
| precision | 1 |                1.0 |               0.0 |                −1.0 |
| mrr       | 1 |                1.0 |               0.0 |                −1.0 |
| ndcg      | 1 |                1.0 |              0.25 |               −0.75 |
| hit       | 3 |                1.0 |               0.0 |                −1.0 |
| recall    | 3 |                1.0 |               0.0 |                −1.0 |
| precision | 3 | 0.3333333333333333 |               0.0 | −0.3333333333333333 |
| mrr       | 3 |                1.0 |               0.0 |                −1.0 |
| ndcg      | 3 |                1.0 |              0.25 |               −0.75 |
| hit       | 5 |                1.0 |               1.0 |                  0.0 |
| recall    | 5 |                1.0 |               1.0 |                  0.0 |
| precision | 5 |                0.2 |               0.2 |                  0.0 |
| mrr       | 5 |                1.0 |             0.225 |              −0.775 |
| ndcg      | 5 |                1.0 | 0.551095543135619 | −0.4489044568643809 |

Eligible queries: **n = 2** for Hit/Recall/Precision/MRR (D02, D03); **n = 4** for nDCG (A01, A02, D02, D03).

### 5.2 What the drop at k=1 and k=3 means

End-to-end lists (same gold chunk `e1c59e60…:chunk:000002`):

| query_id | Retrieval order (gold position) | Reranked order (gold position) |
| -------- | ------------------------------- | ------------------------------ |
| D02      | gold **1st** of 5         | gold **4th** of 5        |
| D03      | gold **1st** of 5         | gold **5th** of 5        |

That explains the table without assuming a broken `Reranker.rerank` implementation:

- Hit@1/@3 go to 0 because the gold left the top 3.
- Hit@5 stays 1.0 because the gold remained in the top-5 pool (`top_n=5`).
- MRR@5 = mean(1/4, 1/5) = **0.225**, matching the aggregate exactly.
- Post nDCG@1 = **0.25** matches mean over four graded queries if only A01 still has its grade-1 chunk at rank 1 after rerank (A02’s topic chunk moved to last place in the e2e list).

**Confirmed:** reranking **reordered** a pool that already contained the gold; it did not drop the gold from top-5. `n_fallback=0` and `n_errors=0`.

**Hypotheses (not proven from missing rerank per-query scores):**

1. `cross-encoder/ms-marco-MiniLM-L-6-v2` is English passage-ranking; D03 is Arabic and D02 is short definitional medical text.
2. The scorer uses `expand_query_with_document_aliases` while the original query is stored on the result (same as retrieval embedding).
3. Label/ID mismatch is **unlikely** for D02/D03: the same `chunk_id` string appears in retrieval rank 1 and rerank rank 4–5.

Do **not** treat this as a crash or empty-output defect. Treat it as **cross-encoder demotion of the dense-retrieval #1 hit**, to be tested with more labeled queries and, if desired, an Arabic-capable reranker experiment.

Context construction uses **reranked** order (`ContextBuilder.build(reranking)`). For D02/D03 the gold still entered the five evidence blocks (`gold_in_context=1.0` in e2e), but not at the top of the prompt.

---

## 6. Stage C — Generation

**Implementation:** `RAGPipeline.run` then `_token_f1` and `looks_like_abstention`. No LLM-as-judge module. Generation `per_query.jsonl` is **absent**; only `aggregate.json` is confirmed.

### 6.1 Automated vs human fields

| Field                                                                                                                                      | Type                                                             | Computed in this run?                                 | Scale in code                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| `token_f1_vs_reference` / `mean_token_f1`                                                                                              | Automated set-overlap F1 on whitespace tokens after `casefold` | Yes (mean 0.2473843303911954)                         | 0–1                                                  |
| `abstention_match` / `mean_abstention_match`                                                                                           | Automated: heuristic abstention vs `answerable is False`       | Yes (mean 0.4)                                        | 0/1 then mean                                         |
| `n_errors`, `n_degraded`, `n_generation_none`                                                                                        | Counts                                                           | 0, 0, 0                                               | —                                                    |
| `human_correctness`, `human_faithfulness`, `human_relevance`, `human_completeness`, `human_abstention`, `human_medical_safety` | Placeholders initialised to `None`                             | **Not filled** (`human_metrics_filled=false`) | **No 1–5 (or other) scale is defined in code** |

Token F1 (`evaluate_generation.py`, `_token_f1`): unique-token precision/recall harmonic mean. It does **not** measure medical correctness, evidence faithfulness, or Arabic morphology well. Paraphrase and extra citations lower F1 even when the answer is right.

Abstention (`common.looks_like_abstention`): substring search for phrases such as `insufficient`, `not enough evidence`, `cannot determine`, `لا يكفي`, `غير كاف`, `لا يمكن`. For `answerable=false`, a match is expected; for `answerable=true`, a match is a mismatch. Mean 0.4 on five labeled `answerable` flags is consistent with **2/5 matches**. End-to-end `answerability_ok` is `false` for A01, A02, D01 and `true` for D02, D03, i.e. the model **answered the three items marked unanswerable** instead of abstaining. That is a **quality/policy** issue, not a crash.

### 6.2 Confirmed generation aggregate

| Metric                    | Value              |
| ------------------------- | ------------------ |
| `n_rows`                | 5                  |
| `n_errors`              | 0                  |
| `n_degraded`            | 0                  |
| `n_generation_none`     | 0                  |
| `mean_token_f1`         | 0.2473843303911954 |
| `mean_abstention_match` | 0.4                |
| `human_metrics_filled`  | false              |

Without generation per-query rows, the token-F1 mean cannot be split by language or query. Reference answers exist on the current JSONL for all five items; A01–D01 references describe facts the mapped chunks were labelled **not sufficient** to support, so high F1 on those items would be **lexically matching an ungrounded reference**, not proof of grounded RAG.

---

## 7. Stage D — Hallucination

**Implementation:** labels `supported`, `unsupported`, `contradicted`, `indeterminate`. Rates use **all claims including indeterminate** as denominator. Answer-level hallucination is 1 if any **determined** claim is unsupported or contradicted; if every claim is indeterminate, answer-level rate is `None`.

Empty `claim_annotations` in the dataset causes the runner to sentence-split the answer and label every span `indeterminate` (`evaluate_hallucination.py`).

### 7.1 Results — not found on disk

No hallucination artifacts exist under `evaluation/reports/runs/`. The following are **the operator’s latest known report**, **not verified as files in this tree**. They are **consistent** with the unlabeled path (all indeterminate → support/unsupported/contradiction 0, indeterminate 1, answer-level null):

| Metric                                   | Reported value | Meaning if that run followed the code |
| ---------------------------------------- | -------------- | ------------------------------------- |
| `n_rows`                               | 5              | five queries                          |
| `n_errors`                             | 0              | no exceptions                         |
| `mean_claim_support_rate`              | 0.0            | no claims labelled supported          |
| `mean_unsupported_claim_rate`          | 0.0            | no claims labelled unsupported        |
| `mean_contradiction_rate`              | 0.0            | no claims labelled contradicted       |
| `mean_indeterminate_claim_rate`        | 1.0            | all claims indeterminate              |
| `mean_answer_level_hallucination_rate` | null           | no determined claims                  |

**These results do not show that answers are hallucination-free.** They show that **no claim was judged supported or unsupported**. Filling `claim_annotations` (or implementing a real judge; the flag currently no-ops) is required before hallucination rates are interpretable.

A weak lexical-overlap heuristic exists (`_lexical_overlap_supported`) but **never assigns unsupported**; it only records `True` or `None` and does not drive the aggregate rates.

---

## 8. Stage E — End-to-end

**Implementation:** in-process `RAGPipeline.run` plus `classify_query`. `e2e_success` is schema/control success (`QueryResponse`-shaped fields, no exception), **not** answer quality (`evaluate_end_to_end.py`).

### 8.1 Confirmed aggregate

From `evaluation/reports/runs/end to end/aggregate.json`.

| Metric                         | Value              | Unit / notes                                      |
| ------------------------------ | ------------------ | ------------------------------------------------- |
| `n_rows`                     | 5                  |                                                   |
| `n_errors`                   | 0                  |                                                   |
| `e2e_success_rate`           | 1.0                | execution/schema success                          |
| `n_degraded`                 | 0                  |                                                   |
| `routing_accuracy`           | 1.0                | all five `route_expected=RAG` and predicted RAG |
| `mean_retrieval_latency_ms`  | 91.17339599997649  | ms                                                |
| `mean_rerank_latency_ms`     | 23.058541800310195 | ms                                                |
| `mean_generation_latency_ms` | 77778.49430360002  | ms (~77.8 s)                                      |

Per-query generation latency (ms): A01 64975; A02 149967; D01 36147; D02 35638; D03 102166. **Generation is the bottleneck** in these traces. Hardware cause is **not** recorded. Context build is ~0.2–0.3 ms.

Additional per-query fields (not in the aggregate): `gold_hit_retrieval` and `gold_in_context` are 1.0 for D02/D03 and null otherwise. `answerability_ok` is false for the three `answerable=false` queries. `prompt_safety_mode` is `strict`. `schema_valid` is true for all five.

---

## 9. Cross-stage interpretation

1. **Retrieval found the right chunk; rerank buried it; context still included it at a worse position.** Generation therefore saw the gold among five blocks, but not as the first evidence item.
2. **The pipeline is operationally healthy** on this slice (no errors, no degraded, e2e 1.0) while **quality instrumentation is incomplete** (human scores empty, hallucination unlabelled, IR gold n = 2).
3. **Unanswerable items did not trigger abstention**, despite `safety_mode=strict` and `answerable=false`. Combined with topic-only retrieval, the model likely answered from parametric knowledge or weak context. Token-F1 cannot confirm that.
4. **nDCG looks perfect at retrieval because grade-1 topic chunks sat at rank 1**, which is a different story from “the answer is in the evidence.”

---

## 9b. Framework upgrade (code on disk; live numbers above were not re-executed)

The evaluation package now:

- Treats empty `relevant_chunk_ids` as **unlabelled** (`None`), not zero, unless `no_relevant_results=true`.
- Reports **binary** `ndcg@k` on the same eligible set as Hit/Recall/Precision/MRR, and **graded** `ndcg_graded@k` separately. The live retrieval table in §4 still shows the **old** mixed nDCG (n=4) because those artifacts were not regenerated.
- Labels token F1 / exact match as **lexical overlap**, not clinical correctness.
- Scores answer-level hallucination only when at least one claim is determined; indeterminate-only queries stay excluded.
- Writes per-metric `n_eligible` / `n_excluded` and exclusion reasons.
- Validates the JSONL via `evaluation/validate_dataset.py` without contacting Qdrant.
- Documents human ratings on a 0/1/2 scale in `evaluation/ANNOTATION.md`. Completeness requires `key_facts` or `human_completeness`; it is not inferred from answer length.
- Keeps end-to-end success as **in-process execution**, distinct from HTTP `/query`. An optional API test runs only if `RAGSHIELD_EVAL_API_URL` is set.

Do not treat this section as a new live scorecard. Re-run with `--execute` to refresh aggregates.

---

## 10. Known issues and threats to validity

- Sample size: five queries; two with binary chunk gold; both golds share one `chunk_id`.
- Perfect Hit@1 is not a general retrieval result.
- Rerank per-query file missing; rank analysis used e2e lists.
- Hallucination outputs missing from disk.
- Generation per-query missing; answers redacted in e2e (`redact_evidence_text`).
- Human 0/1/2 rubric exists in code; live rows remain unannotated.
- Chunk IDs were not re-validated against the ingest manifest in this reporting pass (they did appear in live Qdrant results).
- English MS MARCO MiniLM on mixed Arabic/English medical text.
- Dataset SHA-256 was not recomputed against the current JSONL.
- `config.json` on disk still says `execute: false`.
- No security-layer evaluation suite.

---

## 11. Prioritized next steps

1. **Keep the two-query IR result as a smoke test**, not a published retrieval score. Add more `direct` golds, including items where gold is not already rank 1.
2. **Inspect rerank score lists** (restore `reranker/per_query.jsonl` on the next run). Compare MiniLM vs a multilingual reranker on D02/D03 as an experiment—not a production change in this report.
3. **Fill human faithfulness/correctness/abstention** on the five answers using the 0/1/2 scale in `evaluation/ANNOTATION.md`.
4. **Add `claim_annotations`** so hallucination rates are determined; do not cite 0.0 unsupported as safety evidence.
5. **Tighten abstention evaluation** on A01/A02/D01: the e2e `answerability_ok=false` pattern is the main generation-policy finding.
6. **Archive hallucination and generation JSONL** with the other suites so a report can be fully file-backed.
7. Optionally add RAGAS later as a **separate** optional runner; do not replace Hit/MRR.

---

## 12. Supervisor / reviewer summary

RAGShield Phase 2 was evaluated with a **custom Python harness** (not RAGAS/TruLens/sklearn) on **five** bilingual queries. Live execution succeeded: no crashes, routing accuracy 1.0, end-to-end schema success 1.0. Dense retrieval ranked the only two fully labelled evidence chunks first. The **cross-encoder then moved those chunks to ranks 4 and 5**, so post-rerank Hit@1/@3 is 0 while Hit@5 remains 1. Generation always returned text; lexical F1 versus references is about 0.25; abstention matching is 0.40 and the three unanswerable items were not handled as abstentions in the e2e records. **Hallucination metrics, if taken from the operator report, are entirely indeterminate and must not be read as “no hallucination.”** Human quality scores were not filled. Latency is dominated by generation (~78 s mean). The study is a useful **pipeline health and rerank-behaviour diagnostic** on a tiny labelled slice, not a clinical validation of answer quality.

---

## Appendix — files inspected

**Code:** `evaluation/config.json`, `evaluation/metrics.py`, `evaluation/common.py`, `evaluation/run_all.py`, `evaluation/retrieval/evaluate_retrieval.py`, `evaluation/reranker/evaluate_reranker.py`, `evaluation/generation/evaluate_generation.py`, `evaluation/hallucination/evaluate_hallucination.py`, `evaluation/end_to_end/evaluate_end_to_end.py`, `evaluation/datasets/evaluation_template.jsonl`.

**Artifacts (verified at write-up time; some later overwritten):**
Live end-to-end remains at `evaluation/reports/runs/end to end/`.
A later dry-run wrote under `evaluation/reports/runs/dry_run/` so unexecuted suites no longer clobber live JSON.
Live retrieval/reranker/generation JSONL from 2026-10-25 01:55–02:03 UTC were **not in git** and were overwritten once by a dry-run; numeric values in this report still come from that live family (and from the surviving `end to end` files). Re-run `--execute` to regenerate retrieval/reranker/generation artifacts.

**Also present now:** `evaluation/reports/runs/validation/`; dry-run hallucination outputs.

**This report path:** `evaluation/reports/RAGSHIELD_EVALUATION_REPORT.md`

**Unresolved:** on-disk SHA-256 vs recorded hash; location of hallucination artifacts; per-query token-F1 and rerank scores; git commit of the Colab tree.
