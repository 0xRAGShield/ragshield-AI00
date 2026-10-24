# RAGShield Phase 2 evaluation report

Fill this after **you** execute the runners. Scaffolding produced no scores.

## Environment and configuration

- Date (UTC):
- Git revision: UNKNOWN until recorded
- Collection: `ragshield_test` (confirm)
- Snapshot checksum:
- Dataset path / sha256:
- n queries / split:
- `top_k` / `top_n` / `k_values`:
- Prompt `safety_mode`:
- Embedding: `BAAI/bge-m3`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM id / quantization:
- `execute` flag:
- QdrantStore `vector_size` argument: must be `None` (auto-create disabled)

## Dataset coverage

| Label type | n queries with non-empty labels |
| --- | --- |
| chunk gold | |
| source gold | |
| graded judgments | |
| image gold | |
| reference answers | |
| claim annotations | |
| route_expected | |

## Retrieval

| metric | k | mean | n_scored | notes |
| --- | --- | --- | --- | --- |
| Hit@k |  |  |  |  |
| Recall@k |  |  |  |  |
| Precision@k |  |  |  | denom = k |
| MRR@k |  |  |  |  |
| nDCG@k |  |  |  | skip if no grades/gold |

No-result count / errors:

## Reranking (same candidate pool)

| metric | pre | post | delta | n |
| --- | --- | --- | --- | --- |
| MRR@k |  |  |  |  |
| Hit@k |  |  |  |  |

Golds missing from the input pool (unrecoverable):

## Generation (human primary)

| metric | mean | n | method |
| --- | --- | --- | --- |
| correctness |  |  | human |
| faithfulness |  |  | human vs evidence text |
| relevance |  |  | human |
| completeness |  |  | human |
| abstention |  |  | human + heuristic |
| medical safety |  |  | human |
| token-F1 |  |  | automated, secondary |

## Hallucination

Denominator = all claims including indeterminate, except answer-level hallucination (determined claims only).

| metric | mean | n |
| --- | --- | --- |
| Claim Support Rate |  |  |
| Unsupported Claim Rate |  |  |
| Contradiction Rate |  |  |
| Indeterminate Claim Rate |  |  |
| Answer-level Hallucination Rate |  |  |

## End-to-end

- e2e_success_rate (schema/control, **not** quality):
- degraded rate:
- routing accuracy:
- stage latencies (ms): retrieval / rerank / context / generation

## Limitations

- No PII detector or prompt-injection classifier on the request path.
- Images collected from **reranked** candidates only.
- English MiniLM reranker on Arabic queries.
- Template dataset is unlabeled.
- Not evaluated:
