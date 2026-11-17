# CodeAnswer Experiments

## D1 citation and fallback audit

The [D1 audit](FALLBACK_CITATION_AUDIT.md) exercises the current citation
gate with 11 synthetic fixtures and separates raw generated text from the
final output. Four of ten returning fixtures used fallback, one raised a
synthetic generator error, and two final outputs still had 0/0 citation
metrics due to empty evidence. These fixture counts are **not** production
rates. The historical 0.619048 fallback aggregate cannot be attributed to
specific triggers because raw examples and logs are absent. The gate checks
numeric citation syntax and coverage of long lines, not claim support.

## C6 paired error analysis

The [C6 analysis](ERROR_ANALYSIS.md) reused A3's 1,200 frozen StackSample
test queries. Hybrid minus dense Recall@10 was +0.029167, with a 95% paired
bootstrap interval of [+0.010833, +0.046667] over 10,000 query-level
resamples. The hybrid gained 78 and lost 43 top-10 positives; every gain had
the gold document in BM25's top 40, while every loss lacked it there.
Identifier, version, error-word, and query-length slices are descriptive,
with no subgroup significance claim. This remains an in-corpus known-item
evaluation, not a test of generated answer quality.

## B4 resource profile

The [B4 resource profile](RESOURCE_COSTS.md) measured full and 20,000-vector
subset PCA/build work, then profiled full-index loading and two warmed
100-query retrieval rounds. On the full artifacts, Flat-384 / Flat-128 /
HNSW-128 add times were 0.827 / 0.242 / 64.437 seconds, with process peak
RSS 1,740.87 / 724.99 / 819.80 MiB. In the first warmed runtime round,
HNSW-128 search p95 was 0.933 ms, but fetching and cleaning its 40 Parquet
rows took 574.776 ms p95; process RSS peaked at 3,926.91 MiB. This excludes
reranking and generation. First index load was process-first, with OS cache
uncontrolled; GPU driver access was unavailable, so VRAM is unmeasured.

These are recorded historical results. The [C1 evaluation audit](EVALUATION_AUDIT.md)
explains which aggregates can be checked from committed files and which still
need raw labels, per-query outputs, or a rerun. Citation validity here concerns
number formatting, not whether a source supports an answer.

The [C2 frozen evaluation protocol](EVALUATION_PROTOCOL.md) was run on the full
corpus. Its scores are reported separately below because its query selection
differs from the historical experiments.

## C2 frozen Flat-384 baseline

The deterministic split has 400 development, 400 calibration, and 1,200 test
queries. Two split builds produced byte-identical files. The test run used the
520,000-row corpus, normalized MiniLM embeddings, and an exact Flat-384 index.

| Test metric | Result |
|---|---:|
| Recall@1 | 0.515000 |
| Recall@5 | 0.685833 |
| Recall@10 | 0.743333 |
| MRR@10 | 0.587885 |
| nDCG@10 | 0.625245 |
| Search p95 | 88.379 ms |

The [result summary](../results/evaluation/c2_frozen_baseline_summary.json),
[split IDs and labels](../results/evaluation/c2_query_ids.csv), and
[per-query test metrics](../results/evaluation/c2_flat_384_test_per_query.csv)
record configuration, hashes, and the query-level evidence. Search latency
excludes encoding and generation. The test is in-corpus known-item retrieval.

## A1 BM25 on the same frozen test

The [A1 BM25 baseline](BM25_BASELINE.md) indexes question bodies in SQLite
FTS5 with the same 520,000 `doc_id`s and evaluates the same 1,200 C2 test
queries. It recorded Recall@10 0.627500, Recall@40 0.726667, MRR@10
0.498723, and nDCG@10 0.529569. Search p95 was 614.834 ms at depth 10
and 626.449 ms at depth 40; the index was 236.73 MiB and took 30.089 s
to build. See the A1 page for the paired comparison, tokenization, exact-term
inspection, and per-query rankings.

## A2 hybrid RRF on the same frozen test

The [A2 fusion experiment](HYBRID_RRF.md) used fixed `k=60` and compared
source depths 20 and 40 on development before evaluating depth 40 on test.
On 1,200 test queries, hybrid Recall@10 was 0.772500 and Recall@40 was
0.849167, versus dense 0.743333 and 0.828333. MRR@10 was 0.609828 and
nDCG@10 was 0.648677. BM25 uniquely supplied 56 relevant documents to
the top-40 source union. Hybrid search p95 was 648.974 ms on 100 queries,
with query vectors precomputed. The A2 page includes source contributions,
latency scope, full results, and paired per-query rankings.

## A3 paired retrieval comparison

The [A3 comparison](FAIR_RETRIEVAL_COMPARISON.md) reran all 1,200 frozen test
rankings for BM25, dense, and hybrid, then measured raw-title single-query
latency on the same 100 queries. Hybrid Recall@10 was 0.772500 versus dense
0.743333, with 78 paired gains and 43 losses. Hybrid Recall@40 was 0.849167
versus dense 0.828333, with 48 gains and 23 losses. Raw-title p95 was
596.505 ms for BM25, 56.232 ms for dense, and 653.776 ms for hybrid.
The fixed 29-query exact-term slice and paired CSV are linked from the A3
page. These timings include encoding for dense and hybrid and use a different
boundary from the prior search-only latency figures.

## A4 cross-encoder reranking on the frozen test

The [A4 ablation](RERANKING_ABLATION.md) reranked each strategy's saved top-40
set using the configured MS MARCO cross-encoder. On the 1,200 paired known-item
queries, vote-off title-plus-body Recall@10 was 0.726667 for BM25, 0.828333
for dense, and 0.848333 for hybrid. Body-only Recall@10 was 0.677500,
0.757500, and 0.751667 respectively, versus 0.627500, 0.743333, and
0.772500 before reranking. The title-plus-body mode includes the query title
in its own judged document and should not be read as new-user-question
performance. The A4 page gives the vote-bonus ablation, paired gains and
losses, resource costs, and per-query evidence.

## B1 exact PCA dimension sweep

The [B1 sweep](PCA_DIMENSION_SWEEP.md) fitted one 256-component PCA basis on
100,000 recorded document IDs, then compared its 256/128/64 prefixes with
the exact 384-dimensional baseline on all 1,200 frozen test queries.
Recall@10 was 0.743333 at 384, 0.737500 at 256, 0.710000 at 128, and
0.602500 at 64. Search p95 fell from 80.618 ms at 384 to 47.200, 17.834,
and 8.859 ms respectively. Every index was exact, so these quality losses
come from projection and normalization, not HNSW approximation. The B1 page
links the fit IDs, per-query rankings, dimension manifests, and trade-off plot.

## B2 HNSW versus exact search

The [B2 comparison](HNSW_EXACT_COMPARISON.md) searched the identical frozen
queries and normalized corpus vectors with exact FlatIP and HNSW in both
128 and 384 dimensions. At 128, HNSW Recall@10 was 0.700000 versus exact
0.710000; at 384 it was 0.732500 versus exact 0.743333. Mean ANN top-10
overlap with the exact rankings was 0.989750 and 0.986083 respectively.
HNSW search p95 was 0.715 ms at 128 and 1.283 ms at 384, while its indexes
were larger than the exact indexes. The B2 page separates PCA loss from
same-space ANN loss and links all 2,400 paired per-query rankings.

## B3 frozen efSearch sweep

The [B3 sweep](EFSEARCH_SWEEP.md) reused the B2 HNSW-128 index and exact
same-space reference on all 1,200 frozen test queries. Recall@10 rose from
0.630833 at `efSearch=16` to 0.700000 at the configured 128 and 0.705000
at 256. Search p95 rose from 0.178 to 0.872 and 1.303 ms respectively.
Top-40 ANN overlap at 128 was 0.975417, so the curve had not saturated.
The configured 128 remains the stated choice pending a development-set
selection. Query-level rankings, overlap, QPS, and the curve are linked from
the B3 page. Earlier sweep figures below remain historical evidence.

## Historical experiments

## Evaluation protocol

The experiments use 520,000 answered Stack Overflow questions.

- Query: question title
- Indexed document: question body
- Returned content: highest-scored answer
- Relevant result: document belonging to the same question
- Retrieval metrics: Recall@k, MRR@10, and nDCG@10
- Performance metrics: QPS, latency, memory, and index size

This known-item task provides a reproducible offline evaluation, although
it does not fully represent real user queries.

## Iteration 1: exact dense retrieval

The baseline uses `all-MiniLM-L6-v2` as a bi-encoder and stores normalized
384-dimensional vectors in `FAISS IndexFlatIP`.

| Metric | Result |
|---|---:|
| Objects | 520,000 |
| Recall@1 | 0.4850 |
| Recall@5 | 0.6465 |
| Recall@10 | 0.7070 |
| MRR@10 | 0.5547 |
| nDCG@10 | 0.5912 |
| Search p95 | 71.059 ms |
| Index size | 761.72 MiB |

The exact index is used as the retrieval-quality reference.

## Iteration 2: PCA and HNSW

PCA reduces the vectors from 384 to 128 dimensions and preserves 79.53%
of the measured variance. HNSW then replaces exhaustive vector scanning.

| System | Recall@10 | QPS | p95 | Size |
|---|---:|---:|---:|---:|
| Flat-384 | 0.7070 | 193.13 | 71.059 ms | 761.72 MiB |
| Flat-PCA128 | 0.6655 | 590.26 | 11.170 ms | 253.91 MiB |
| HNSW-PCA128, ef=64 | 0.6430 | 14,345.05 | 0.310 ms | 325.45 MiB |
| HNSW-PCA128, ef=128 | 0.6555 | 7,672.06 | 0.514 ms | 325.45 MiB |

The final retriever uses `efSearch=128`. Compared with Flat-384, its
single-query search p95 is approximately 138 times lower and its index is
approximately 57% smaller. Recall@10 decreases by 0.0515.

## Iteration 3: reranked cited-answer RAG

The final pipeline retrieves 40 candidates, reranks them using
`cross-encoder/ms-marco-MiniLM-L-6-v2`, and gives the best three sources
to local `qwen2.5:3b`.

A threshold calibrated on separate in-domain and out-of-domain queries
decides whether the retrieved evidence is answerable.

| Metric | Result |
|---|---:|
| Calibration F1 | 0.9877 |
| Behavior accuracy | 0.9643 |
| In-domain answer rate | 1.0000 |
| Out-of-domain abstention | 0.8750 |
| Citation coverage | 1.0000 |
| Citation validity | 0.9728 |
| Faithfulness proxy | 0.6667 |
| Extractive fallback rate | 0.6190 |
| Retrieval p50 | 7.619 ms |
| Reranking p50 | 167.517 ms |
| Generation p50 | 2,493.236 ms |
| Total p50 | 2,941.517 ms |
| Total p95 | 5,476.624 ms |

When generated citation-number validity is below 1.0 or long-line citation
coverage is below 0.5, the current code returns a shortened version of the
top retrieved human-written answer. It does not check claim-to-source support
in either the generated or extractive answer. The raw examples behind the
historical aggregate are unavailable, so the causes of its fallback rate
cannot be verified.

## Limitations

- StackSample is historical, so some answers may be outdated.
- The known-item task does not fully represent natural user queries.
- The faithfulness metric is an embedding-based proxy, not official RAGAS.
- The answerability evaluation set is relatively small.
- One tested out-of-domain query incorrectly passed the answerability gate.
- Generation is the dominant latency component.
- The extractive fallback rate is high.
