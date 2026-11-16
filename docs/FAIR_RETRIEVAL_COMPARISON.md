# A3 — Paired retrieval comparison

The A3 comparison evaluates BM25, exact Flat-384 dense retrieval, and their
depth-40 RRF hybrid on the same 1,200 frozen C2 test titles. Each title has
one positive `doc_id`: the body from its own question. The full run verifies
the A1 and A2 result hashes and row IDs against the C2 manifest, then reruns
all 1,200 live top-40 rankings. It rejects any mismatch with the saved
per-query rankings before reporting metrics.

## Protocol

The [comparison configuration](../configs/fair_comparison.json) fixes the
retrieval depth at 40, Recall cutoffs at 1, 5, 10, and 40, MRR and nDCG at
10, and a shared 100-query latency prefix. The hybrid uses the
[A2 configuration](../configs/hybrid.json) with RRF constant 60. All systems
use the same corpus bytes, C2 query order and relevance labels, and their
saved model/index provenance is checked before comparison.

For the latency and sequential-throughput sample, each method starts from
the **raw title** on the same 100 queries after three warmups. BM25 time
includes query tokenization and SQLite search. Dense time includes MiniLM
encoding and FAISS search. Hybrid time includes both and RRF. These are
single-query measurements; the full-set rankings use batched MiniLM inference
and are verified separately. The comparison uses ranks and label-based
metrics. BM25 scores, cosine similarities, and RRF scores are never compared
on a common numeric scale.

The exact-term slice is chosen without looking at outcomes: take the first
10 frozen-order titles matching each of the A1 patterns for underscore
identifiers, version numbers, and `error`/`exception`/`traceback`, then
deduplicate the IDs. It is descriptive and may overlap by category. The
slice is a subset of the same test queries and labels, so it supports paired
inspection but is too small for a broad generalization.

## Reproduce

With the C2, A1, and A2 artifacts available, run:

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/compare_retrievers.py \
  --evaluation-manifest artifacts/evaluation/frozen_v1/manifest.json \
  --split test --device cuda \
  --output-dir artifacts/evaluation/frozen_v1/fair_test
```

The historical mode of `compare_retrievers.py` remains available without
`--evaluation-manifest`. The A3 output contains the paired per-query CSV,
exact-term slice CSV, system table CSV, and JSON with source hashes, settings,
quality metrics, latency, and environment details.

## Results and limits

| Full frozen test, 1,200 queries | Recall@1 | Recall@5 | Recall@10 | Recall@40 | MRR@10 | nDCG@10 | p50 / p95 | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.439167 | 0.576667 | 0.627500 | 0.726667 | 0.498723 | 0.529569 | 336.531 / 596.505 ms | 2.95 |
| Dense Flat-384 | 0.515000 | 0.685833 | 0.743333 | 0.828333 | 0.587885 | 0.625245 | 52.758 / 56.232 ms | 18.82 |
| Hybrid RRF | 0.537500 | 0.703333 | 0.772500 | 0.849167 | 0.609828 | 0.648677 | 393.128 / 653.776 ms | 2.53 |

Compared with dense, hybrid recovered 78 additional positives at rank 10
and lost 43, for a net gain of 35. At rank 40, it recovered 48 and lost 23,
for a net gain of 25. These are paired counts on identical query IDs and
labels. The full 1,200 live top-40 rankings for each method matched the
saved A1/A2 outputs exactly.

| Exact-term slice, 29 queries | Recall@10 | Recall@40 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.827586 | 0.862069 | 0.675616 | 0.711655 |
| Dense Flat-384 | 0.758621 | 0.793103 | 0.632512 | 0.662765 |
| Hybrid RRF | 0.827586 | 0.896552 | 0.740558 | 0.761903 |

The slice has 10 version-number titles, 10 underscore-identifier titles,
and 10 error-word titles, with one title in two categories. It is a small,
outcome-blind diagnostic subset, not evidence of a population-wide effect.

The [result JSON](../results/evaluation/a3_test_results.json),
[paired per-query CSV](../results/evaluation/a3_test_paired_per_query.csv),
[exact-term slice CSV](../results/evaluation/a3_test_exact_term_slice.csv),
and [system table](../results/evaluation/a3_test_system_table.csv) retain
the exact IDs, labels, ranks, configuration hashes, environment, and
measured results without query text. This is an in-corpus known-item test.
It does not label arbitrary relevant answers for real user questions or
measure generated-answer quality. The latency sample is one run on one
machine; caches, device load, and single-query versus batch execution affect
timings. Prior C2/A1/A2 latency figures used search-only boundaries, so they
should not be read as the raw-title timing reported here.
