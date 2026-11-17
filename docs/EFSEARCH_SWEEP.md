# B3 — HNSW efSearch sweep

**Status: DONE.** The sweep reuses B2's fixed 128-dimensional HNSW index,
PCA fit, and frozen C2 test query vectors. It compares `efSearch` 16, 32, 64,
and 128 from the historical sweep, then 256 because B2's top-40 ANN overlap
at 128 was still below 1.0. The exact FlatIP top-40 IDs from B2 are the
same-space ANN reference. All results use the same 1,200 query IDs, corpus,
and single-positive relevance judgments.

| efSearch | Recall@10 | Recall@40 | ANN overlap@10 | ANN overlap@40 | Search p50 / p95 ms | Sequential QPS |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.630833 | 0.695833 | 0.865167 | 0.736979 | 0.110 / 0.178 | 8547.28 |
| 32 | 0.670000 | 0.747500 | 0.936583 | 0.861167 | 0.194 / 0.285 | 4935.07 |
| 64 | 0.690833 | 0.775833 | 0.973583 | 0.939083 | 0.350 / 0.514 | 2784.65 |
| **128** | **0.700000** | **0.792500** | **0.989750** | **0.975417** | **0.584 / 0.872** | **1638.19** |
| 256 | 0.705000 | 0.801667 | 0.995667 | 0.990979 | 1.004 / 1.303 | 980.72 |

The same-space exact reference reaches Recall@10 0.710000 and Recall@40
0.810000. At the **currently configured `efSearch=128`**, the ANN losses are
0.010000 and 0.017500. Moving to 256 recovers 0.005000 Recall@10 (six of
1,200 queries) and 0.009167 Recall@40 (eleven queries), while search p95
rises 0.431 ms, about 49%. The result is not yet saturated at 128. We retain
128 as the existing setting; selecting a replacement from test outcomes would
use the frozen test set for tuning. A development-set selection and a fresh
test would be needed to change the production setting on this evidence.

ANN overlap is the mean fraction of exact top-k IDs returned by HNSW; it is
separate from whether the single judged document appears in the result.
At 128, the complete per-query rankings have the same IDs as
B2's saved HNSW rankings. The [comparison CSV](../results/evaluation/b3/comparison.csv),
[paired query outputs](../results/evaluation/b3/per_query.csv.gz),
[quality/latency curve](../results/evaluation/b3/tradeoff.svg), and
[result manifest](../results/evaluation/b3/results.json) save the measurements,
settings, checksums, model revision, and exact command.

Search times use the same first 100 frozen-order queries after three warmups,
one top-40 HNSW search per query with precomputed vectors. QPS is 100 divided
by the sum of those measured search durations. Encoding, PCA projection,
index load, and corpus fetch are outside this boundary. The 200-query run
under ignored `artifacts/evaluation/frozen_v1/b3_smoke/` was a smoke test,
not the full benchmark. These times are one run on this laptop; the earlier
historical sweep in [EXPERIMENTS.md](EXPERIMENTS.md) remains historical evidence
with a different query protocol.

## Reproduce

With the C2, B1, and B2 local artifacts and cached MiniLM model available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  python scripts/compare_retrievers.py --ef-sweep --limit 200 \
  --output-dir artifacts/evaluation/frozen_v1/b3_smoke

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  python scripts/compare_retrievers.py --ef-sweep \
  --output-dir results/evaluation/b3
```

This remains in-corpus title-to-own-body known-item retrieval. It does not
measure relevance to new user questions or answer quality.
