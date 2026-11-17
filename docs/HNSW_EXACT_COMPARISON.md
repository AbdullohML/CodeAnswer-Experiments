# B2 — HNSW versus exact search

**Status: DONE.** This comparison uses the same normalized corpus and query
vectors within each dimension. The 128-dimensional vectors and exact index
come from B1's shared PCA fit; the 384-dimensional vectors and exact index
come from the verified C2 baseline. In each space, FAISS `IndexFlatIP` is the
reference for `IndexHNSWFlat`. Both return 40 documents for the same 1,200
frozen C2 test titles and single-positive judgments.

The fixed [settings](../configs/hnsw_exact.json) use `M=16`,
`efConstruction=100`, and `efSearch=128`, matching the current project
configuration. The [result manifest](../results/evaluation/b2/results.json)
records the model revision, B1 lineage, index hashes, build times, hardware,
packages, and exact command. The ignored HNSW indexes reside in
`artifacts/b2/`. The runner checks the live exact top-40 IDs against every
corresponding B1 row before reporting an HNSW result.

## Full frozen test

| Dimensions | Method | Recall@10 | Recall@40 | Search p50 / p95 ms | Index MiB |
|---:|---|---:|---:|---:|---:|
| 128 | Exact | 0.710000 | 0.810000 | 16.569 / 18.108 | 253.91 |
| 128 | HNSW | 0.700000 | 0.792500 | 0.561 / 0.715 | 325.45 |
| 384 | Exact | 0.743333 | 0.828333 | 58.800 / 71.804 | 761.72 |
| 384 | HNSW | 0.732500 | 0.814167 | 0.896 / 1.283 | 833.27 |

Within 128 dimensions, HNSW loses 0.010000 Recall@10 and 0.017500
Recall@40 against exact search. Within 384, the losses are 0.010833 and
0.014166. HNSW build times were 67.111 seconds at 128 and 143.080 seconds
at 384; its indexes are larger because they also store graph links. These
are one-run measurements on this laptop.

The mean fraction of **exact top-k IDs** returned by HNSW is 0.989750 at
k=10 and 0.975417 at k=40 for 128 dimensions, and 0.986083 and 0.973083
for 384. This ANN overlap is a ranking agreement measure. Recall@k instead
asks whether the single judged document was returned. The
[comparison table](../results/evaluation/b2/comparison.csv) and
[per-query ranks and IDs](../results/evaluation/b2/per_query.csv.gz) retain
both measurements and their paired query IDs.

The B1 exact 384-to-128 Recall@10 change is a 0.033333 loss from PCA and
renormalization under exact search. The additional exact-128-to-HNSW-128
loss is 0.010000 from approximate search at `efSearch=128`. This separates
the two effects without treating HNSW overlap as a relevance label. B3 will
measure how `efSearch` changes the ANN trade-off.

Search latency times 100 identical frozen-order queries sequentially after
three warmups, with vectors precomputed. It excludes query encoding, PCA
projection, index loading, and corpus fetch. The full rankings use batched
top-40 search. The 200-query 128-dimensional validation under ignored
`artifacts/evaluation/frozen_v1/b2_smoke/` was a smoke run.

## Reproduce

With the C2 and B1 local artifacts and cached MiniLM model available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  python scripts/evaluate_hnsw_exact.py --smoke-limit 200 \
  --only-dimension 128 --output-dir artifacts/evaluation/frozen_v1/b2_smoke

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  python scripts/evaluate_hnsw_exact.py --output-dir results/evaluation/b2
```

This remains in-corpus title-to-own-body known-item retrieval. It does not
measure relevance to new user questions or generated answer quality.
