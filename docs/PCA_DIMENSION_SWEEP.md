# B1 — PCA dimension sweep

**Status: DONE.** This experiment compares exact inner-product search on
normalized 384-dimensional MiniLM embeddings with 256-, 128-, and
64-dimensional PCA projections. All four systems search the same 520,000
document IDs with FAISS `IndexFlatIP` and use the same 1,200 frozen C2 test
titles and single-positive judgments. The exact 384-dimensional run reproduces
the A3 dense quality metrics.

## Fit and evaluation protocol

The runner verifies the C2 corpus, source embedding, exact index, and frozen
split hashes before evaluation. It encodes every query with the cached
`sentence-transformers/all-MiniLM-L6-v2` revision recorded for the corpus,
using the same 256-token maximum sequence length. The 384-dimensional query
vectors are reused for all four conditions.

PCA is fitted once with randomized SVD and seed `20260714` on 100,000 unique
document IDs sampled without replacement from the 520,000 source embeddings.
The exact fit order is saved in [fit_sample_ids.csv.gz](../results/evaluation/b1/fit_sample_ids.csv.gz).
The fitted 256-component basis is shared: the 128- and 64-dimensional
conditions retain its first 128 or 64 components. Each projected document and
query vector is L2 normalized before inner-product search. This nested basis
isolates the number of retained components; it differs from the previously
used independently fitted PCA-128/HNSW configuration.

The fixed [settings](../configs/pca_sweep.json) and [result manifest](../results/evaluation/b1/results.json)
record the commands, model revision, package versions, hardware, artifact
hashes, and sample identity. The dimension manifests retain each index and
embedding hash. Reduced embeddings, PCA model, and exact indexes are under
ignored `artifacts/b1/`; they are built one dimension at a time and reused
only after identity checks. The [per-query rankings](../results/evaluation/b1/per_query.csv.gz)
retain all 40 IDs and ranks for every dimension.

## Results

| Dimensions | PCA variance retained | Recall@10 | Recall@40 | MRR@10 | Index MiB | Transform p95 ms | Search p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 384, no PCA | 1.000000 | 0.743333 | 0.828333 | 0.587885 | 761.72 | 0.000 | 80.618 |
| 256 | 0.970214 | 0.737500 | 0.822500 | 0.581411 | 507.81 | 0.293 | 47.200 |
| 128 | 0.798672 | 0.710000 | 0.810000 | 0.535694 | 253.91 | 0.277 | 17.834 |
| 64 | 0.609507 | 0.602500 | 0.732500 | 0.442380 | 126.95 | 0.219 | 8.859 |

Relative to exact 384 search, the Recall@10 loss is 0.005833 at 256,
0.033333 at 128, and 0.140833 at 64. Paired rank-10 gains/losses are 6/13,
16/56, and 14/183 respectively; a projection can improve individual queries
even when aggregate quality falls. The [comparison CSV](../results/evaluation/b1/comparison.csv)
and [trade-off plot](../results/evaluation/b1/tradeoff.svg) provide the full
table and curve.

Each system uses exact search in its own vector space. Thus its within-space
ANN loss is zero by construction, while the differences versus 384 dimensions
measure PCA projection and renormalization effects. B2 will compare exact and
HNSW search within a fixed dimensional space. The index sizes above exclude
the embedding files, Python process, and model memory.

Latency uses 100 identical frozen-order queries after three warmups. Search
p95 is a sequential top-40 FAISS call on precomputed vectors; transform p95
times one query's PCA projection and normalization separately. Query encoding,
index loading, and corpus fetching are excluded. These are one-run measurements
on this laptop, so they should not be mixed with earlier top-10 search or
end-to-end timings. The 200-query 384/64 validation under ignored
`artifacts/evaluation/frozen_v1/b1_smoke/` was a smoke run, not the reported
benchmark.

## Reproduce

With the C2 corpus, embeddings, exact 384 index, frozen split, and cached
MiniLM model available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  python scripts/evaluate_pca_sweep.py --smoke-limit 200 \
  --only-dimension 64 --output-dir artifacts/evaluation/frozen_v1/b1_smoke

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  python scripts/evaluate_pca_sweep.py --output-dir results/evaluation/b1
```

This remains an in-corpus title-to-own-body known-item benchmark. It does not
establish relevance for new user questions, answer quality, or the best
production dimension without the later indexing and end-to-end evaluations.
