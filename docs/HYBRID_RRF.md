# A2 — Dense and BM25 reciprocal rank fusion

A2 combines the exact Flat-384 retriever with the A1 BM25 baseline on the
frozen C2 title-to-body known-item queries. It uses the same 520,000 document
IDs and single-positive labels for both sources.

## Fixed method

The [pinned configuration](../configs/hybrid.json) sets the reciprocal rank
fusion constant to **60**. Each source contributes `1 / (60 + rank)` for a
one-based rank. A document appearing in both sources receives both scores.
The [fusion code](../src/codeanswer/fusion.py) deduplicates by `doc_id`, sorts
by descending fused score, breaks exact ties by ascending `doc_id`, and pads
short output lists with `-1`. Source rankings with duplicate IDs or invalid
padding are rejected.

The development experiment compares source candidate depths **20** and
**40**, always returning up to 40 fused results. The test run uses the fixed
depth-40 setting. This keeps the test split out of the depth choice. The
evaluator calculates Recall@10/40, MRR@10, and nDCG@10 with the C2 labels.
For union contribution, `dense_only` counts positives present only in the
dense source's top candidates, `bm25_only` only in the lexical source, `both`
in both, and `neither` in neither. Union coverage can exceed fused Recall@40
because RRF may place a positive below the final top 40.

The [evaluation runner](../scripts/evaluate_hybrid.py) checks the frozen split
and corpus hashes, all dense index vectors against the embedding file, model
revision, BM25 index hash and configuration, and every saved A1 query ID and
label before fusion. Dense query vectors are encoded once. Latency samples
run live BM25, FAISS, and RRF sequentially for 100 queries after three
warmups. They exclude MiniLM encoding, corpus loading, and CSV writing.

## Reproduce

Build the C2 and A1 artifacts first, then run:

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/evaluate_hybrid.py --split development --device cuda
PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/evaluate_hybrid.py --split test --device cuda
```

The output under `artifacts/evaluation/frozen_v1/hybrid_<split>/` contains
the dense top-40 rankings, fused per-query rankings, quality and union
metrics, latency, source hashes, settings, and environment details. Tracked
copies in `results/evaluation/` omit query text and large index artifacts.

## Results

The full development results compare the two fixed depths with `k=60`:

| Source depth | Recall@10 | Recall@40 | MRR@10 | nDCG@10 | Union recall | Hybrid search p50 / p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.762500 | 0.847500 | 0.596608 | 0.636665 | 0.847500 | 399.450 / 722.697 ms |
| 40 | 0.752500 | 0.857500 | 0.594403 | 0.632552 | 0.892500 | 396.791 / 699.559 ms |

At depth 20, 64 positives appeared only in dense candidates and 31 only in
BM25 candidates. At depth 40, the respective counts were 64 and 24. The
predefined depth-40 test configuration was retained; the test set was not
used to tune the constant or candidate depth.

| Frozen test system | Recall@10 | Recall@40 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense Flat-384 | 0.743333 | 0.828333 | 0.587885 | 0.625245 |
| BM25 | 0.627500 | 0.726667 | 0.498723 | 0.529569 |
| Hybrid RRF, depth 40 | 0.772500 | 0.849167 | 0.609828 | 0.648677 |

The hybrid recovered 35 more positives in the first 10 and 25 more in the
first 40 than dense alone. At source depth 40, 816 positives appeared in
both lists, 178 only in dense, 56 only in BM25, and 150 in neither. The union
covered 1,050/1,200 queries (0.875000); RRF retained 1,019 in its top 40.
Hybrid search p50 was **399.954 ms** and p95 **648.974 ms** on 100 queries.
The regenerated dense top-10 rankings were identical to the C2 baseline for
all 1,200 test queries.

The [development result](../results/evaluation/a2_development_results.json)
and [test result](../results/evaluation/a2_test_results.json) record settings,
source hashes, metrics, and output checksums. Per-query files retain the
[development dense](../results/evaluation/a2_development_dense_top40.csv),
[development depth-20](../results/evaluation/a2_development_hybrid_depth20.csv),
[development depth-40](../results/evaluation/a2_development_hybrid_depth40.csv),
[test dense](../results/evaluation/a2_test_dense_top40.csv), and
[test hybrid](../results/evaluation/a2_test_hybrid_depth40.csv) rankings.
The task uses in-corpus known-item labels; it does not measure answer quality
or general relevance to new questions.
