# C2 — Frozen retrieval evaluation protocol

**Status: DONE.** The frozen split and exact Flat-384 baseline were run on the
full 520,000-row corpus. The tracked [C2 result summary](../results/evaluation/c2_frozen_baseline_summary.json),
[split ID and label table](../results/evaluation/c2_query_ids.csv),
[per-query test metrics](../results/evaluation/c2_flat_384_test_per_query.csv),
and [frozen manifest](../results/evaluation/c2_frozen_manifest.json) retain the
results and provenance without copying question text or large model artifacts
into Git. The values in the historical results tables use a different query
selection and must not be compared as paired scores.

## Inputs and split policy

`scripts/prepare_evaluation.py` streams the corpus Parquet file, requiring
ordered contiguous `doc_id`s and unique `question_id`s. It chooses 2,000
nonempty titles with the smallest 64-bit BLAKE2b keys of
`seed:question_id` (seed `20260714` by default). This differs from the old
DuckDB `hash(question_id + seed)` query selection; old and new scores are not
directly paired. Each query has one relevant document: its own question body.

The default requested split sizes are 400 development, 400 calibration, and
1,200 test queries. Exact duplicate normalized titles or nonempty bodies are
joined into one family and assigned to one split. Families can make actual
split sizes differ slightly from the requested sizes; the manifest records
actual counts, duplicate rows, and family counts. This guards against exact
text duplicates within the selected set. It does not detect semantic near
duplicates or make the in-corpus known-item task representative of user
questions. The calibration split is for retrieval settings; D2 will handle
separate RAG answerability labels.

The output contains `development.csv`, `calibration.csv`, `test.csv`, and a
schema-versioned `manifest.json`. Rows store query/question ID, title, one
relevant `doc_id`, family ID, and normalized title/body hashes. The manifest
records seed, selection/family rules, corpus SHA-256 and row count, each split's
SHA-256 and counts, and query-to-corpus overlap (all queries by design). The
loader checks every split and the corpus checksum, rejects duplicate IDs,
family overlap, empty queries, invalid labels, and mismatched counts. The
metric helper rejects empty query sets and cutoffs beyond ranking depth.

## Reproduce the full run

Use the StackSample `Questions.csv` and `Answers.csv` under `data/stacksample/`.
The source CSV SHA-256 values, corpus hash, model revision, index hash, and
package versions are recorded in the result summary. The builder refuses to
overwrite an existing frozen set. Keep the generated files and manifest
together; their checksums bind the benchmark.

```bash
PYTHONPATH=src python scripts/build_corpus.py --source-dir data/stacksample
PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/generate_embeddings.py --device cuda --batch-size 64
PYTHONPATH=src python scripts/build_flat_index.py

PYTHONPATH=src python scripts/prepare_evaluation.py \
  --corpus artifacts/corpus/qa_corpus.parquet \
  --output-dir artifacts/evaluation/frozen_v1

PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/evaluate_retrieval.py \
  --corpus artifacts/corpus/qa_corpus.parquet \
  --index artifacts/indexes/flat_ip_384.faiss \
  --evaluation-manifest artifacts/evaluation/frozen_v1/manifest.json \
  --split test \
  --device cuda
```

Use development for debugging and calibration for retrieval settings. The
baseline above was run once on `--split test` after the protocol and settings
were fixed. Its default output is beside the frozen manifest in
`flat_384_<split>/`, containing
`per_query_metrics.csv` and `results.json`. The per-query CSV includes query ID,
family ID, label, rank, and retrieved IDs. JSON records aggregate metrics,
latency, exact command, settings, machine/package details, corpus/index/embedding
checksums, and model revision when available. Search latency is still
search-only, not complete RAG latency.

Frozen exact evaluation requires both existing index and embedding sidecar
manifests. It checks their paths, shapes, model name, max sequence length,
corpus checksum when recorded, and **all** index vectors against the embedding
file. New embedding runs record the corpus SHA-256 and reject resume if that
file changes. Legacy embedding manifests lack a corpus checksum: results are
marked `legacy_path_only`, which cannot prove that those embeddings were
created from the current corpus bytes. If the model revision cannot be read
from the local model, results record `unknown`; this must be resolved before a
fully pinned benchmark claim.

## Completed C2 check

The split was generated twice in separate directories; both manifests and all
three split CSVs were byte-identical. There were 400 development, 400
calibration, and 1,200 test queries, with no query ID or family crossing splits.
The 2,000 selected queries had zero exact duplicate normalized title or body
rows. The corpus hash matched the embedding manifest, every vector in the
exact FAISS index matched the embedding file, and the query encoder revision
matched the embedding model revision.

| Frozen Flat-384 test metric | Result |
|---|---:|
| Recall@1 | 0.515000 |
| Recall@5 | 0.685833 |
| Recall@10 | 0.743333 |
| MRR@10 | 0.587885 |
| nDCG@10 | 0.625245 |
| Search p95, 100 single queries | 88.379 ms |

These are 1,200 in-corpus title-to-own-body known-item queries, with one
positive document each. The latency measures search only. The full frozen
CSV files, raw per-query output, and large corpus/vector/index artifacts remain
local and ignored by Git. The committed ID/label table and metrics preserve
the evaluation evidence without source text. The tracked manifest is a
provenance snapshot; recreating the full split CSVs requires the corpus.
