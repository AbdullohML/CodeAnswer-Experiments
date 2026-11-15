# C2 — Frozen retrieval evaluation protocol

**Status: IN PROGRESS.** The split builder, checksum validation, exact-index
lineage checks, frozen baseline runner, and synthetic smoke test are implemented.
The real 520,000-row corpus, 384-dimensional embeddings, and FAISS index are
absent from this checkout. Model cache availability was not verified. No full
frozen split or baseline score has
been produced. The values in `results/` remain historical and are not results
of this protocol.

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

## Run when the full artifacts are available

Use local paths to the same corpus, embeddings, and Flat-384 index build. The
builder refuses to overwrite an existing frozen set. Keep the generated files
and manifest together; their checksums bind the benchmark.

```bash
PYTHONPATH=src python scripts/prepare_evaluation.py \
  --corpus /path/to/qa_corpus.parquet \
  --output-dir artifacts/evaluation/frozen_v1

PYTHONPATH=src python scripts/evaluate_retrieval.py \
  --corpus /path/to/qa_corpus.parquet \
  --index /path/to/flat_ip_384.faiss \
  --evaluation-manifest artifacts/evaluation/frozen_v1/manifest.json \
  --split development \
  --device cuda
```

Use development for debugging and hyperparameter choices. Run the final
baseline once on `--split test` after the protocol and settings are fixed. Its
default output is beside the frozen manifest in `flat_384_<split>/`, containing
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

## Completion gate

On the real corpus, generate and retain the frozen files twice in separate
locations and compare manifests and split checksums. Check actual split sizes,
zero ID/family intersection, duplicate rates, and corpus/index/embedding
lineage. Run the exact baseline on the frozen test set, save per-query output,
and report Recall@10, MRR@10, nDCG@10 and the exact configuration. Until then,
C2 remains IN PROGRESS; synthetic fixtures are only implementation checks.
