# A1 — BM25 lexical baseline

The A1 baseline indexes the same 520,000 `question_body` values and `doc_id`s
as C2. It uses the local SQLite FTS5 implementation of BM25. The pinned
[configuration](../configs/bm25.json), [index builder](../scripts/build_bm25_index.py),
and [evaluation CLI](../scripts/evaluate_bm25.py) record the full run. SQLite's
[FTS5 documentation](https://www.sqlite.org/fts5.html) defines its built-in
ranking and tokenizer behavior.

## Index and query policy

- FTS5 uses a contentless table, with `rowid = doc_id + 1`; returned IDs are
  converted back to zero-based `doc_id`s. Only question bodies are indexed.
- `unicode61` lowercases terms, removes Latin diacritics, and keeps `_` inside
  identifiers. There is no stemming or stopword removal. Punctuation such as
  `.`, `:`, `+`, and `/` splits tokens, so version strings and symbols are not
  preserved as single terms.
- The query parser extracts unique Unicode word tokens and joins escaped terms
  with `OR`. This favors recall, but common terms can yield many candidates.
- FTS5's built-in BM25 uses `k1=1.2` and `b=0.75`. Numerically lower signed
  scores rank higher. Exact ties use ascending `rowid`, hence ascending
  `doc_id`. Unfilled ranking slots use `-1` and never count as relevant.
- The build streams 4,096 Parquet rows at a time, requires contiguous ordered
  `doc_id`s, optimizes and checks the FTS5 index, and refuses to overwrite an
  existing index. The sidecar records corpus/config/index SHA-256 values,
  SQLite version, document count, build time, and index size. The evaluator
  verifies the hashes, SQLite version, and document count before searching.

## Reproduce

Use the same [C2 corpus and frozen split](EVALUATION_PROTOCOL.md):

```bash
PYTHONPATH=src python scripts/build_bm25_index.py
PYTHONPATH=src python scripts/evaluate_bm25.py \
  --split development --limit 20 \
  --output-dir artifacts/evaluation/frozen_v1/bm25_smoke
PYTHONPATH=src python scripts/evaluate_bm25.py --split development
PYTHONPATH=src python scripts/evaluate_bm25.py --split test
```

The evaluator ranks at 40 once per query, then computes Recall@10, Recall@40,
MRR@10, and nDCG@10 from the C2 single-positive labels. It measures 10- and
40-deep search latency separately on 100 queries after three warmups. The
timing includes Python query tokenization and SQLite search, but excludes
corpus loading and result serialization. The local output contains all 40
ranked IDs and scores for each query, plus raw environment and artifact
provenance. Tracked result files omit question text and large index data.

## Development inspection

The development split has 400 queries. Its Recall@10 is 0.650000 and
Recall@40 is 0.732500. The exact-term slices are descriptive and overlap:

| Title pattern | Queries | Recall@40 |
|---|---:|---:|
| Underscore identifier | 9 | 1.000000 |
| Version number | 5 | 0.800000 |
| `error`, `exception`, or `traceback` | 14 | 0.785714 |

For example, an `EXC_BAD_ACCESS` query found its own body at rank 2. A query
about Android 4.1 missed at 40 even though its body contains `4.1`; decimal
punctuation splits the version into common numeric tokens. This inspection
does not establish performance on all exact identifiers or errors. The test
split was not used to choose tokenization or ranking settings.

## Frozen test result

The full test split has 1,200 queries, matching the C2 Flat-384 baseline's
query IDs and positive `doc_id`s. The two systems were evaluated on the same
frozen corpus and labels. Flat-384 only has a recorded ranking depth of 10.

| System | Recall@10 | Recall@40 | MRR@10 | nDCG@10 | Search p95 @10 | Index size |
|---|---:|---:|---:|---:|---:|---:|
| Flat-384 | 0.743333 | — | 0.587885 | 0.625245 | 88.379 ms | 761.72 MiB |
| BM25 | 0.627500 | 0.726667 | 0.498723 | 0.529569 | 614.834 ms | 236.73 MiB |

BM25 search p50 was 351.803 ms at depth 10 and 368.463 ms at depth 40;
its depth-40 p95 was 626.449 ms. The full-corpus index build took 30.089 s.
The observed BM25 Recall@10 is 0.115833 lower than Flat-384 on this paired
test. The latency figures are search-only but use different query preparation:
BM25 includes term extraction, while Flat-384 starts with encoded vectors.

The [index manifest](../results/evaluation/a1_bm25_index_manifest.json),
[test results](../results/evaluation/a1_bm25_test_results.json),
[test rankings](../results/evaluation/a1_bm25_test_per_query.csv),
[development results](../results/evaluation/a1_bm25_development_results.json),
and [development rankings](../results/evaluation/a1_bm25_development_per_query.csv)
retain hashes, settings, timings, and query-level evidence. The large SQLite
index remains in ignored local artifacts. This is in-corpus title-to-own-body
known-item retrieval, not a relevance judgment for new user questions. It
does not measure answer quality.
