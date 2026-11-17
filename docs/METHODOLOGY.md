# Architecture and methodology

This page separates the configured answer command, the frozen retrieval
experiments, and the newer traced RAG evaluations. Those runs use the same
StackSample corpus; they do not all use the same index, query set, or
threshold. D1 is a separate synthetic-fixture audit.

## Corpus and query definitions

[`build_corpus.py`](../scripts/build_corpus.py) reads StackSample
`Questions.csv` and `Answers.csv`. It keeps the highest-scored answer for each
question, breaking answer-score ties by answer ID, strips HTML, selects
520,000 answered questions in a seeded deterministic order, and assigns
contiguous `doc_id`s. The indexed text is the **question body**. The attached
answer is available to the generator and extractive fallback; answer text is
not the dense or BM25 retrieval field. The corpus Parquet file and raw CSVs
remain local. The [corpus manifest](../artifacts/corpus/qa_corpus.manifest.json)
and [C2 frozen manifest](../results/evaluation/c2_frozen_manifest.json) record
their identities.

The [C2 protocol](EVALUATION_PROTOCOL.md) selects 2,000 nonempty **titles**
from the indexed corpus. Exact duplicate normalized title/body families stay
within one split: 400 development, 400 calibration, and 1,200 test queries in
the completed run. Each title has exactly one judged positive: its own body's
`doc_id`. Thus Recall@k measures whether that one known item appears in the
first k results. An unjudged document may still be useful; these labels do
not measure relevance for new user questions. All query titles are from the
indexed corpus, so title-aware reranking can benefit from a same-question
match. The older result table in the [README](../README.md) used a different
query selection and cannot be paired with C2 results.

## Artifact lineage and answer paths

| Stage | Input and output | Used by |
| --- | --- | --- |
| Corpus | StackSample CSVs → `artifacts/corpus/qa_corpus.parquet` | Every experiment and answer path |
| Dense source | Question bodies → normalized 384-dimensional MiniLM vectors → exact `artifacts/indexes/flat_ip_384.faiss` | C2, A2/A3, B1 baseline |
| Lexical source | Question bodies → SQLite FTS5 BM25 index | A1–A3 and D3 BM25/hybrid |
| B1/B2 vector branch | Saved 256-component PCA basis, using its first 128 components and L2 normalization → B2 HNSW-128 index | B2–B4 and D2/D3 dense branch |
| Standard answer command | Separately built `artifacts/optimized/pca_384_to_128.joblib` and `hnsw_pca_128.faiss`; threshold `0.449093` in `configs/default.yaml` | `scripts/ask_codeanswer.py` |
| Traced RAG runs | B1 PCA + B2 HNSW; D2 calibration threshold `2.803920269`; fixed reranker, three sources and Qwen settings | D2/D3 only |

The [embedding manifest](../artifacts/embeddings/question_body_minilm_384.f32.manifest.json),
[exact-index manifest](../artifacts/indexes/flat_ip_384.faiss.manifest.json),
[B1 result](../results/evaluation/b1/results.json),
[B2 result](../results/evaluation/b2/results.json), and
[D2 run manifest](../results/evaluation/d2/run_local_b1b2/run_manifest.json)
provide the saved artifact hashes, model identity and settings. Large vectors,
FAISS files, corpus data and local model files are ignored by Git. D3 checks
the D2 label, config, corpus, PCA, index and reranker hashes before running;
see its [protocol and trace files](RAG_RETRIEVER_COMPARISON.md).

For one question, the [standard pipeline](../src/codeanswer/rag.py) embeds the
query, projects it to 128 dimensions, and asks HNSW for up to 40 body IDs.
HNSW is **the dense search implementation**, not a stage after dense search.
It fetches their titles, bodies and answers; the cross-encoder scores
query–title-plus-body pairs. A small answer-vote bonus helps order the
sources; the **raw cross-encoder score of the top-ranked source** decides
whether to abstain. If it passes the configured threshold, the top three
reranked sources are supplied to Qwen. The citation gate measures numeric
marker validity and coverage of long nonempty lines, then either keeps the
generated text or returns an extract from the top source. It does not verify
source support. See the [D1 gate audit](FALLBACK_CITATION_AUDIT.md).

D3 varied **only candidate selection** on the same 28 D2 test questions:
HNSW top 40, body-only BM25 top 40, or RRF of both top-40 lists into one
top-40 list. All three then used the same reranker, threshold, three-source
limit, prompt, generator and citation gate. A2/A3's hybrid instead used
**exact Flat-384** as its dense branch on the frozen retrieval test. Its
retrieval scores and D3's RAG outcomes have different scopes.

## Measurement boundaries

| Evidence | Queries and positive labels | Timed work | What it establishes |
| --- | --- | --- | --- |
| [Historical aggregates](../results/experiment_summary.json) | Older dynamic title selection; RAG examples and labels unavailable | Saved aggregate timing only | Historical observations; per-example RAG claims cannot be audited |
| [C2](EVALUATION_PROTOCOL.md), [A1](BM25_BASELINE.md), [A2](HYBRID_RRF.md) | Frozen title-to-own-body known items; 1,200 test queries | Their documented search-only boundaries | Retrieval rankings and single-positive metrics, not answer quality |
| [A3](FAIR_RETRIEVAL_COMPARISON.md) | Same 1,200 test items across BM25, exact dense and RRF; 100-query timing prefix | Raw-title tokenization/encoding, search and fusion; excludes corpus fetch and RAG | Paired retrieval comparison under one latency boundary |
| [B1](PCA_DIMENSION_SWEEP.md), [B2](HNSW_EXACT_COMPARISON.md), [B3](EFSEARCH_SWEEP.md) | Same frozen test titles and single-positive labels | Search on precomputed vectors; query encoding, fetch and RAG excluded | PCA loss, same-space HNSW loss and efSearch trade-off |
| [B4](RESOURCE_COSTS.md) | Same first 100 frozen test titles, twice | Encoding, optional PCA, top-40 FAISS search and 40-row Parquet fetch; no reranker or Qwen | Build, memory and time-to-fetched-rows costs on one laptop |
| [D1](FALLBACK_CITATION_AUDIT.md) | 11 synthetic fixtures, no real model | No representative latency benchmark | Exact citation-gate branches and failure cases |
| [D2](RAG_EVALUATION.md), [D3](RAG_RETRIEVER_COMPARISON.md) | 20 indexed titles labeled *answer*, eight plainly nonprogramming controls labeled *abstain* on held-out test | Complete sequential local calls, with separate stage timings | Behavior on an easy 28-question protocol, fallback and format; no factual-correctness label |

Search p95, time-to-fetched-rows p95 and complete-response p95 have different
boundaries and must not be substituted for one another. Stage percentiles
cannot be added to obtain total-response percentiles. Index file size is not
process RSS. BM25 ranks, vector similarities, RRF scores and raw reranker
scores are on different scales. The D2 threshold was chosen from its
calibration labels and is not the standard command's default threshold.

## Interpretation limits

The D2/D3 labels check only whether the system answered or abstained. The
in-domain titles are drawn from indexed questions and the out-of-domain
controls are far from programming; neither tests near-domain unanswerable
questions. Citation validity checks number syntax, not whether a source
supports a claim. No human answer-quality judgment was collected. The
historical faithfulness proxy is an embedding similarity, not a factual
correctness measure. Timings are single-machine observations under each
documented boundary, not a production throughput guarantee.
