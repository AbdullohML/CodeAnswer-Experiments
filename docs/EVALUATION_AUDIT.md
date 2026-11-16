# C1 — Evaluation and leakage audit

Scope: tracked code, configuration, tests, and committed result files at base
commit `298a018` (2026-11-15). This is an audit of the **recorded** experiments,
not a rerun of the 520,000-document benchmark. At audit time, `data/` and
`artifacts/` contained no dataset, embeddings, indexes, or query logs.

## What is measured

| Reported quantity | Definition supported by code | What it does not establish |
|---|---|---|
| Known-item Recall@k | Share of titles whose own question's `doc_id` appears in the first k retrieved bodies. There is exactly one labeled positive per query. | Recall over all relevant programming answers, or relevance to new user queries. |
| MRR@10 | Mean reciprocal rank of that one `doc_id`, with zero if absent from the first 10. | Ranking quality over multiple relevant documents. |
| nDCG@10 | Mean `1/log2(rank + 1)` for that one `doc_id`, with zero if absent. Ideal DCG is 1. | Graded or multi-positive nDCG. |
| `ann_overlap@10` | Mean top-10 set intersection divided by 10. For HNSW it compares with exact search **in the same PCA-128 space**. | Answer accuracy or label-based retrieval recall. Flat-PCA128 versus Flat-384 overlap also includes projection effects. |
| Behavior accuracy | Historical answer-versus-abstain aggregate for an in-domain/out-of-domain set, as described in `docs/EXPERIMENTS.md`. | Programming-answer correctness. Raw labels and runner are absent, so the exact computation is not independently checkable. |
| Citation coverage/validity | Runtime helper counts cited long lines and in-range numeric citations; the final response is scored again after fallback. | Whether cited sources support claims, or whether historical aggregate scores used this exact denominator. |
| Faithfulness proxy | Named in committed results and described as embedding-based in `docs/EXPERIMENTS.md`. | Factual faithfulness; its formula and raw judgments are absent. |

Evidence: [`data.py`](../src/codeanswer/data.py) selects the highest-scored
answer per `question_id`, then assigns local `doc_id` to sampled question rows
by a seeded question-ID ordering; it does not validate uniqueness of source
question rows. [`evaluate_retrieval.py`](../scripts/evaluate_retrieval.py)
selects 2,000 titles by seeded question-ID hash, retrieves their question
bodies, and labels only the matching `doc_id` relevant. Its query title is not
concatenated into the indexed body by the embedding pipeline; nevertheless,
title and body come from the **same question**. [`evaluation.py`](../src/codeanswer/evaluation.py)
implements the single-positive formulas above. [`compare_retrievers.py`](../scripts/compare_retrievers.py)
uses the same title selection and evaluates Flat-384, Flat-PCA128, then
HNSW-PCA128 at efSearch 16/32/64/128. It saves aggregate rows but not the
per-query rankings. [`comparison.py`](../src/codeanswer/comparison.py) defines
the set-overlap metric. [`rag.py`](../src/codeanswer/rag.py) and
[`reranking.py`](../src/codeanswer/reranking.py) define runtime citation checks,
the raw cross-encoder gate, and threshold selection; no committed CLI runs
RAG calibration or evaluation.

## Counts, overlap, and reproducibility

| Item | What can be established now | Unknown until raw data is restored |
|---|---|---|
| Corpus | 520,000 documents are reported in results and configured as the target. The SQL selects one answer per `question_id` and orders selected questions into `doc_id`s. | Actual row count, duplicate `question_id`/title/body rate, and corpus fingerprint. |
| Retrieval set | 2,000 queries are configured and documented; each query's gold document is inside the indexed corpus **by construction** (2,000/2,000 design overlap). | Actual sampled IDs/text, duplicate-query rate, near-duplicate rate, and per-query rankings for the comparison. |
| PCA fit | A seeded sample of up to 100,000 rows is drawn from the same embedding matrix that contains evaluation documents. This is transductive, unsupervised fitting, not observed label leakage. | Exact fit/evaluation ID overlap; sample indices and artifact hash are not saved in the committed files. |
| Calibration/RAG | Config requests 40 in-domain + 20 out-of-domain calibration queries and 20 + 8 RAG evaluation queries. | Actual counts, labels, split intersection, threshold selection run, raw/final answers, citation denominators, and error cases. |

The retrieval queries are in-corpus by design. This is a valid known-item
experiment, but a same-source query is easier and narrower than an independent
user question. The gold item's title is also present in the cross-encoder
input (`title + question_body`); if reranking is evaluated on these original
titles, the gold candidate can benefit from exact-title matching. That is a
direct evaluation shortcut for A4, even if candidate retrieval indexed only
bodies. It does not affect the recorded first-stage retrieval comparison,
which has no reranking stage. Calibration/test IDs are not published, so
their disjointness cannot be checked. PCA fitting uses document vectors but
no relevance labels; the larger risk is selecting PCA dimension, efSearch,
RRF settings, or threshold on the final test queries.

The corpus cleaner removes HTML tags and collapses whitespace, including code
layout. This can change token and identifier evidence for both lexical and
dense retrieval; it is a preprocessing limitation, not a measured leak. The
corpus manifest records paths and statistics, but no content checksum. Index
loading validates vector count and dimension, not that corpus, embeddings,
PCA, and index came from the same build. Mismatched 520,000-row artifacts
could silently attach the wrong answers to retrieved IDs.

## Checks performed for this audit

Using the committed [`retrieval_comparison.csv`](../results/retrieval_comparison.csv)
and 2,000 reported queries, all six Recall@1/5/10 values correspond to whole
query counts. Flat-384 has 970/1,293/1,414 hits at 1/5/10; Flat-PCA128 has
866/1,193/1,331; HNSW-PCA128 efSearch=128 has 856/1,174/1,311. The selected
HNSW Recall@10 and overlap agree with [`experiment_summary.json`](../results/experiment_summary.json).
The 0.0515 Recall@10 loss, 71.059/0.514 = 138.25 search-p95 ratio, and
57.27% index-file size reduction recompute from the committed rows. MRR,
nDCG, latency distributions, and RAG aggregates cannot be independently
recomputed without raw per-query records or a rerun. The historical behavior
accuracy of 0.964286 is numerically consistent with 27/28 correct decisions,
and 0.875 out-of-domain abstention with 7/8, matching configured set sizes;
this does not verify the missing labels or decisions.

Tiny in-memory fixtures exercised the actual helpers (no dataset or model
download): ranks `[1, 2, 3, 0]` gave Recall@1/2/3 = 0.25/0.50/0.75,
MRR@3 = 0.458333, and nDCG@3 = 0.532732. A hypothetical alternate useful
document at rank 1, with the sole labeled `doc_id` at rank 2, scores
Recall@1 = 0 and MRR@2 = 0.5; the metric cannot recognize unjudged useful
documents. Identical top-3 lists scored overlap 1, disjoint lists 0. A
four-example threshold fixture yielded accuracy, precision, recall and F1
of 0.5, as expected. These checks verify formulas, not historical runs.

Two input-validation gaps are reproducible: an empty query matrix returns
`NaN` retrieval metrics, and requesting Recall@40 from a top-2 ranking is
accepted and silently reports a value. C2 should reject empty evaluations and
cutoffs greater than retrieved depth before reporting results.

## Prevention plan for C2 and later tasks

1. Freeze query IDs/text and one-positive labels for the existing known-item
   benchmark; save corpus checksum, preprocessing version, model revision,
   embedding/PCA/index identifiers, commands and per-query rankings. Validate
   unique IDs, nonempty sets, cutoff depth, and artifact compatibility.
2. Create disjoint development, calibration and final-test query IDs; record
   exact/near-duplicate rates and any overlap with PCA fit IDs. Keep the
   final test untouched while choosing BM25 tokenization, RRF, PCA, HNSW and
   answerability settings. Report paired results on the frozen test.
3. For A4, report title-plus-body and body-only reranking separately on this
   benchmark. Use independent paraphrased or externally judged queries before
   claiming that title-visible reranking improves real-user relevance.
4. For D2, restore labeled RAG splits and a runnable harness that stores raw
   generation, final answer, fallback cause, sources and metric denominators.
   Keep citation syntax, source support, answer correctness, and
   answer/abstain behavior as separate metrics. If the historical inputs
   cannot be recovered, label the existing RAG numbers as unverified history.

The current trustworthy claim is a **recorded search-only trade-off on a
single-positive known-item task**, pending raw-data reproduction. Neither ANN
overlap nor citation-number validity establishes answer quality.
