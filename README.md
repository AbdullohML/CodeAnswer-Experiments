# CodeAnswer

CodeAnswer is a semantic search and cited-answer RAG system for
programming questions.

It searches 520,000 answered Stack Overflow questions, reranks the most
relevant candidates, and attempts to produce a concise answer using the
original human-written sources. Citation formatting checks do not verify
whether a source supports the answer.

The repository contains reusable Python modules and command-line scripts.
Notebooks, datasets, embeddings, indexes, and model files are not tracked
in Git.

## Architecture

The standard `scripts/ask_codeanswer.py` command uses this configured dense
path. Here **dense** means vector search; HNSW is its approximate search
index, not a second retrieval stage. The 40-candidate and three-source limits
come from [`configs/default.yaml`](configs/default.yaml).

```text
User question
    ↓
all-MiniLM-L6-v2 bi-encoder
    ↓
PCA projection: 384 → 128 dimensions
    ↓
FAISS HNSW dense search: up to 40 candidates
    ↓
ms-marco-MiniLM-L-6-v2 cross-encoder reranks candidates
    ↓
Raw reranker score of top-ranked source compared with threshold
    ├── score below threshold → abstain
    └── score at or above threshold
            ↓
        Top 3 Stack Overflow sources
            ↓
        Qwen2.5:3B through Ollama
            ↓
        numeric citation-format gate
            ↓
        raw answer or extractive fallback
```

The top source is ranked by cross-encoder score plus a small answer-vote bonus;
its **raw** cross-encoder score is compared with the configured threshold.
That score is not a probability or a check that the evidence supports the
answer. The citation gate checks numeric citation
format and line coverage, not factual correctness. The separate D3 experiment
replaced candidate selection with BM25 top 40 or a hybrid: HNSW top 40 plus
BM25 top 40, combined by RRF into 40 before the same reranker. The standard
answer command does not select those alternatives. See the
[architecture and methodology notes](docs/METHODOLOGY.md) for artifact lineage,
query definitions and benchmark boundaries.

## Recorded iterations

### 1. Exact dense-search baseline

Normalized 384-dimensional MiniLM embeddings are stored in
`FAISS IndexFlatIP`.

### 2. PCA and HNSW optimization

PCA reduces the vectors to 128 dimensions. HNSW provides approximate
nearest-neighbor search with a controllable latency-quality trade-off.

### 3. Reranked cited-answer RAG

The optimized retriever returns 40 candidates, a cross-encoder reranks
them, and local Qwen generates an answer from the best three sources.
The system checks citation-number syntax and abstains when its top reranker
score is below a configured threshold; that score is not a support judgment.

## Recorded results

The table below contains historical experiment aggregates. See the
[C1 evaluation audit](docs/EVALUATION_AUDIT.md) for their definitions,
reproducibility limits, and leakage risks.

The completed [C2 frozen Flat-384 test](docs/EVALUATION_PROTOCOL.md) used
1,200 held-out query IDs and recorded Recall@10 **0.743333**, MRR@10
**0.587885**, nDCG@10 **0.625245**, and search p95 **88.379 ms**.
Its query selection differs from the historical experiments, so these scores
are not paired with the table below.

The [A1 BM25 baseline](docs/BM25_BASELINE.md) uses the same frozen C2 test
queries and labels. It recorded Recall@10 **0.627500**, Recall@40 **0.726667**,
MRR@10 **0.498723**, and nDCG@10 **0.529569**. Its index and search settings,
per-query rankings, build time, size, and latency are saved with the results.

The [A2 hybrid RRF experiment](docs/HYBRID_RRF.md) combined dense and BM25
top-40 candidates on those same test queries. It recorded Recall@10
**0.772500**, Recall@40 **0.849167**, MRR@10 **0.609828**, and nDCG@10
**0.648677**; the documented search p95 was **648.974 ms**.

The [A3 paired comparison](docs/FAIR_RETRIEVAL_COMPARISON.md) reran all three
top-40 rankings on the same 1,200 test queries and measured raw-title
single-query latency on a shared 100-query prefix. It includes a fixed
29-query exact-term slice and per-query wins and losses.

The [E2 verified results summary](docs/RESULTS_SUMMARY.md) brings the
completed retrieval, indexing, resource and RAG measurements into one place
with denominators, timing boundaries, source files and uncertainty.

### Historical retrieval aggregate (search-only timing)

| System | Recall@10 | Search p95 | Index size |
|---|---:|---:|---:|
| Flat-384 | 0.7070 | 71.059 ms | 761.72 MiB |
| Flat-PCA128 | 0.6655 | 11.170 ms | 253.91 MiB |
| HNSW-PCA128, ef=128 | 0.6555 | 0.514 ms | 325.45 MiB |

In that historical run, HNSW had approximately 138 times lower search p95
than the exact baseline and used approximately 57% less index storage. This
does not describe whole-pipeline latency or the frozen C2/B1/B2 tests.

### Historical RAG aggregate (unverified per-example evidence)

| Metric | Result |
|---|---:|
| Behavior accuracy (answer/abstain) | 96.43% |
| In-domain answer rate | 100% |
| Out-of-domain abstention | 87.5% |
| Citation coverage | 100% |
| Citation validity | 97.28% |
| Total latency p50 | 2.94 s |
| Total latency p95 | 5.48 s |

Detailed results are available in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
The RAG table above comes from a historical aggregate without saved labels,
raw answers or per-query traces. It is not a result of the newer D2/D3
protocols. Their separately measured behavior and latency are documented in
[D2](docs/RAG_EVALUATION.md) and [D3](docs/RAG_RETRIEVER_COMPARISON.md).

## Repository structure

Selected paths:

```text
CodeAnswer/
├── configs/
│   └── default.yaml
├── docs/
│   ├── METHODOLOGY.md
│   ├── RESULTS_SUMMARY.md
│   ├── EXPERIMENTS.md
│   └── RAG_RETRIEVER_COMPARISON.md
├── results/
│   ├── experiment_summary.json
│   └── evaluation/
├── scripts/
│   ├── ask_codeanswer.py
│   ├── build_corpus.py
│   ├── prepare_evaluation.py
│   ├── evaluate_rag.py
│   ├── compare_rag_retrievers.py
│   └── verify_results_summary.py
├── src/codeanswer/
│   ├── data.py
│   ├── rag.py
│   ├── rag_retrievers.py
│   └── reranking.py
├── tests/
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.11–3.13
- Approximately 15 GiB RAM for the validated laptop setup
- Optional NVIDIA GPU for faster embedding and reranking
- Ollama with `qwen2.5:3b` for answer generation

The FAISS indexes use CPU search. MiniLM inference can use CPU or CUDA.

## Installation

```bash
git clone git@github.com:AbdullohML/CodeAnswer.git
cd CodeAnswer
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
```

## Data preparation

Download StackSample and locate:

```text
Questions.csv
Answers.csv
```

Build the 520,000-object corpus:

```bash
PYTHONPATH=src python scripts/build_corpus.py \
  --source-dir /path/to/stacksample
```

Each question is joined to its highest-scored answer. HTML is removed and
the resulting collection is saved as Parquet.

## Generate embeddings

```bash
PYTHONPATH=src python scripts/generate_embeddings.py \
  --device cuda
```

The pipeline writes normalized vectors incrementally to a resumable
memory-mapped float32 file.

## Build the exact baseline

```bash
PYTHONPATH=src python scripts/build_flat_index.py
```

Evaluate it:

```bash
PYTHONPATH=src python scripts/evaluate_retrieval.py \
  --device cuda
```

This command uses the historical dynamic query selection. For the completed
frozen C2 protocol, exact run command, and result files, see
[`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md).

## Build the optimized retriever

```bash
PYTHONPATH=src python scripts/build_optimized_index.py
```

Compare all retrieval configurations:

```bash
PYTHONPATH=src python scripts/compare_retrievers.py \
  --device cuda
```

## Ask CodeAnswer

Start Ollama:

```bash
ollama serve
```

Install the local generator:

```bash
ollama pull qwen2.5:3b
```

Ask a programming question:

```bash
PYTHONPATH=src python scripts/ask_codeanswer.py \
  "Why does an asynchronous JavaScript loop return the wrong result?" \
  --device cuda
```

This command loads the original `artifacts/optimized/` PCA/HNSW files and
the threshold `0.449093` from the default config. Those original optimized
files are not tracked here. D2/D3 instead used the saved B1 PCA basis,
B2 HNSW index and a threshold of `2.803920269` selected from D2 calibration
labels; their results do not validate the command above or its threshold.

Return full JSON containing scores, latency, and sources:

```bash
PYTHONPATH=src python scripts/ask_codeanswer.py \
  "How do I split a string in SQL?" \
  --device cuda \
  --json
```

## Testing

```bash
PYTHONPATH=src python -m pytest -q
```

The test suite covers corpus construction, resumable embeddings, FAISS
indexing, retrieval metrics, PCA, HNSW, reranking, threshold calibration,
citation-format checks and extractive fallback behavior.

## Generated artifacts

The following are intentionally excluded from Git:

```text
data/
artifacts/
models/
*.parquet
*.f32
*.faiss
*.index
```

## Limitations

- The source collection is historical and may contain outdated answers.
- Retrieval evaluation uses a known-item task rather than live user labels.
- The answerability and RAG evaluation sets are relatively small.
- The historical faithfulness value is an embedding-based proxy, not a
  source-support judgment.
- Search-only timing omits corpus fetch, reranking and generation. B4 measured
  corpus fetch separately; D2/D3 measured complete calls on small query sets.
- The historical 61.9% fallback aggregate has no saved raw answers or trigger
  labels; the current syntax gate can trigger an extractive fallback.

## Project Improvement Roadmap

This roadmap extends the existing exact MiniLM/FAISS baseline, PCA-128 and
HNSW comparison, cross-encoder, and Ollama RAG pipeline. The committed
historical retrieval results use 2,000 title-to-body **known-item** queries
from the indexed corpus; the frozen protocol uses 400 development, 400
calibration, and 1,200 test queries from that same corpus. Neither measures
relevance to new user questions. The committed RAG aggregates lack their
labeled examples and runnable evaluation
script. Tasks below have tracked statuses; existing components are starting
points, not evidence that these new experiments are complete. The completed
paired error analysis was renumbered from C6 to C3 after the unused C3–C5
dataset branch was removed. D4 and D5 are outside the active roadmap.

Use the same frozen query IDs, corpus and relevance judgments for paired
comparisons. Save exact commands, seeds, artifact identities, model versions,
hardware, and per-query outputs alongside aggregates. Use the existing
embeddings and indexes where compatible. The laptop budget is 16 GiB RAM and
an RTX 3050 with 4 GiB VRAM; start with 200-query smoke runs, batch model
inference, build one extra index at a time, and run Qwen2.5:3B via local
Ollama. Do not call a smoke result a full benchmark.

### Recommended execution order and progress

This is one dependency-respecting order for active tasks. The **Independent after** column
marks work that can proceed once the listed gate is done, without waiting for
the preceding row. The remaining evaluation work uses the existing StackSample
corpus and C2 frozen splits; no second dataset is planned.

| Order | Task | Independent after | Status |
|---:|---|---|---|
| 1 | C1 | Start | DONE |
| 2 | C2 | C1 | DONE |
| 3 | A1 | C2 | DONE |
| 4 | A2 | A1 | DONE |
| 5 | A3 | A2 | DONE |
| 6 | A4 | A3 | DONE |
| 7 | B1 | C2; can run alongside A tasks | DONE |
| 8 | B2 | C2; can run alongside B1 | DONE |
| 9 | B3 | C2; can run alongside B1/B2 | DONE |
| 10 | B4 | B1, B2, B3 | DONE |
| 11 | C3 | A3 | DONE |
| 12 | D1 | C1; can run alongside retrieval work | DONE |
| 13 | D2 | C2 and D1 | DONE |
| 14 | D3 | A4 and D2 | DONE |
| 15 | E1 | A3, B4, D3 | DONE |
| 16 | E2 | C3, D3, E1 | DONE |

### A — Retrieval improvements

#### A1 — BM25 baseline

- **Priority:** P0. **Status:** DONE. **Estimated effort:** Medium.
- **Goal:** Test lexical matching of identifiers, error strings, and versions.
- **Implementation:** Add a reproducible BM25 index/search path over the same corpus IDs; document tokenization and build limits.
- **Experiments:** Smoke run, then rank the C2 frozen queries at 10 and 40; inspect exact-term cases.
- **Metrics:** Recall@10/40, MRR@10, nDCG@10, build time, index size, search p50/p95.
- **Deliverables:** BM25 module/CLI, tests, config, per-query rankings, results table.
- **Dependencies:** C2.
- **Acceptance criteria:** Deterministic IDs and rankings; all metrics use C2 judgments; results and settings are saved.

#### A2 — Hybrid retrieval with RRF

- **Priority:** P0. **Status:** DONE. **Estimated effort:** Small.
- **Goal:** Recover relevant candidates missed by either lexical or dense search.
- **Implementation:** Fuse A1 BM25 and existing dense top-40 lists by reciprocal rank fusion; deduplicate by `doc_id` and expose the fusion constant.
- **Experiments:** Compare at least two candidate depths and one fixed RRF setting on C2 queries.
- **Metrics:** Recall@10/40, MRR@10, nDCG@10, union contribution, search p50/p95.
- **Deliverables:** Fusion code, tests for ties/duplicates, rankings and config.
- **Dependencies:** A1, C2.
- **Acceptance criteria:** Deterministic fusion; both sources contribute; paired outputs share query IDs and relevance labels.

#### A3 — Fair retrieval comparison

- **Priority:** P0. **Status:** DONE. **Estimated effort:** Small.
- **Goal:** Establish a defensible BM25, dense, and hybrid comparison.
- **Implementation:** Extend the comparison runner to use C2 queries, common cutoffs, identical corpus, saved configurations and per-query rankings.
- **Experiments:** Run all three methods on the full frozen set; repeat on a small exact-term slice.
- **Metrics:** Recall@1/5/10/40, MRR@10, nDCG@10, p50/p95 and query throughput.
- **Deliverables:** Reproducible command, paired CSV/JSON, summary table and limitations.
- **Dependencies:** A2, C2.
- **Acceptance criteria:** Each reported difference uses the same queries and labels; no retrieved-score scale is compared across methods.

#### A4 — Cross-encoder reranking ablation

- **Priority:** P0. **Status:** DONE. **Estimated effort:** Medium.
- **Goal:** Measure the reranker's added value for each candidate generator.
- **Implementation:** Reuse the existing cross-encoder on equal top-40 candidate budgets for BM25, dense and hybrid; separate cross-encoder score from the answer-vote bonus.
- **Experiments:** Paired before/after reranking for all three, plus vote-bonus on/off and title-plus-body versus body-only input; batch on GPU or CPU.
- **Metrics:** Candidate Recall@40, final Recall@3/10, MRR@10, nDCG@10, rerank p50/p95, VRAM.
- **Deliverables:** Ablation runner, tests, per-query ranks and results table.
- **Dependencies:** A3.
- **Acceptance criteria:** Same candidate and scoring rules across strategies; report gains, losses and cost; do not generalize title-matched known-item results to new user queries.

The completed [A4 ablation](docs/RERANKING_ABLATION.md) reports the full
1,200-query paired test, vote and input-mode variants, query-level rankings,
and measured GPU cost.

### B — Indexing and optimization

#### B1 — PCA dimension sweep

- **Priority:** P1. **Status:** DONE. **Estimated effort:** Medium.
- **Goal:** Test whether the current 128 dimensions are a good trade-off.
- **Implementation:** Reuse 384-dimensional embeddings; compare no PCA (384) with PCA 256, 128 and 64, each with exact IP search and normalized vectors.
- **Experiments:** Fit on a fixed documented sample, then evaluate C2 queries at each dimension; use one extra artifact at a time.
- **Metrics:** Explained variance, Recall@10/40, MRR@10, index bytes, transform and search p95.
- **Deliverables:** Dimension-specific manifests, comparison table and plot.
- **Dependencies:** C2.
- **Acceptance criteria:** All dimensions use the same queries; compare retrieval loss separately from ANN loss and record fit sample IDs.

The completed [B1 sweep](docs/PCA_DIMENSION_SWEEP.md) records the shared PCA
fit sample, exact per-dimension rankings, measured trade-off, and plot.

#### B2 — HNSW versus exact search

- **Priority:** P1. **Status:** DONE. **Estimated effort:** Medium.
- **Goal:** Isolate approximate-search loss from PCA loss.
- **Implementation:** Compare HNSW and IndexFlatIP in the same dimension, at least 128; add 384 only if memory permits.
- **Experiments:** Search identical C2 query vectors and corpus vectors with matched top-k.
- **Metrics:** Known-item Recall@10/40, top-k ANN overlap, p50/p95, index bytes.
- **Deliverables:** Same-space paired rankings and quality/latency table.
- **Dependencies:** C2.
- **Acceptance criteria:** Exact search is the ANN reference in each dimensional space; distinguish ANN overlap from relevance recall.

The completed [B2 comparison](docs/HNSW_EXACT_COMPARISON.md) reports paired
exact/HNSW rankings in 128 and 384 dimensions, overlap, recall, costs, and
the separate PCA and ANN losses.

#### B3 — efSearch sweep

- **Priority:** P1. **Status:** DONE. **Estimated effort:** Small.
- **Goal:** Reassess the existing 16/32/64/128 sweep on the frozen evaluation set.
- **Implementation:** Extend `compare_retrievers.py` to save per-query rankings and parameterized efSearch values for the selected HNSW index.
- **Experiments:** Repeat the four existing settings at top-10 and top-40; add a higher value only if the curve has not saturated.
- **Metrics:** Recall@10/40, same-space ANN overlap, QPS, p50/p95.
- **Deliverables:** Reproducible sweep table and quality/latency curve.
- **Dependencies:** C2.
- **Acceptance criteria:** State the chosen efSearch and its measured trade-off; retain the historical sweep as historical evidence.

The completed [B3 sweep](docs/EFSEARCH_SWEEP.md) records the full frozen
query rankings, ANN overlap and recall, latency/QPS curve, and the current
`efSearch=128` choice.

#### B4 — Build and runtime resource costs

- **Priority:** P1. **Status:** DONE. **Estimated effort:** Medium.
- **Goal:** Measure the cost omitted by the current search-only timing and index-size reports.
- **Implementation:** Instrument PCA fit/transform, index build, process peak RSS, GPU VRAM, load time, and query stages, including corpus fetch.
- **Experiments:** Run on the available full artifacts where feasible, plus a named small subset; repeat latency after warm-up.
- **Metrics:** Build seconds, peak RSS/VRAM, file size, load seconds, search and fetch p50/p95.
- **Deliverables:** Hardware/config manifest and resource table for exact and selected HNSW setups.
- **Dependencies:** B1, B2, B3.
- **Acceptance criteria:** Specify cold/warm and full/subset results; do not equate index bytes with total memory.

The completed [B4 profile](docs/RESOURCE_COSTS.md) measures full and 20k-subset
PCA/build costs, full-index load and warmed query stages, process peak RSS,
and the unavailable GPU VRAM boundary. It exposes corpus fetch time omitted
by the earlier search-only results.

### C — Dataset and evaluation

#### C1 — Audit evaluation and leakage

- **Priority:** P0. **Status:** DONE. **Estimated effort:** Small.
- **Goal:** Establish what existing results actually prove and where leakage or label bias may occur.
- **Implementation:** Trace corpus sampling, title-to-body labels, PCA fit, query selection, threshold calibration and RAG aggregates; inventory missing raw inputs.
- **Experiments:** Recompute available aggregate formulas from committed data and run tiny adversarial metric fixtures; no full benchmark required.
- **Metrics:** Query/label counts, overlap across splits, duplicate rate, reproducibility gaps.
- **Deliverables:** [Evaluation audit](docs/EVALUATION_AUDIT.md) with evidence, corrected metric definitions and a leakage-prevention plan.
- **Dependencies:** None.
- **Acceptance criteria:** Explicitly distinguish known-item retrieval, ANN overlap, answer/abstain behavior and answer correctness; identify unverified claims.

#### C2 — Freeze evaluation inputs and provenance

- **Priority:** P0. **Status:** DONE. **Estimated effort:** Medium.
- **Goal:** Make future comparisons paired and prevent tuning on the final test set.
- **Implementation:** Save deterministic query IDs and labels; separate development/calibration/test IDs; check duplicate question families and corpus/artifact identity; reject empty sets and cutoffs beyond ranking depth; version the evaluation manifest.
- **Experiments:** Recreate the split twice and run the existing dense baseline on the frozen test set.
- **Metrics:** Split sizes, ID intersection, duplicate rate, baseline Recall@10, MRR@10, nDCG@10.
- **Deliverables:** [Frozen evaluation protocol](docs/EVALUATION_PROTOCOL.md), split manifests, validation tests, baseline per-query outputs and command.
- **Dependencies:** C1.
- **Acceptance criteria:** Zero unintended split overlap; identical reruns; document that title queries are still in-corpus known-item tests.

#### C3 — Error analysis and uncertainty

- **Priority:** P1. **Status:** DONE. **Estimated effort:** Small.
- **Goal:** Explain where measured retrieval gains and losses arise.
- **Implementation:** Join A3 per-query ranks on the C2 frozen StackSample test IDs, group by identifier/error/version and query length, and inspect representative wins/losses.
- **Experiments:** Paired bootstrap confidence intervals or query-level randomization on the fixed 1,200-query test set.
- **Metrics:** Per-query reciprocal-rank changes, subgroup Recall@10/40, confidence intervals.
- **Deliverables:** Error taxonomy, examples and uncertainty table.
- **Dependencies:** A3.
- **Acceptance criteria:** No significance claim without paired query-level evidence; discuss known-item label limits.

The completed [C3 analysis](docs/ERROR_ANALYSIS.md) reports paired bootstrap
intervals, full-test rank-change mechanisms, descriptive title slices, and
inspected wins/losses on the same frozen StackSample test set.

### D — End-to-end RAG

#### D1 — Audit extractive fallback and citation gate

- **Priority:** P0. **Status:** DONE. **Estimated effort:** Small.
- **Goal:** Explain the recorded 61.9% fallback rate and actual safety behavior.
- **Implementation:** Trace citation coverage, invalid-number handling, word limit, source truncation, empty evidence and Ollama failures; classify final versus raw generated output.
- **Experiments:** Add targeted fixture cases and manually inspect a small saved answer sample if available.
- **Metrics:** Trigger counts, raw/final citation coverage and validity, fallback and error rates.
- **Deliverables:** Behavior matrix, tests and prioritized fixes.
- **Dependencies:** C1.
- **Acceptance criteria:** Document that the current 50% line-coverage gate checks citation syntax, not claim support; no unsupported safety claim remains.

The completed [D1 audit](docs/FALLBACK_CITATION_AUDIT.md) records synthetic
raw/final gate behavior, source truncation, empty-evidence and generator
failures, and prioritized fixes. The historical fallback rate remains
unverified without its raw examples.

#### D2 — Reproducible RAG evaluation harness

- **Priority:** P0. **Status:** DONE. **Estimated effort:** Medium.
- **Goal:** Make the existing calibration and small RAG aggregates auditable.
- **Implementation:** Add labeled development/calibration/test fixtures and a runner that saves raw generation, final answer, abstention, sources and all stage timings; recalibrate only on calibration data.
- **Experiments:** Reproduce the available small protocol with local Qwen, or mark historical figures unverified if raw inputs cannot be recovered.
- **Metrics:** Behavior accuracy, in-domain answer rate, out-of-domain abstention, fallback rate and p50/p95 latency.
- **Deliverables:** Harness, tests, labels/manifest and per-query JSONL.
- **Dependencies:** C2, D1.
- **Acceptance criteria:** Test labels never select thresholds; each aggregate is traceable to individual runs and generator settings.

The completed [D2 evaluation](docs/RAG_EVALUATION.md) freezes 60
StackSample-title and handwritten out-of-domain behavior labels across
development, calibration and test. It selects the threshold on calibration
scores only, then saves local-Qwen prompts, raw/final answers, sources,
fallback reasons, errors and stage timings per query. On the 28-query test,
simple answer/abstain behavior was 28/28; 14/20 returned answers used
extractive fallback. These controls do not measure answer correctness or
source support, and the historical RAG aggregates remain unverified.

#### D3 — BM25, dense and hybrid RAG

- **Priority:** P1. **Status:** DONE. **Estimated effort:** Medium.
- **Goal:** Measure whether retrieval changes improve final answers.
- **Implementation:** Add a retriever adapter to the existing pipeline; hold top-40 candidates, reranker, three sources, prompt and Qwen2.5:3B settings fixed.
- **Experiments:** Run the three strategies on D2's small held-out set; repeat a subset for runtime stability.
- **Metrics:** Answer/abstain behavior, fallback rate, citation validity, total and stage p50/p95, token use.
- **Deliverables:** Paired per-query outputs and RAG comparison table.
- **Dependencies:** A4, D2.
- **Acceptance criteria:** Only the candidate strategy varies; report model failures and fallback separately from successful generation.

The [D3 paired RAG comparison](docs/RAG_RETRIEVER_COMPARISON.md) ran all
three top-40 candidate strategies on D2's same 28 held-out questions with
the threshold, reranker, top-three sources, Qwen settings and citation gate
fixed. Dense and hybrid matched 28/28 simple answer/abstain labels; BM25
matched 27/28. Extractive fallback occurred in 13/20, 14/19 and 12/20
returned answers respectively; no model or retrieval errors occurred.
Saved per-query outputs and a six-query repeat show changed sources and
some raw generation variation. Answer correctness and source support remain
unjudged.

Human answer-quality review and its dependent failure-analysis task were
removed from the active roadmap. D2/D3 measure behavior and citation
format only; their answers are not quality-rated.

### E — Documentation and reproducibility

#### E1 — Architecture and methodology update

- **Priority:** P1. **Status:** DONE. **Estimated effort:** Small.
- **Goal:** Make the current design and evaluation boundaries easy to explain.
- **Implementation:** Update the architecture diagram, corpus/query definitions, artifact lineage and benchmark protocols in README/docs.
- **Experiments:** Check every claim against code, manifests and finished task outputs.
- **Metrics:** Documented configuration coverage and count of unsupported claims corrected.
- **Deliverables:** Updated README and methodology notes.
- **Dependencies:** A3, B4, D3.
- **Acceptance criteria:** Descriptions match implementation and mark historical numbers clearly.

The completed [methodology notes](docs/METHODOLOGY.md) trace the corpus,
indexes, query labels, standard answer path and D2/D3 evaluation path. The
README architecture now distinguishes dense/HNSW from BM25/RRF, labels the
historical RAG figures as unaudited per-example aggregates, and states the
different thresholds and latency boundaries.

#### E2 — Experimental results summary

- **Priority:** P1. **Status:** DONE. **Estimated effort:** Small.
- **Goal:** Present quality, speed and resource trade-offs in one place.
- **Implementation:** Consolidate verified A/B/C/D1–D3 tables with artifact references, denominators and confidence intervals where applicable.
- **Experiments:** Recompute tables from saved outputs and verify cited configs.
- **Metrics:** Retrieval quality, answer/abstain behavior, citation format, p50/p95, memory, build time and uncertainty.
- **Deliverables:** Results summary with compact plots and reproducibility commands.
- **Dependencies:** C3, D3, E1.
- **Acceptance criteria:** Every headline number has a source file, dataset/split, method and measurement scope.

The completed [results summary](docs/RESULTS_SUMMARY.md) links each table
and plot to its saved outputs. Run
`PYTHONPATH=src python scripts/verify_results_summary.py` to recompute
headline retrieval, reranking, uncertainty and RAG values from committed
per-query records and check the cited configs.

## License

MIT
