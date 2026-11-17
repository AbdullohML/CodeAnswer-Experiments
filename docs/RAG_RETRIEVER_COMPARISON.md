# D3 paired dense, BM25, and hybrid RAG

## Question and fixed protocol

Does changing the first 40 retrieved candidates change final RAG behavior,
citation formatting, fallback use, or latency on the **same** D2 questions?
The answer-quality effect remains unknown because D2 has no factual
correctness or claim-to-source support labels.

The [runner](../scripts/compare_rag_retrievers.py) used all 28 held-out
[D2 test labels](../results/evaluation/d2/labels.jsonl): 20 StackSample
question titles labeled *answer* and eight plainly nonprogramming controls
labeled *abstain*. Each strategy received the same query, in the same run.
Only candidate retrieval changed:

| Strategy | Candidate selection before shared reranker |
| --- | --- |
| Dense | B1 PCA-128 query projection, B2 HNSW-128 top 40 |
| BM25 | Existing body-only SQLite FTS5 BM25 top 40 |
| Hybrid | Top 40 from each of the above, RRF with constant 60, output top 40 |

Every query had 40 candidates in every strategy. The cross-encoder model,
title-plus-body pair input, answer-score bonus, top-three sources, raw-score
threshold **2.803920269**, Qwen2.5:3B prompt and generation settings,
citation gate, and fallback were shared. The threshold came solely from
D2's calibration split; it was never selected on D3 test outcomes. The
runner rotates which strategy runs first for each query. The
[manifest](../results/evaluation/d3/run_local_b1b2/run_manifest.json)
records labels, artifact hashes, generator settings and output hashes.
This hybrid uses HNSW-128 as its dense branch; A2's retrieval experiment
used exact Flat-384, so its retrieval scores are not directly comparable.

## Paired held-out results

The [summary](../results/evaluation/d3/run_local_b1b2/summary.json) is
computed from three per-query JSONL files and a
[paired table](../results/evaluation/d3/run_local_b1b2/paired_per_query.csv).

| Measure | Dense | BM25 | Hybrid |
| --- | ---: | ---: | ---: |
| Answer/abstain decisions matching labels | 28/28 | 27/28 | 28/28 |
| In-domain titles answered | 20/20 | 19/20 | 20/20 |
| Nonprogramming controls abstained | 8/8 | 8/8 | 8/8 |
| Extractive fallback / returned answers | 13/20 | 14/19 | 12/20 |
| Generator errors | 0 | 0 | 0 |
| Pre-generation errors | 0 | 0 | 0 |
| Mean raw citation-number validity on answers | 0.450 | 0.316 | 0.540 |
| Mean final citation-number validity on answers | 1.000 | 1.000 | 1.000 |
| Total response p50 / p95 (all 28) | 3.599 / 5.083 s | 4.098 / 6.159 s | 3.988 / 6.695 s |
| Candidate retrieval p50 / p95 (all 28) | 10.1 / 23.7 ms | 384.9 / 709.8 ms | 403.4 / 732.1 ms |
| Qwen output tokens, total | 1,735 | 2,190 | 2,010 |

Fallback counts are **not** generator failures. The citation gate triggered
fallback when raw numeric validity was below 1.0 or raw line coverage below
0.5; reasons may overlap. Final citation-number validity was 1.0 under all
three strategies, but it checks number format and does not establish source
support. The hybrid final coverage mean was 0.9875, so even format coverage
was not perfect for every returned answer. Token totals have different
answer denominators because BM25 abstained once more.

BM25's only different behavior was query `so-29695630`, “How to
concatenate vector of integers interleaving elements with a character?”.
Its top reranker score was 1.078, below the fixed threshold. Dense and
hybrid scored 9.577 and answered. Dense found the indexed same-title source
in its top three; BM25's top three were different. This is a useful
mechanism example, but a title-to-own-body case can favor the dense branch
and does not prove an answer is correct.

The top-three source IDs matched dense for only 3/28 BM25 queries and
6/28 hybrid queries. Each of BM25 and hybrid produced a different final
answer string from dense on 10/28 queries. Those text differences need
human quality judgments in a separate review. The fallback difference between dense and
hybrid is one answer on this small set, with no claim of a general gain.

## Runtime repeat and interpretation

The first four programming titles and last two nonprogramming controls were
run a second time under all three strategies. The
[repeat traces](../results/evaluation/d3/run_local_b1b2/paired_repeat.csv)
show identical candidate lists and prompts in all six pairs per strategy.
Final answers and fallback decisions matched in all six pairs. Raw Qwen
texts still changed in 2/6 dense pairs, 2/6 BM25 pairs, and 1/6 hybrid
pairs despite temperature 0 and a fixed seed. This is one local repeat,
not a guarantee of deterministic generation.

On those same six queries, total-latency p50 moved from 3.391 to 3.233 s
for dense, 4.191 to 4.387 s for BM25, and 3.880 to 3.780 s for hybrid.
For example, the first dense query took 8.505 s in the main pass and
4.275 s in the repeat, consistent with startup or runtime variation.
These sequential CPU timings are descriptive and do not establish stable
throughput or a statistically significant latency ordering. Stage p50/p95
values have their own denominators and cannot be added to form a total
percentile.

## Reproduce and next step

With the local artifacts and Ollama model available:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/compare_rag_retrievers.py --device cpu
```

The runner verifies D2 labels, calibration threshold, model and artifact
settings before executing. It saves raw and final answers, sources, exact
prompts, candidate IDs, fallback reasons, token counts, stage timings,
model errors, and retrieval errors per query. A later quality review would
need to blind the strategy names and score correctness, source support,
usefulness and abstention on these saved answers. D4 was skipped, so
candidate changes cannot be judged as answer-quality improvements here.
