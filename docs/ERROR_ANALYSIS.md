# C3 — Paired retrieval error analysis and uncertainty

**Status: DONE.** This analysis reuses the [A3 paired per-query rankings](../results/evaluation/a3_test_paired_per_query.csv)
for all 1,200 frozen C2 StackSample test titles. Each query has one judged
positive: the body of that same indexed question. No new corpus, index, model,
or relevance label was introduced. The [fixed settings](../configs/error_analysis.json),
[result manifest](../results/evaluation/c3/results.json),
[per-query deltas](../results/evaluation/c3/per_query.csv),
[subgroup table](../results/evaluation/c3/group_table.csv), and
[illustrative query IDs](../results/evaluation/c3/illustrative_examples.csv)
retain the evidence. The runner validates A3's hash, query order, labels,
per-query ranks, and aggregate metrics against the frozen manifest before
calculating anything.

## Paired uncertainty

The unit of resampling is a **query ID**. For each metric, the runner first
subtracts dense from hybrid on every query, then resamples the 1,200 paired
differences 10,000 times with replacement (seed `20260714`). The table gives
the observed mean difference and a 95% percentile bootstrap interval.

| Hybrid minus exact dense | Observed difference | 95% paired bootstrap interval |
|---|---:|---:|
| Recall@10 | +0.029167 | [+0.010833, +0.046667] |
| Recall@40 | +0.020833 | [+0.007500, +0.034167] |
| MRR@10 | +0.021943 | [+0.003959, +0.040049] |

For Recall@10, hybrid put the gold document inside the cutoff on **78**
queries where dense did not, and moved it outside on **43** where dense did.
At Recall@40, the counts were 48 gains and 23 losses. The intervals quantify
resampling variability on this fixed known-item query set. They do **not**
establish performance on new user questions or correct generated answers.

## Where the rank changes came from

| Observed paired top-10 pattern | Query count |
|---|---:|
| Hybrid gain; gold in BM25 top 40 | 78 of 78 gains |
| Hybrid gain; gold absent from dense top 40 | 26 of 78 gains |
| Hybrid loss; gold absent from BM25 top 40 | 43 of 43 losses |
| Gold absent from both source top-40 lists | 150 of 1,200 queries |

These counts show genuine source complementarity and a fusion risk. BM25 can
supply a gold document dense missed, but when BM25 does not contain it,
fusion can demote a dense hit. The final ranking also depends on other
candidates in each source list, so these counts alone do not prove a
particular term or score caused any individual change.

## Descriptive slices

The identifier, version, and error-word patterns are exactly A3's regex
definitions, now applied to **all 1,200 titles** rather than A3's selected
29-query diagnostic slice. Query length is a whitespace-token count with
fixed bins (1–5, 6–10, 11+). Regex groups may overlap; the length groups do
not. These were not used to choose the retrieval configuration.

| Slice | Queries | Dense R@10 | Hybrid R@10 | Hybrid − dense R@10 | Dense R@40 | Hybrid R@40 |
|---|---:|---:|---:|---:|---:|---:|
| All | 1,200 | 0.743333 | 0.772500 | +0.029167 | 0.828333 | 0.849167 |
| Underscore identifier | 25 | 0.840000 | 0.880000 | +0.040000 | 0.880000 | 0.920000 |
| Version number | 19 | 0.894737 | 0.894737 | 0.000000 | 0.894737 | 0.947368 |
| Error / exception / traceback | 53 | 0.679245 | 0.679245 | 0.000000 | 0.735849 | 0.811321 |
| Any of those patterns | 91 | 0.769231 | 0.780220 | +0.010989 | 0.813187 | 0.868132 |
| Other titles | 1,109 | 0.741208 | 0.771867 | +0.030658 | 0.829576 | 0.847610 |
| 1–5 tokens | 225 | 0.680000 | 0.702222 | +0.022222 | 0.786667 | 0.817778 |
| 6–10 tokens | 681 | 0.740088 | 0.765051 | +0.024963 | 0.826725 | 0.831131 |
| 11+ tokens | 294 | 0.799320 | 0.843537 | +0.044218 | 0.863946 | 0.914966 |

The small regex categories support diagnosis, not subgroup significance
claims. We did not adjust for multiple comparisons or construct subgroup
confidence intervals. The long-title gain may reflect this query set's
composition rather than a general rule about query length.

## Inspected examples

The [example CSV](../results/evaluation/c3/illustrative_examples.csv)
selects the six largest top-10 reciprocal-rank gains and losses, breaking
ties by frozen query order. Its query IDs join directly to the C3 per-query
table and A3 rankings. I inspected the selected titles and top candidates
using the local corpus and A1/A2 top-40 ID lists. The examples below are
deliberately illustrative; selection by outcome makes them unsuitable for
an unbiased subgroup estimate.

| Query ID | Topic summary | BM25 / dense / hybrid gold rank | Observation |
|---:|---|---|---|
| 32242670 | Postgres duplicate-column error | 1 / absent / 1 | BM25 retrieved the exact known item first; dense's top results were related SQL errors. |
| 19809550 | Updating a view without a model change | 1 / 11 / 1 | Lexical retrieval moved the gold document into first place after fusion. |
| 23560410 | PHPMailer authentication failure | absent / 1 / 16 | Dense found the gold document first; the hybrid list placed other mail-error posts above it. |
| 3524110 | Android input-method settings | absent / 2 / 13 | Dense ranked the gold second; fusion moved it below the top-10 cutoff. |

“Absent” means outside the top 40, not absent from the corpus. The saved
[A1 BM25 rankings](../results/evaluation/a1_bm25_test_per_query.csv),
[A2 dense rankings](../results/evaluation/a2_test_dense_top40.csv), and
[A2 hybrid rankings](../results/evaluation/a2_test_hybrid_depth40.csv)
provide the candidate IDs behind these inspections.

## Reproduce and limits

```bash
PYTHONPATH=src python scripts/analyze_retrieval_errors.py
```

This reads the frozen local corpus manifest and A3's committed rankings;
large source artifacts remain ignored. The query text is read for regex and
length analysis but is not copied into the result files. The only positive
label is each title's own body; useful alternative answers are unjudged.
The frozen test was also inspected in earlier tasks, so these descriptive
slices must not be used to retune the system and then claimed as fresh
held-out validation. No result here measures answer faithfulness or RAG
response quality.
