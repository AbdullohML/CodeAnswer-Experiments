# D2 reproducible RAG behavior evaluation

## Purpose and evidence

D2 makes a **new** small CodeAnswer run inspectable. The historical
`0.964286` behavior accuracy, `0.875000` out-of-domain abstention,
`0.619048` fallback rate, and latency aggregates are still unverified:
their raw questions, labels, source lists, and generated answers were not
available. This run does not reproduce those values or their denominator.

The fixed [labels](../results/evaluation/d2/labels.jsonl) and
[manifest](../results/evaluation/d2/manifest.json) define 60 queries:

| Split | StackSample titles labeled answer | Handwritten nonprogramming controls labeled abstain | Used for |
| --- | ---: | ---: | --- |
| Development | 8 | 4 | Inspect behavior; never choose threshold |
| Calibration | 10 | 10 | Choose threshold from top reranker scores only |
| Test | 20 | 8 | Held-out behavior and latency report |

In-domain questions are the first fixed rows from the disjoint C2 frozen
StackSample splits. They reuse the indexed question titles, so this is an
easy known-item setup. The out-of-domain controls are plainly unrelated to
programming. The labels describe desired **answer versus abstain behavior**;
they do not judge factual correctness, whether the retrieved source supports
an answer, or whether an in-domain title is truly answerable from the
retrieved evidence. Near-domain unanswerable cases and human judgments are
needed later.

## Reproduce

With the checked-out artifacts and local `qwen2.5:3b` Ollama model:

```bash
PYTHONPATH=src python scripts/prepare_rag_evaluation.py
HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/evaluate_rag.py --device cpu
```

The preparation script fixes query IDs, split labels, C2 file hashes, and
the labels hash. The runner verifies these inputs. It uses the saved B1 PCA
basis, takes its first 128 components, searches the B2 HNSW-128 index at
`efSearch=128`, reranks 40 candidates, gives three sources to Qwen, and
uses the existing citation gate. This is a **new B1/B2 artifact path**;
the old optimized PCA/index files and old RAG examples are unavailable.
The [run manifest](../results/evaluation/d2/run_local_b1b2/run_manifest.json)
records artifact hashes, model/settings, and hashes of the saved traces.
Qwen ran locally with temperature 0, seed 20260714, context length 4096,
and 320 output tokens requested.

Calibration saved [per-query scores and sources](../results/evaluation/d2/run_local_b1b2/calibration.jsonl).
The F1-maximizing threshold was **2.803920269** using those 20 calibration
labels only; calibration F1 was 1.0 on the easy controls. The selected
threshold was fixed before the test run. Test labels did not select it.

Development and test JSONL files include every query label, actual
answer/abstain/error, top score, full source records, prompt text when
generation was attempted, raw generation, final answer, raw/final citation
metrics, fallback reasons, token counts, and measured stage times. Errors
are explicit records and count as incorrect behavior. Abstentions have no
raw generation or citation metrics because no generation occurs.

## Observed results

The [summary](../results/evaluation/d2/run_local_b1b2/summary.json)
is recomputable from [development](../results/evaluation/d2/run_local_b1b2/development.jsonl)
and [test](../results/evaluation/d2/run_local_b1b2/test.jsonl) records.

| Metric | Development | Held-out test |
| --- | ---: | ---: |
| Queries | 12 | 28 |
| Behavior accuracy (correct / all) | 12/12 = 1.000 | 28/28 = 1.000 |
| In-domain answer rate | 8/8 = 1.000 | 20/20 = 1.000 |
| Out-of-domain abstention rate | 4/4 = 1.000 | 8/8 = 1.000 |
| Extractive fallback among returned answers | 6/8 = 0.750 | 14/20 = 0.700 |
| Errors | 0/12 | 0/28 |
| Total latency p50 / p95 | 3.445 / 6.901 s | 3.657 / 6.036 s |

On test, 12 answers had invalid or absent numeric citations in the raw
generation, and 14 had long-line coverage below 0.5; those trigger counts
overlap. All 14 fallback answers had valid final citation syntax under the
current gate. This does **not** establish source support or correctness.
Test top-score ranges were 6.163–10.413 for the selected StackSample
titles and -11.433–-3.245 for the far-domain controls. That wide separation
explains the perfect behavior score on this small protocol.

Test total latency uses all 28 completed calls, including abstentions:
p50 3,657.107 ms and p95 6,035.553 ms. The per-stage summary stores its own
sample count: gate timing exists only for the 20 generated answers;
generation is recorded as zero for the eight threshold abstentions.
The test p50/p95 for corpus fetch were 561.027/623.094 ms and for the
rerank phase 1,498.804/1,753.516 ms. These timings are sequential local
CPU measurements, not a load test. The historical latency run is not
directly comparable.

## Metric definitions and limits

- Behavior accuracy = correct answer/abstain decisions divided by all
  labeled queries; errors are wrong.
- In-domain answer rate = answered StackSample titles divided by all
  StackSample titles in the split.
- Out-of-domain abstention rate = abstained nonprogramming controls divided
  by all such controls in the split.
- Fallback rate = returned answers using extractive fallback divided by all
  returned answers; abstentions and errors are excluded.
- Total latency percentiles are calculated from completed per-query calls,
  with errors excluded. Stage percentiles use only records where that stage
  was measured and cannot be added to form a total percentile.

The score threshold is a raw cross-encoder logit, not a calibrated
probability. The reranker input includes the indexed title and body, which
can make title-to-own-body examples easy. This protocol measures whether
the configured system follows simple domain behavior. D4 was skipped, so
answer correctness and claim-to-source support remain unjudged.
