# A4 — Cross-encoder reranking ablation

This experiment reorders the saved A3 top-40 candidate lists for BM25, exact
Flat-384 dense retrieval, and depth-40 RRF hybrid retrieval. The test set has
1,200 frozen C2 title queries, one judged positive document per query, and the
same 520,000-document corpus for all strategies. The A4 runner checks the
frozen manifest, corpus identity, A3 ranking checksums, query IDs, labels, and
baseline metrics before scoring. It never regenerates retrieval candidates.

## Reproduce

The configured model is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Download only
its configuration, weights, and tokenizer into the ignored
`artifacts/models/ms-marco-MiniLM-L-6-v2` directory. Its exact Hub revision and
file SHA-256 checksums are saved in the [result manifest](../results/evaluation/a4_test/results.json).

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/ms-marco-MiniLM-L-6-v2', local_dir='artifacts/models/ms-marco-MiniLM-L-6-v2', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json','sentence_bert_config.json'])"
PYTHONPATH=src python scripts/evaluate_reranking.py --split test --device cuda --output-dir results/evaluation/a4_test
```

The fixed [settings](../configs/reranking_ablation.json) use a batch size of
32, 384-token maximum input, and the pipeline's `0.03 * log1p(max(answer_score,
0))` vote bonus. The recorded seed identifies the existing C2 split; A4 does
no random sampling. The scorer takes the same query title for all variants.
For each query and input mode, it scores the union of candidates across the three
strategies once. Each strategy then uses its own saved top-40 subset. The
`title_body` mode passes `title + "\n" + question_body`, matching the runtime
reranker; `body_only` passes just `question_body`. Both vote settings use the
same raw cross-encoder scores, so the bonus comparison isolates sorting.
Ties retain candidate order. The ignored progress file allows a run with the
same model, device, settings, and A3 inputs to resume without rescoring.

## Results

The [results table](../results/evaluation/a4_test/results_table.csv) reports
candidate Recall@40, before and after Recall@3/10, MRR@10, and nDCG@10 for
all 12 strategy/input/bonus combinations. It also reports paired gains and
losses at Recall@10, rank improvements and regressions, and scoring p50/p95.
The [per-query output](../results/evaluation/a4_test/per_query.jsonl.gz) records
IDs, labels, before and after ranks, all 40 reranked IDs, raw cross-encoder
scores, and vote bonuses. All result file checksums are in the manifest.
Each per-query timing is the same shared-union value repeated on its three
strategy rows and both vote rows for that input mode.

The main paired results below use **vote bonus off**. Candidate Recall@40 is
the fixed ceiling from each retrieval strategy. The full CSV also contains
Recall@3, nDCG@10, and all vote-on rows.

| Generator | Candidate Recall@40 | Before Recall@10 | Title + body Recall@10 | Body-only Recall@10 | Before MRR@10 | Title + body MRR@10 | Body-only MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.726667 | 0.627500 | 0.726667 | 0.677500 | 0.498723 | 0.723160 | 0.574118 |
| Dense | 0.828333 | 0.743333 | 0.828333 | 0.757500 | 0.587885 | 0.822687 | 0.619740 |
| Hybrid | 0.849167 | 0.772500 | 0.848333 | 0.751667 | 0.609828 | 0.843565 | 0.611172 |

At Recall@10, title-plus-body reranking gained 119/102/91 queries and lost
none for BM25/dense/hybrid respectively. Body-only gained 90/68/47 and lost
30/51/72. The title-plus-body result is close to the candidate Recall@40
ceiling, which is expected when the exact query title is also present in its
judged document. On hybrid, body-only reranking reduced Recall@10 from
0.772500 to 0.751667. The vote bonus made no difference to the listed
title-plus-body target metrics. With body-only, it changed BM25 MRR@10 from
0.574118 to 0.575254, dense Recall@10 from 0.757500 to 0.756667, and
hybrid MRR@10 from 0.611172 to 0.612193. These are small paired effects on
this test, not evidence about answer quality.

The latency is the per-query scoring of the **shared union** of three top-40
sets for one input mode. It excludes retrieval, corpus loading, model loading,
and output serialization. The table repeats this shared cost on each strategy
row; it is not a separate per-strategy latency. Vote on/off reuses inference.
On the full GPU run, title-plus-body scoring took p50/p95 290.894/321.754 ms
per query; body-only took 285.594/320.485 ms. The two modes together spent
667.730 seconds in cross-encoder scoring. Peak PyTorch VRAM allocation was
313,358,336 bytes (298.84 MiB), peak reservation was 400,556,032 bytes
(382.00 MiB), and maximum process RSS was 2,054,976 KiB (1.96 GiB).
These are single-run measurements on the RTX 3050 with batch size 32; model
loading time is outside the timed region, and PyTorch VRAM accounting omits
driver and other non-PyTorch allocations. The 10-query CPU validation in the
ignored artifacts directory was a smoke run, not a benchmark.

## Interpretation limits

This is a known-item test: each query is the indexed question's own title and
its body is the single judged positive. The title also appears in the
`title_body` cross-encoder input, giving that mode a direct title match.
`body_only` reduces this particular shortcut but does not turn the test into
new-user-question evaluation. Unjudged retrieved questions may still be
useful. No metric here measures answer generation, citation support, or
real-user relevance.
