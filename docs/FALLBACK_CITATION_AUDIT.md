# D1 — Citation gate and extractive fallback audit

**Status: DONE.** This audits the existing [RAG implementation](../src/codeanswer/rag.py)
with deterministic synthetic fixtures. The [fixture result](../results/evaluation/d1/fixture_audit.json)
contains each raw generated text, raw citation metrics, trigger reasons,
final text and metrics, fallback flag, error type, source counts, source hash,
and command. Run `PYTHONPATH=src python scripts/audit_rag_gate.py` to recreate
it. No Ollama server or model is used by the fixtures. The audit preserves
the production gate; it does not recalibrate the threshold or change answer
behavior before D2 can evaluate such changes on labeled examples.

## What the code actually checks

`CodeAnswerPipeline.answer` retrieves and reranks up to 40 candidates, then
reads the top source's raw `reranker_score`. A score **below** the configured
threshold returns an abstention without calling Ollama. Otherwise,
`generate_grounded_answer` sends the three selected sources to Ollama and
scores the returned text with `citation_metrics`.

That helper treats each **nonempty line of at least 15 characters** as one
unit, although its variable name says “paragraph.” Coverage is the fraction
of those long lines containing at least one numeric marker like `[1]`.
Validity is the fraction of numeric markers whose number lies from 1 through
the number of supplied sources. With no qualifying long line, both metrics
are zero. The fallback runs if numeric validity is **below 1.0** or long-line
coverage is **below 0.5**. At exactly 0.5 coverage, the raw generation passes.
After fallback, the helper computes metrics on the **final answer** again,
so raw and final numbers may differ. Neither metric checks that a cited
source supports a claim, that an answer is correct, or that it is current.

## Synthetic behavior matrix

These 11 controlled fixtures exercise the current helper. They are a
diagnostic set chosen to hit branches, not a representative RAG evaluation.

| Probe | Raw gate / observed outcome | Final behavior |
|---|---|---|
| Valid `[1]` | Coverage 1, validity 1 | Raw text kept. |
| Out-of-range `[3]` with one source | Validity 0 | Extractive fallback, final coverage/validity 1/1. |
| No citation | Coverage 0, validity 0 | Extractive fallback, final 1/1 for a nonempty source answer. |
| Two long lines, one cited | Coverage exactly 0.5 | Raw text kept, including the uncited line. |
| Short uncited line plus long cited line | Short line omitted from denominator | Raw text kept. |
| Intentionally unsupported claim with `[1]` | Coverage/validity 1/1 | Raw claim kept; support was not checked. |
| Valid `[1]` plus malformed `[x]` | Regex ignores `[x]` | Raw text kept. |
| Generated text above requested word limit | Coverage/validity 1/1 | Raw overlong text kept; the limit is prompt-only. |
| Empty top-source answer, invalid raw text | Fallback triggered | Final text is ` [1]`; both final citation metrics are 0. |
| No source, raw `[1]` | Out-of-range marker triggers fallback | Direct helper returns abstention text with 0/0 metrics and `used_fallback=true`. |
| Generator raises an error | No text reaches the gate | Error propagates; no fallback result. |

Across these fixtures, 10 returned normally; 4 used fallback; 1 raised a
synthetic generator error. Four raw outputs had numeric validity below 1.0
(including no numeric marker), two had coverage below 0.5, and two final
outputs still had invalid 0/0 metrics. These counts overlap and are **not**
production fallback, failure, or citation rates.

## Other boundaries and failure paths

- Context construction includes at most 700 characters of each source's
  question body and 1,400 characters of its answer. The extractive fallback
  uses the **full** top-source answer (up to `maximum_answer_words` words),
  including text that may not have appeared in the prompt. The source-truncation
  fixture confirmed a deliberately placed tail absent from the prompt but
  present in the fallback.
- The configured 140-word limit is requested in the prompt. It is not
  enforced on an accepted generation. Fallback slices to the configured
  number of answer words, then appends an ellipsis when truncated and `[1]`;
  those appended tokens are outside the sliced answer-word count.
- `rerank_candidates` rejects empty candidates. If an empty source frame
  nonetheless reaches `CodeAnswerPipeline.answer`, its `sources.iloc[0]`
  access raises `IndexError`. There is no explicit empty-evidence abstention
  path in the pipeline. A below-threshold abstention is separate: it sets
  `used_fallback=false` and omits citation metric keys.
- Ollama transport, HTTP, or response-parsing errors are not converted into
  extractive fallback or abstention; they propagate. A mocked request timeout
  is covered by a test.
- The first source's raw cross-encoder score controls abstention. It is not
  a source-support judgment, and the gate does not inspect whether the
  answer text is entailed by any source.

## Historical 61.9% rate: evidence status

The historical [aggregate](../results/experiment_summary.json) reports an
extractive fallback rate of **0.619048** and citation validity **0.972789**.
No raw generated/final answers, labels, source lists, gate triggers, or
per-example logs were found in tracked results or local artifacts. The
aggregate therefore cannot tell us how often missing citations, invalid
numbers, short answers, or generator errors occurred. We cannot reproduce
its denominator or reconcile its citation-validity value with particular
final outputs. The fixture's 4-of-10 fallback count is deliberately
incomparable to that historical rate. D2 must save per-example inputs,
raw/final outputs and trigger reasons before making a new rate claim.

## Prioritized fixes and evaluation work

1. **D2: make every result traceable.** Record raw generation, raw/final
   citation metrics with denominators, fallback reason, source IDs, threshold
   decision, abstention, generator errors, and stage timings. Use separate
   labeled development/calibration/test examples.
2. **Handle absent or empty evidence explicitly.** Define an abstention or
   controlled failure for no candidates, no sources, and an empty top-source
   answer. Avoid returning an invalid bare `[1]` fallback.
3. **Tighten output-format behavior on development examples.** Decide
   whether each substantial line needs a citation, reject malformed
   citation-like tokens, enforce the word limit in code, and state whether
   fallback may quote answer text omitted from the generation prompt.
4. **D4: assess source support separately.** Use human judgments on claims
   and cited evidence; syntax checks and the historical embedding proxy
   cannot establish correctness or faithfulness.

These are proposals, except for the measured fixture behavior and the
documentation corrections in D1. Changing the gate requires a new labeled
evaluation; it should not be justified by the synthetic fixture rate.
