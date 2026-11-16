"""Paired A4 ranking, scoring, and gain/loss checks."""

import numpy as np
import pandas as pd
import pytest

from codeanswer.reranking_ablation import order_candidates, score_query, summarize


class RecordingScorer:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, *, batch_size):
        self.calls.append((pairs, batch_size))
        return np.asarray([0.5 if "first" in body else 0.49 for _, body in pairs])


def test_vote_bonus_and_deterministic_ties():
    ids = [4, 5]
    scores = np.asarray([0.5, 0.49])
    assert order_candidates(ids, scores, [0, 100], 0)[0] == [4, 5]
    assert order_candidates(ids, scores, [0, 100], 0.03)[0] == [5, 4]
    assert order_candidates(ids, np.asarray([1.0, 1.0]), [0, 0], 0)[0] == ids
    with pytest.raises(ValueError, match="invalid"):
        order_candidates(ids, np.asarray([np.nan, 1.0]), [0, 0], 0)


def test_union_scored_once_per_mode_and_input_text_changes():
    corpus = {4: ("first title", "body one", 0), 5: ("second title", "body two", 100)}
    candidates = {"bm25": [4, 5], "dense": [5, 4], "hybrid": [4, 5]}
    scorer = RecordingScorer()
    records, timing = score_query("question", candidates, corpus, scorer, 8, 0.03)
    assert len(scorer.calls) == 2
    assert all(len(pairs) == 2 and batch == 8 for pairs, batch in scorer.calls)
    assert scorer.calls[0][0][0] == ("question", "first title\nbody one")
    assert scorer.calls[1][0][0] == ("question", "body one")
    assert len(records) == 12
    assert set(timing) == {"title_body", "body_only"}
    assert next(
        row
        for row in records
        if row["system"] == "bm25"
        and row["input_mode"] == "title_body"
        and row["vote_bonus"]
    )["ranked_doc_ids@40"] == [5, 4]


def test_summary_handles_missing_gold_and_counts_paired_gains_losses():
    queries = pd.DataFrame({"relevant_doc_id": [4, 5]})
    base = list(range(100, 139))
    matrices = {
        name: np.asarray([[4, *base], [*base, 5]])
        for name in ("bm25", "dense", "hybrid")
    }
    records = []
    for system in matrices:
        for mode in ("title_body", "body_only"):
            for bonus in (False, True):
                for i, ids in enumerate(([*base, 4], [5, *base])):
                    records.append(
                        {
                            "system": system,
                            "input_mode": mode,
                            "vote_bonus": bonus,
                            "query_index": i,
                            "ranked_doc_ids@40": ids,
                        }
                    )
    table, _ = summarize(
        pd.DataFrame(records),
        queries,
        matrices,
        {"title_body": [1.0, 2.0], "body_only": [3.0, 4.0]},
    )
    assert len(table) == 12
    assert set(table["gained@10"]) == {1}
    assert set(table["lost@10"]) == {1}
    assert set(table["rank_improved"]) == {1}
    assert set(table["rank_worsened"]) == {1}
