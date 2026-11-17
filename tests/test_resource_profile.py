"""Checks for B4 timing summaries and production-equivalent corpus fetch."""

from __future__ import annotations

import pandas as pd
import pyarrow.dataset as pads
import pytest

from codeanswer.resource_profile import fetch_candidates, percentiles


def test_percentiles_reject_missing_or_invalid_samples() -> None:
    assert percentiles([1.0, 2.0, 3.0]) == {"p50_ms": 2.0, "p95_ms": 2.9}
    with pytest.raises(ValueError):
        percentiles([])
    with pytest.raises(ValueError):
        percentiles([float("nan")])


def test_fetch_candidates_matches_pipeline_order_and_cleaning(tmp_path) -> None:
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "title": ["one", "two", "three"],
            "question_body": ["first", "second", "third"],
            "answer": ["A", "B", "C"],
        }
    ).to_parquet(corpus)
    dataset = pads.dataset(corpus, format="parquet")
    assert fetch_candidates(dataset, [3, 1]) == [3, 1]
    with pytest.raises(ValueError):
        fetch_candidates(dataset, [1, 1])
    with pytest.raises(KeyError):
        fetch_candidates(dataset, [99])
