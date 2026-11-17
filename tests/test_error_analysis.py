"""Checks paired C3 metrics and resampling semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from codeanswer.error_analysis import (
    _group_table,
    paired_bootstrap,
    reciprocal_rank,
    relevance_at,
)


def test_one_positive_rank_metrics() -> None:
    ranks = np.array([1, 10, 11, 0])
    np.testing.assert_array_equal(relevance_at(ranks, 10), [1, 1, 0, 0])
    np.testing.assert_array_equal(relevance_at(ranks, 40), [1, 1, 1, 0])
    np.testing.assert_allclose(reciprocal_rank(ranks), [1, 0.1, 0, 0])
    with pytest.raises(ValueError):
        relevance_at(np.array([-1]), 10)


def test_bootstrap_resamples_paired_differences_deterministically() -> None:
    deltas = np.array([1.0, -1.0, 0.0, 1.0, 0.0])
    values = {"gain": deltas, "loss": -deltas}
    first = paired_bootstrap(values, seed=7, replicates=1000, level=0.95)
    second = paired_bootstrap(values, seed=7, replicates=1000, level=0.95)
    assert first == second
    assert first["gain"]["mean_delta"] == 0.2
    assert first["gain"]["ci_low"] == -first["loss"]["ci_high"]
    assert first["gain"]["ci_high"] == -first["loss"]["ci_low"]
    with pytest.raises(ValueError):
        paired_bootstrap({"a": [1, 2], "b": [1]}, seed=1, replicates=100, level=0.95)


def test_group_delta_is_paired_mean() -> None:
    paired = pd.DataFrame(
        {
            "bm25_rank": [0, 1, 0],
            "dense_rank": [1, 11, 0],
            "hybrid_rank": [11, 1, 0],
        }
    )
    table = _group_table(paired, {"all": np.array([True, True, True])})
    assert table.iloc[0]["hybrid_minus_dense_recall@10"] == 0.0
    assert table.iloc[0]["dense_recall@10"] == 0.333333
    assert table.iloc[0]["hybrid_recall@10"] == 0.333333
