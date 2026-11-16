"""A3 paired comparison inputs, labels, and metric checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from codeanswer.fair_comparison import (
    _ranking_matrix,
    benchmark_method,
    build_paired_table,
    exact_term_slice,
    load_settings,
)


def _padded(*ids: int) -> list[int]:
    return list(ids) + [-1] * (40 - len(ids))


def test_paired_metrics_use_identical_queries_and_labels() -> None:
    queries = pd.DataFrame(
        {
            "query_id": [10, 11, 12],
            "family_id": [10, 11, 12],
            "relevant_doc_id": [1, 2, 3],
            "query": ["foo_bar error", "Python 3.11", "ordinary title"],
        }
    )
    matrices = {
        "bm25": np.asarray([_padded(1), _padded(8, 2), _padded(8)]),
        "dense": np.asarray([_padded(8, 1), _padded(2), _padded(3)]),
        "hybrid": np.asarray([_padded(1), _padded(2), _padded(8, 3)]),
    }
    indices, tags = exact_term_slice(queries, 1)
    assert indices == [0, 1]
    assert tags == ["underscore_identifier|error_word", "version_number", ""]
    paired, quality = build_paired_table(queries, matrices, slice_tags=tags)
    assert paired["query_id"].tolist() == [10, 11, 12]
    assert paired["hybrid_rank"].tolist() == [1, 1, 2]
    assert quality["bm25"]["recall@10"] == 0.666667
    assert quality["dense"]["recall@10"] == 1.0
    assert quality["hybrid"]["recall@40"] == 1.0


def test_saved_ranking_rejects_label_and_padding_drift() -> None:
    queries = pd.DataFrame(
        {"query_id": [10], "family_id": [10], "relevant_doc_id": [2]}
    )
    row = {
        "query_index": 0,
        "query_id": 10,
        "family_id": 10,
        "relevant_doc_id": 2,
        "rank": 2,
        "retrieved_doc_ids@40": json.dumps(_padded(1, 2)),
    }
    assert _ranking_matrix(pd.DataFrame([row]), queries, 5)[0, :2].tolist() == [1, 2]
    with pytest.raises(ValueError, match="labels"):
        _ranking_matrix(pd.DataFrame([{**row, "relevant_doc_id": 3}]), queries, 5)
    with pytest.raises(ValueError, match="padding"):
        _ranking_matrix(
            pd.DataFrame(
                [{**row, "retrieved_doc_ids@40": json.dumps([1, -1, 2] + [-1] * 37)}]
            ),
            queries,
            5,
        )


def test_live_benchmark_checks_saved_ranking() -> None:
    expected = np.asarray([_padded(1), _padded(2)])
    result = benchmark_method(
        lambda i: expected[i].tolist(),
        expected,
        latency_queries=2,
        warmup_queries=1,
    )
    assert result["queries"] == 2
    assert result["sequential_qps"] > 0
    with pytest.raises(ValueError, match="differs"):
        benchmark_method(
            lambda i: _padded(9),
            expected,
            latency_queries=2,
            warmup_queries=0,
        )


def test_fair_comparison_config_has_common_cutoffs() -> None:
    settings = load_settings("configs/fair_comparison.json")
    assert settings["recall_cutoffs"] == [1, 5, 10, 40]
    assert settings["exact_term_queries_per_category"] == 10
