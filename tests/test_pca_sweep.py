"""B1 fixed fit sample and nested PCA search checks."""

from __future__ import annotations

import gzip
import json

import faiss
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from codeanswer.pca_sweep import (
    evaluate_dimension,
    load_settings,
    project_prefix,
    sample_bytes,
    sample_ids,
)


def test_nested_prefix_matches_fitted_pca_projection() -> None:
    rng = np.random.default_rng(7)
    training = rng.normal(size=(50, 8)).astype(np.float32)
    training /= np.linalg.norm(training, axis=1, keepdims=True)
    pca = PCA(n_components=6, svd_solver="randomized", random_state=11).fit(training)
    query = training[:5]
    for dimension in (2, 4, 6):
        actual = project_prefix(pca, query, dimension)
        expected = pca.transform(query)[:, :dimension]
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
    with pytest.raises(ValueError, match="dimension"):
        project_prefix(pca, query, 7)


def test_fit_ids_are_stable_and_saved_in_fit_order() -> None:
    first = sample_ids(100, 30, 20260714)
    second = sample_ids(100, 30, 20260714)
    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == 30
    assert all(0 <= i < 100 for i in first)
    raw = gzip.decompress(sample_bytes(first)).decode().splitlines()
    assert raw[0] == "doc_id"
    assert [int(value) for value in raw[1:]] == first.tolist()


def test_exact_search_metrics_are_paired_and_timing_scoped() -> None:
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(50, 6)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(6)
    index.add(vectors)
    queries = pd.DataFrame({"relevant_doc_id": [4, 12, 32]})
    rankings, quality, per_query, timing = evaluate_dimension(
        index,
        vectors[[4, 12, 32]],
        queries,
        {"search_batch_size": 2, "latency_queries": 2, "warmup_queries": 1},
        None,
        vectors[[4, 12, 32]],
    )
    assert rankings.shape == (3, 40)
    assert per_query["rank"].tolist() == [1, 1, 1]
    assert quality["recall@10"] == quality["recall@40"] == 1
    assert timing["queries"] == 2
    assert timing["transform_p95_ms"] == 0


def test_pinned_settings_reject_changes(tmp_path) -> None:
    path = tmp_path / "settings.json"
    from codeanswer.pca_sweep import EXPECTED

    path.write_text(json.dumps({**EXPECTED, "sample_size": 1000}))
    with pytest.raises(ValueError, match="Unsupported"):
        load_settings(path)
