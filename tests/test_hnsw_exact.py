"""B2 same-space overlap and matched-search checks."""

import faiss
import numpy as np
import pytest

from codeanswer.hnsw_exact import paired_overlap, search_and_time


def test_overlap_is_distinct_from_relevance() -> None:
    exact = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    ann = np.array([[2, 9, 3, 4], [5, 8, 10, 11]])
    np.testing.assert_array_equal(paired_overlap(exact, ann, 2), [0.5, 0.5])
    np.testing.assert_array_equal(paired_overlap(exact, ann, 4), [0.75, 0.5])
    with pytest.raises(ValueError, match="cutoff"):
        paired_overlap(exact, ann, 5)


def test_exact_and_hnsw_search_share_query_budget() -> None:
    rng = np.random.default_rng(8)
    corpus = rng.normal(size=(50, 8)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = corpus[[2, 11, 27]]
    exact = faiss.IndexFlatIP(8)
    exact.add(corpus)
    ann = faiss.IndexHNSWFlat(8, 8, faiss.METRIC_INNER_PRODUCT)
    ann.hnsw.efConstruction = 40
    ann.hnsw.efSearch = 40
    ann.add(corpus)
    settings = {"top_k": 40, "latency_queries": 2, "warmup_queries": 1}
    exact_ids, exact_time = search_and_time(exact, queries, settings)
    ann_ids, ann_time = search_and_time(ann, queries, settings)
    assert exact_ids.shape == ann_ids.shape == (3, 40)
    assert exact_ids[:, 0].tolist() == ann_ids[:, 0].tolist() == [2, 11, 27]
    assert exact_time["queries"] == ann_time["queries"] == 2
