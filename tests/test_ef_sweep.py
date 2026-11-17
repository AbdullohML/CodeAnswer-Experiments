"""B3 search timing and frozen sweep settings."""

import faiss
import numpy as np

from codeanswer.ef_sweep import SETTINGS, search_and_time


def test_sweep_times_sequential_queries_and_preserves_top40() -> None:
    rng = np.random.default_rng(17)
    corpus = rng.normal(size=(80, 8)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    index = faiss.IndexHNSWFlat(8, 8, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.add(corpus)
    index.hnsw.efSearch = 64
    ids, timing = search_and_time(index, corpus[[3, 20, 45]], SETTINGS)
    assert ids.shape == (3, 40)
    assert ids[:, 0].tolist() == [3, 20, 45]
    assert timing["queries"] == 3
    assert timing["p95_ms"] >= timing["p50_ms"] > 0
    assert timing["sequential_qps"] > 0
