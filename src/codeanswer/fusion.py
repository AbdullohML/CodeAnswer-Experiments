"""Deterministic reciprocal rank fusion for dense and lexical rankings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _valid_ids(ids: Sequence[int], depth: int) -> list[int]:
    """Validate one padded ranking and return its first depth real IDs."""
    if depth <= 0 or depth > len(ids):
        raise ValueError("Candidate depth exceeds ranking length")
    selected = [int(value) for value in ids[:depth]]
    if any(value < -1 for value in selected):
        raise ValueError("Ranking contains invalid document ID")
    if -1 in selected:
        first_padding = selected.index(-1)
        if any(value != -1 for value in selected[first_padding:]):
            raise ValueError("Ranking has a document after padding")
        selected = selected[:first_padding]
    if len(selected) != len(set(selected)):
        raise ValueError("Ranking contains duplicate document IDs")
    return selected


def fuse_rrf(
    dense_ids: Sequence[int],
    bm25_ids: Sequence[int],
    *,
    candidate_depth: int,
    output_depth: int,
    rrf_constant: int,
) -> tuple[list[int], list[float]]:
    """Fuse two lists with score sum 1/(constant + one-based rank).

    Cross-source duplicate IDs receive both contributions. Exact score ties
    break by ascending doc_id. Results are padded with -1 to output_depth.
    """
    if rrf_constant <= 0 or output_depth <= 0:
        raise ValueError("RRF constant and output depth must be positive")
    dense = _valid_ids(dense_ids, candidate_depth)
    bm25 = _valid_ids(bm25_ids, candidate_depth)
    scores: dict[int, float] = {}
    for ranking in (dense, bm25):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_constant + rank)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    selected = ordered[:output_depth]
    return (
        selected + [-1] * (output_depth - len(selected)),
        [scores[doc_id] for doc_id in selected],
    )


def union_contribution(
    dense_ids: np.ndarray,
    bm25_ids: np.ndarray,
    relevant_ids: np.ndarray,
    *,
    candidate_depth: int,
) -> tuple[dict[str, float | int], list[str]]:
    """Count where the single positive occurs in the two source lists."""
    dense = np.asarray(dense_ids, dtype=np.int64)
    bm25 = np.asarray(bm25_ids, dtype=np.int64)
    relevant = np.asarray(relevant_ids, dtype=np.int64)
    if (
        dense.ndim != 2
        or bm25.ndim != 2
        or dense.shape != bm25.shape
        or relevant.shape != (dense.shape[0],)
        or dense.shape[0] == 0
    ):
        raise ValueError("Source rankings and labels must align")
    counts = {"both": 0, "dense_only": 0, "bm25_only": 0, "neither": 0}
    cases: list[str] = []
    for dense_row, bm25_row, label in zip(dense, bm25, relevant, strict=True):
        in_dense = int(label) in _valid_ids(dense_row, candidate_depth)
        in_bm25 = int(label) in _valid_ids(bm25_row, candidate_depth)
        case = (
            "both"
            if in_dense and in_bm25
            else "dense_only"
            if in_dense
            else "bm25_only"
            if in_bm25
            else "neither"
        )
        counts[case] += 1
        cases.append(case)
    total = len(cases)
    return {
        "queries": total,
        **counts,
        "union_recall": round(1.0 - counts["neither"] / total, 6),
    }, cases
