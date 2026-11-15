"""A2 RRF ranking and saved-source alignment checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codeanswer.fusion import fuse_rrf, union_contribution
from codeanswer.provenance import sha256_file
from scripts.evaluate_hybrid import load_hybrid_settings, load_saved_bm25


def test_rrf_deduplicates_and_breaks_ties_by_doc_id() -> None:
    ids, scores = fuse_rrf(
        [9, 2, 3, -1],
        [2, 4, 9, -1],
        candidate_depth=4,
        output_depth=5,
        rrf_constant=60,
    )
    assert ids == [2, 9, 4, 3, -1]
    assert len(scores) == 4
    assert scores[0] == pytest.approx(1 / 62 + 1 / 61)
    assert scores == sorted(scores, reverse=True)
    assert (
        ids
        == fuse_rrf(
            [9, 2, 3, -1],
            [2, 4, 9, -1],
            candidate_depth=4,
            output_depth=5,
            rrf_constant=60,
        )[0]
    )

    tied, _ = fuse_rrf(
        [8, -1],
        [5, -1],
        candidate_depth=2,
        output_depth=4,
        rrf_constant=60,
    )
    assert tied == [5, 8, -1, -1]


@pytest.mark.parametrize(
    "dense,bm25",
    [([1, 1], [2, 3]), ([1, -1, 2], [3, 4, 5]), ([-2, -1], [1, 2])],
)
def test_rrf_rejects_invalid_source_rankings(dense: list[int], bm25: list[int]) -> None:
    with pytest.raises(ValueError):
        fuse_rrf(
            dense,
            bm25,
            candidate_depth=len(dense),
            output_depth=4,
            rrf_constant=60,
        )


def test_union_contribution_counts_each_source() -> None:
    dense = np.asarray([[1, 2], [3, 4], [5, 12], [7, 8]])
    bm25 = np.asarray([[1, 9], [9, 10], [6, 5], [9, 10]])
    relevant = np.asarray([1, 3, 6, 11])
    summary, cases = union_contribution(dense, bm25, relevant, candidate_depth=2)
    assert cases == ["both", "dense_only", "bm25_only", "neither"]
    assert summary == {
        "queries": 4,
        "both": 1,
        "dense_only": 1,
        "bm25_only": 1,
        "neither": 1,
        "union_recall": 0.75,
    }


def test_saved_bm25_must_match_frozen_ids_and_labels(tmp_path: Path) -> None:
    queries = pd.DataFrame(
        {
            "query_id": [100, 101],
            "family_id": [100, 101],
            "relevant_doc_id": [2, 3],
        }
    )
    rankings = tmp_path / "a1_bm25_development_per_query.csv"
    with rankings.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "query_id",
                "family_id",
                "relevant_doc_id",
                "retrieved_doc_ids@40",
            ],
        )
        writer.writeheader()
        for query_id, doc_id in ((100, 2), (101, 3)):
            writer.writerow(
                {
                    "query_id": query_id,
                    "family_id": query_id,
                    "relevant_doc_id": doc_id,
                    "retrieved_doc_ids@40": json.dumps([doc_id] + [-1] * 39),
                }
            )
    result = {
        "split": "development",
        "smoke_limit": None,
        "evaluation_manifest_sha256": "manifest-hash",
        "per_query_sha256": sha256_file(rankings),
        "config_sha256": "config-hash",
        "index": {"index_sha256": "index-hash", "corpus_rows": 10},
        "quality": {"queries": 2},
    }
    (tmp_path / "a1_bm25_development_results.json").write_text(json.dumps(result))
    loaded, _, _ = load_saved_bm25(
        "development",
        queries,
        results_dir=tmp_path,
        evaluation_manifest_sha256="manifest-hash",
        index_sha256="index-hash",
        config_sha256="config-hash",
    )
    assert loaded.shape == (2, 40)
    assert loaded[:, 0].tolist() == [2, 3]
    altered = queries.copy()
    altered.loc[1, "relevant_doc_id"] = 7
    with pytest.raises(ValueError, match="differ from C2"):
        load_saved_bm25(
            "development",
            altered,
            results_dir=tmp_path,
            evaluation_manifest_sha256="manifest-hash",
            index_sha256="index-hash",
            config_sha256="config-hash",
        )


def test_hybrid_config_exposes_pinned_rrf_constant() -> None:
    settings = load_hybrid_settings("configs/hybrid.json")
    assert settings["rrf_constant"] == 60
    assert settings["candidate_depths"] == [20, 40]
