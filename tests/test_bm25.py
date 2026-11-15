"""A1 lexical retrieval and frozen evaluation checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.evaluate_bm25 as evaluation_script
from codeanswer.bm25 import build_index, match_expression, open_index, search
from codeanswer.evaluation_protocol import freeze_known_item_splits


def _corpus(path: Path) -> None:
    rows = [
        {
            "doc_id": i,
            "question_id": 1000 + i,
            "title": f"How to fix code_token_{i} error in version 3.11?",
            "question_body": f"code_token_{i} error in version 3.11 explanation",
        }
        for i in range(12)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_bm25_build_search_and_identity(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.parquet"
    _corpus(corpus)
    index = tmp_path / "bm25.sqlite"
    config = Path("configs/bm25.json")
    manifest = build_index(corpus, index, config)
    assert manifest["corpus_rows"] == 12
    assert manifest["index_size_bytes"] > 0
    connection, loaded = open_index(index, corpus, config)
    try:
        assert loaded["index_sha256"] == manifest["index_sha256"]
        assert search(connection, "code_token_5", 10)[0][0] == 5
        assert search(connection, "error", 12) == search(connection, "error", 12)
        assert [doc_id for doc_id, _ in search(connection, "error", 12)] == list(
            range(12)
        )
        assert search(connection, "!!!", 10) == []
        assert match_expression('foo OR "bar"') == '"foo" OR "or" OR "bar"'
        assert match_expression("3.11 C++") == '"3" OR "11" OR "c"'
        with pytest.raises(ValueError, match="top_k"):
            search(connection, "error", 0)
    finally:
        connection.close()
    reopened, _ = open_index(index, corpus, config)
    try:
        assert [doc_id for doc_id, _ in search(reopened, "error", 12)] == list(
            range(12)
        )
    finally:
        reopened.close()
    with pytest.raises(FileExistsError):
        build_index(corpus, index, config)
    other_corpus = tmp_path / "other.parquet"
    pd.read_parquet(corpus).iloc[:-1].to_parquet(other_corpus, index=False)
    with pytest.raises(ValueError, match="differs"):
        open_index(index, other_corpus, config)


def test_bm25_frozen_evaluation_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.parquet"
    _corpus(corpus)
    index = tmp_path / "bm25.sqlite"
    config = Path("configs/bm25.json")
    build_index(corpus, index, config)
    frozen = tmp_path / "frozen"
    freeze_known_item_splits(
        corpus,
        frozen,
        query_count=12,
        seed=12,
        development_count=4,
        calibration_count=4,
    )
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_bm25.py",
            "--corpus",
            str(corpus),
            "--index",
            str(index),
            "--config",
            str(config),
            "--evaluation-manifest",
            str(frozen / "manifest.json"),
            "--split",
            "test",
            "--output-dir",
            str(output),
        ],
    )
    evaluation_script.main()
    result = json.loads((output / "results.json").read_text())
    rows = pd.read_csv(output / "per_query_rankings.csv")
    assert result["quality"]["queries"] == 4
    assert result["quality"]["recall@40"] == 1.0
    assert len(rows) == 4
    assert "query" not in rows.columns
    assert all(rows["rank"] > 0)
    assert all(
        len(json.loads(value)) == 40 and json.loads(value)[-1] == -1
        for value in rows["retrieved_doc_ids@40"]
    )


def test_bm25_rejects_noncontiguous_doc_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "bad.parquet"
    _corpus(corpus)
    frame = pd.read_parquet(corpus)
    frame.loc[3, "doc_id"] = 13
    frame.to_parquet(corpus, index=False)
    with pytest.raises(ValueError, match="contiguous"):
        build_index(corpus, tmp_path / "bad.sqlite", "configs/bm25.json")
    assert not (tmp_path / "bad.sqlite").exists()
