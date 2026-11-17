"""D3 candidate swapping and paired-output invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codeanswer.provenance import sha256_file
from codeanswer.rag import CodeAnswerPipeline
from codeanswer.rag_comparison import pair_rows, strategy_summary
from codeanswer.rag_retrievers import CandidateRetriever


class FakeEncoder:
    def encode(self, _texts: list[str], *, batch_size: int) -> np.ndarray:
        assert batch_size == 1
        return np.ones((1, 2), dtype=np.float32)


def test_candidate_strategies_change_only_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "codeanswer.rag_retrievers.transform_queries", lambda _pca, value: value
    )
    monkeypatch.setattr(
        "codeanswer.rag_retrievers.search_index",
        lambda _index, _value, *, top_k: (
            np.array([[0.9, 0.8]], dtype=np.float32),
            np.array([[1, 2]], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        "codeanswer.rag_retrievers.search_bm25",
        lambda _connection, _query, _depth: [(2, -5.0), (3, -4.0)],
    )
    results = {}
    for strategy in ("dense", "bm25", "hybrid"):
        retriever = CandidateRetriever(strategy, FakeEncoder(), None, None, None)
        ids, scores, timing = retriever.retrieve("same question", 2)
        results[strategy] = ids
        assert len(ids) == len(scores) == 2
        assert retriever.last_doc_ids == ids
        assert timing["retrieval_ms"] >= 0
    assert results == {"dense": [1, 2], "bm25": [2, 3], "hybrid": [2, 1]}


def test_pipeline_accepts_candidate_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    class Adapter:
        def retrieve(
            self, query: str, depth: int
        ) -> tuple[list[int], list[float], dict]:
            assert (query, depth) == ("question", 2)
            return [7, 8], [0.7, 0.6], {"retrieval_ms": 1.0, "encode_ms": 0.0}

    pipeline = object.__new__(CodeAnswerPipeline)
    pipeline.candidate_retriever = Adapter()
    pipeline.retrieval_candidates = 2
    pipeline.reranker = object()
    pipeline.final_sources = 1
    pipeline.reranker_batch_size = 2
    pipeline._fetch_documents = lambda ids: pd.DataFrame(
        {
            "doc_id": ids,
            "title": ["a", "b"],
            "question_body": ["x", "y"],
            "answer": ["one", "two"],
            "answer_score": [1, 1],
        }
    )
    monkeypatch.setattr(
        "codeanswer.rag.rerank_candidates",
        lambda _query, frame, **_kwargs: frame.head(1),
    )
    sources, timing = pipeline.retrieve_and_rerank("question")
    assert sources["doc_id"].tolist() == [7]
    assert sources["candidate_score"].tolist() == [0.7]
    assert timing["fetch_ms"] >= 0
    assert timing["rerank_model_ms"] >= 0


def test_paired_records_require_same_ids_and_labels() -> None:
    base = {
        "id": "one",
        "query": "question",
        "expected_behavior": "answer",
        "actual_behavior": "answer",
        "used_fallback": False,
        "total_ms": 1.0,
        "answer": "Answer [1]",
        "raw_answer": "Answer [1]",
        "sources": [{"doc_id": 1}],
    }
    records = {name: [base.copy()] for name in ("dense", "bm25", "hybrid")}
    assert pair_rows(records)[0]["hybrid_source_doc_ids"] == [1]
    records["bm25"][0]["id"] = "different"
    with pytest.raises(ValueError, match="Paired query"):
        pair_rows(records)


def test_strategy_summary_separates_generator_errors() -> None:
    error = {
        "expected_behavior": "answer",
        "actual_behavior": "error",
        "behavior_correct": False,
        "generation_attempted": True,
        "candidate_count": 40,
    }
    result = strategy_summary([error])
    assert result["counts"]["generator_errors"] == 1
    assert result["counts"]["pre_generation_errors"] == 0
    assert result["error_rate"] == 1.0
    assert result["fallback_rate_among_answers"] is None


def test_saved_d3_run_has_fixed_pairs_and_recomputable_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/evaluation/d3/run_local_b1b2"
    manifest = json.loads((output / "run_manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert sha256_file(output / name) == expected
    for phase, suffix, count in (("main", "", 28), ("repeat", "_repeat", 6)):
        records = {
            strategy: [
                json.loads(line)
                for line in (output / f"{strategy}{suffix}.jsonl")
                .read_text()
                .splitlines()
            ]
            for strategy in ("dense", "bm25", "hybrid")
        }
        assert len(pair_rows(records)) == count
        for strategy, rows in records.items():
            assert all(row["candidate_count"] == 40 for row in rows)
            assert all(
                row["answerability_threshold"] == manifest["answerability_threshold"]
                for row in rows
            )
            assert strategy_summary(rows) == summary[phase][strategy]
