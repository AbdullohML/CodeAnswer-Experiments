"""D2 split isolation, trace, and aggregate checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from codeanswer.provenance import sha256_file
from codeanswer.rag_evaluation import (
    TracedGenerator,
    calibrate,
    evaluate_row,
    load_labels,
    summarize,
)


def _row(identifier: str, split: str, behavior: str) -> dict:
    return {
        "id": identifier,
        "split": split,
        "query": identifier,
        "expected_behavior": behavior,
        "family_id": None,
    }


class CalibrationPipeline:
    answerability_threshold = 999.0

    def retrieve_and_rerank(self, query: str) -> tuple[pd.DataFrame, dict]:
        scores = {"cal-answer": 0.8, "cal-abstain": 0.2}
        if query not in scores:
            raise AssertionError("A noncalibration query selected the threshold")
        return pd.DataFrame([{"reranker_score": scores[query]}]), {
            "retrieval_ms": 1.0,
            "rerank_ms": 2.0,
        }


def test_calibration_ignores_development_and_test_labels() -> None:
    rows = [
        _row("development", "development", "answer"),
        _row("cal-answer", "calibration", "answer"),
        _row("cal-abstain", "calibration", "abstain"),
        _row("test", "test", "abstain"),
    ]
    records, chosen = calibrate(rows, CalibrationPipeline())
    assert [row["id"] for row in records] == ["cal-answer", "cal-abstain"]
    assert chosen["threshold"] == 0.8
    rows[-1]["expected_behavior"] = "answer"
    assert calibrate(rows, CalibrationPipeline())[1] == chosen


def test_manifest_rejects_changed_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    manifest_path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_row("one", "test", "answer")) + "\n")
    manifest_path.write_text(json.dumps({"labels_sha256": "wrong", "counts": {}}))
    with pytest.raises(ValueError, match="frozen D2 manifest"):
        load_labels(path, manifest_path)


class FakeGenerator:
    model_name = "fake"

    def generate(
        self, *, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> dict:
        assert max_tokens == 320
        return {"text": "A supported answer. [1]", "elapsed_ms": 1.0}


class AnswerPipeline:
    answerability_threshold = 0.5

    def __init__(self, generator: TracedGenerator) -> None:
        self.generator = generator

    def answer(self, query: str) -> dict:
        self.generator.generate(
            system_prompt="system", user_prompt=f"Sources for {query}", max_tokens=320
        )
        return {
            "query": query,
            "answer": "A supported answer. [1]",
            "raw_answer": "A supported answer. [1]",
            "abstained": False,
            "used_fallback": False,
            "sources": [{"doc_id": 1, "answer": "A supported answer."}],
            "total_ms": 3.0,
            "generation_ms": 1.0,
        }


def test_trace_and_denominators_include_errors() -> None:
    generator = TracedGenerator(FakeGenerator())
    answered = evaluate_row(
        _row("one", "test", "answer"), AnswerPipeline(generator), generator
    )
    assert answered["raw_answer"] == answered["answer"]
    assert answered["user_prompt"] == "Sources for one"
    assert answered["generation_attempted"] is True

    class FailingPipeline:
        answerability_threshold = 0.5

        def answer(self, _query: str) -> dict:
            raise RuntimeError("offline")

    failed = evaluate_row(_row("two", "test", "abstain"), FailingPipeline(), generator)
    assert failed["error"]["type"] == "RuntimeError"
    assert failed["generation_attempted"] is False
    result = summarize([answered, failed])
    assert result["behavior_accuracy"] == 0.5
    assert result["error_rate"] == 0.5
    assert result["fallback_rate_among_answers"] == 0.0


def test_saved_run_is_traceable_to_per_query_records() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/evaluation/d2/run_local_b1b2"
    manifest = json.loads((output / "run_manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    labels, _ = load_labels(
        root / "results/evaluation/d2/labels.jsonl",
        root / "results/evaluation/d2/manifest.json",
    )
    assert len(labels) == 60
    for name, expected in manifest["files_sha256"].items():
        assert sha256_file(output / name) == expected
    chosen = json.loads((output / "selected_threshold.json").read_text())
    calibration = [
        json.loads(line)
        for line in (output / "calibration.jsonl").read_text().splitlines()
    ]
    assert len(calibration) == 20
    assert all(row["split"] == "calibration" for row in calibration)
    assert chosen["threshold"] == manifest["selected_threshold"]
    for split in ("development", "test"):
        records = [
            json.loads(line)
            for line in (output / f"{split}.jsonl").read_text().splitlines()
        ]
        assert all(row["split"] == split for row in records)
        assert all(
            row["answerability_threshold"] == chosen["threshold"] for row in records
        )
        assert summarize(records) == summary[split]
