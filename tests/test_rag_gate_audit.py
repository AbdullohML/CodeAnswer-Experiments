"""Targeted D1 probes for citation, fallback and generation failures."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from codeanswer.rag import OllamaGenerator
from codeanswer.rag_gate_audit import run


def test_fixture_audit_records_raw_and_final_behavior(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = run(root, tmp_path / "fixture_audit.json")
    cases = {case["case"]: case for case in result["cases"]}

    assert result["fixture_counts"] == {
        "total": 11,
        "returned": 10,
        "fallback": 4,
        "generator_errors": 1,
        "raw_invalid_numeric_or_absent": 4,
        "raw_coverage_below_half": 2,
        "final_invalid_citation_metrics": 2,
    }
    assert cases["out_of_range_number"]["used_fallback"] is True
    assert cases["half_lines_cited"]["used_fallback"] is False
    assert cases["half_lines_cited"]["raw_citation_metrics"]["citation_coverage"] == 0.5
    assert cases["unsupported_claim_valid_syntax"]["used_fallback"] is False
    assert cases["malformed_and_valid_citation"]["used_fallback"] is False
    assert cases["over_word_limit"]["final_word_count"] > 5
    assert cases["empty_answer_fallback"]["final_citation_metrics"] == {
        "citation_coverage": 0.0,
        "citation_validity": 0.0,
    }
    assert cases["generator_error"]["error_type"] == "RuntimeError"
    assert result["empty_pipeline_error_type"] == "IndexError"
    assert result["below_threshold"] == {
        "abstained": True,
        "used_fallback": False,
        "citation_metrics_present": False,
    }
    assert result["source_truncation"]["tail_in_prompt"] is False
    assert result["source_truncation"]["tail_in_fallback"] is True


def test_ollama_transport_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise requests.Timeout("synthetic timeout")

    monkeypatch.setattr(requests, "post", fail)
    generator = OllamaGenerator(model_name="synthetic-model")
    with pytest.raises(requests.Timeout, match="synthetic timeout"):
        generator.generate(system_prompt="system", user_prompt="user", max_tokens=10)
