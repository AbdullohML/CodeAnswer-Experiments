"""Synthetic, reproducible probes of the current RAG citation gate."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from codeanswer.provenance import sha256_file
from codeanswer.rag import (
    CodeAnswerPipeline,
    build_context,
    citation_metrics,
    extractive_fallback,
    generate_grounded_answer,
)


@dataclass
class FixtureGenerator:
    text: str | None
    model_name: str = "synthetic-fixture"

    def generate(
        self, *, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> dict:
        del system_prompt, user_prompt, max_tokens
        if self.text is None:
            raise RuntimeError("synthetic generator failure")
        return {
            "text": self.text,
            "elapsed_ms": 1.0,
            "prompt_tokens": 1,
            "output_tokens": 1,
        }


def _sources(answers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": [f"Source {i}" for i in range(1, len(answers) + 1)],
            "question_body": ["How to use the supported implementation?"]
            * len(answers),
            "answer": answers,
            "url": [f"https://example.test/{i}" for i in range(1, len(answers) + 1)],
        }
    )


def _cases() -> list[dict[str, Any]]:
    normal = ["Use the supported implementation."]
    return [
        {
            "name": "valid_number",
            "raw": "Use the implementation. [1]",
            "answers": normal,
            "max_words": 50,
        },
        {
            "name": "out_of_range_number",
            "raw": "Use the implementation. [3]",
            "answers": normal,
            "max_words": 50,
        },
        {
            "name": "missing_citation",
            "raw": "Use the implementation without citation.",
            "answers": normal,
            "max_words": 50,
        },
        {
            "name": "half_lines_cited",
            "raw": (
                "First long statement has a citation. [1]\n"
                "Second long statement has no citation."
            ),
            "answers": normal,
            "max_words": 50,
        },
        {
            "name": "short_uncited_line",
            "raw": "Short\nA long cited line follows. [1]",
            "answers": normal,
            "max_words": 50,
        },
        {
            "name": "unsupported_claim_valid_syntax",
            "raw": ("The moon is made of blue code. [1]"),
            "answers": normal,
            "max_words": 50,
        },
        {
            "name": "malformed_and_valid_citation",
            "raw": ("Use the implementation [1] and maybe [x]."),
            "answers": normal,
            "max_words": 50,
        },
        {
            "name": "over_word_limit",
            "raw": ("One two three four five six seven eight [1]"),
            "answers": normal,
            "max_words": 5,
        },
        {
            "name": "empty_answer_fallback",
            "raw": "No citation in this long text.",
            "answers": [""],
            "max_words": 50,
        },
        {
            "name": "no_source_fallback",
            "raw": "A cited claim without a source. [1]",
            "answers": [],
            "max_words": 50,
        },
        {"name": "generator_error", "raw": None, "answers": normal, "max_words": 50},
    ]


def _trigger_reasons(metrics: dict[str, float]) -> list[str]:
    reasons = []
    if metrics["citation_validity"] < 1.0:
        reasons.append("numeric_citation_validity_below_one")
    if metrics["citation_coverage"] < 0.5:
        reasons.append("long_line_coverage_below_half")
    return reasons


def _empty_pipeline_probe() -> str:
    """Exercise the public answer method after an empty rerank result."""
    pipeline = object.__new__(CodeAnswerPipeline)
    pipeline.retrieve_and_rerank = lambda _query: (
        pd.DataFrame(columns=["reranker_score"]),
        {"retrieval_ms": 0.0, "rerank_ms": 0.0},
    )
    try:
        pipeline.answer("A programming question")
    except Exception as exc:  # Audit the actual exception type.
        return type(exc).__name__
    return "no_error"


def _threshold_probe() -> dict[str, Any]:
    pipeline = object.__new__(CodeAnswerPipeline)
    pipeline.answerability_threshold = 0.5
    pipeline.retrieve_and_rerank = lambda _query: (
        pd.DataFrame([{"reranker_score": 0.4, "title": "Synthetic source"}]),
        {"retrieval_ms": 0.0, "rerank_ms": 0.0},
    )
    result = pipeline.answer("A programming question")
    return {
        "abstained": result["abstained"],
        "used_fallback": result["used_fallback"],
        "citation_metrics_present": {"citation_coverage", "citation_validity"}.issubset(
            result
        ),
    }


def _truncation_probe() -> dict[str, Any]:
    answer = "A" * 1401 + " TAIL_NOT_IN_PROMPT"
    sources = _sources([answer])
    context = build_context(sources)
    fallback = extractive_fallback(sources, maximum_words=140)
    return {
        "question_limit_chars": 700,
        "answer_limit_chars": 1400,
        "tail_in_prompt": "TAIL_NOT_IN_PROMPT" in context,
        "tail_in_fallback": "TAIL_NOT_IN_PROMPT" in fallback,
    }


def run(root: Path, output: Path) -> dict:
    records = []
    for case in _cases():
        raw = case["raw"]
        sources = _sources(case["answers"])
        record: dict[str, Any] = {
            "case": case["name"],
            "source_count": len(sources),
            "maximum_words": case["max_words"],
            "raw_text": raw,
            "raw_word_count": len(raw.split()) if raw is not None else None,
        }
        if raw is not None:
            record["raw_citation_metrics"] = citation_metrics(
                raw, source_count=len(sources)
            )
            record["raw_trigger_reasons"] = _trigger_reasons(
                record["raw_citation_metrics"]
            )
        try:
            result = generate_grounded_answer(
                "How should this work?",
                sources,
                generator=FixtureGenerator(raw),
                maximum_words=case["max_words"],
            )
        except Exception as exc:  # Record, rather than conceal, failures.
            record["error_type"] = type(exc).__name__
        else:
            record["used_fallback"] = result["used_fallback"]
            record["final_text"] = result["answer"]
            record["final_word_count"] = len(result["answer"].split())
            record["final_citation_metrics"] = {
                "citation_coverage": result["citation_coverage"],
                "citation_validity": result["citation_validity"],
            }
        records.append(record)

    successful = [row for row in records if "error_type" not in row]
    result = {
        "task": "D1",
        "scope": "synthetic fixtures against current production helper; no Ollama call",
        "historical_fallback_rate": 0.619048,
        "historical_rate_status": (
            "unverified aggregate; raw examples and triggers unavailable"
        ),
        "cases": records,
        "fixture_counts": {
            "total": len(records),
            "returned": len(successful),
            "fallback": sum(bool(row["used_fallback"]) for row in successful),
            "generator_errors": sum(
                row.get("error_type") is not None for row in records
            ),
            "raw_invalid_numeric_or_absent": sum(
                "numeric_citation_validity_below_one"
                in row.get("raw_trigger_reasons", [])
                for row in records
            ),
            "raw_coverage_below_half": sum(
                "long_line_coverage_below_half" in row.get("raw_trigger_reasons", [])
                for row in records
            ),
            "final_invalid_citation_metrics": sum(
                row["final_citation_metrics"]["citation_validity"] < 1.0
                or row["final_citation_metrics"]["citation_coverage"] < 0.5
                for row in successful
            ),
        },
        "empty_pipeline_error_type": _empty_pipeline_probe(),
        "below_threshold": _threshold_probe(),
        "source_truncation": _truncation_probe(),
        "provenance": {
            "rag_py_sha256": sha256_file(root / "src/codeanswer/rag.py"),
            "audit_py_sha256": sha256_file(root / "src/codeanswer/rag_gate_audit.py"),
            "historical_summary_sha256": sha256_file(
                root / "results/experiment_summary.json"
            ),
        },
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "command": " ".join(sys.argv),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
