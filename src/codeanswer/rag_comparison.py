"""Paired RAG comparison summaries from saved per-query records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from codeanswer.rag_evaluation import summarize
from codeanswer.rag_retrievers import STRATEGIES


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
    }


def strategy_summary(rows: list[dict]) -> dict:
    summary = summarize(rows)
    answers = [row for row in rows if row["actual_behavior"] == "answer"]
    errors = [row for row in rows if row["actual_behavior"] == "error"]
    returned = [row for row in rows if row["actual_behavior"] != "error"]
    summary["counts"].update(
        {
            "generator_errors": sum(row["generation_attempted"] for row in errors),
            "pre_generation_errors": sum(
                not row["generation_attempted"] for row in errors
            ),
        }
    )
    summary["citation_format"] = {
        key: (
            round(sum(float(row[key]) for row in answers) / len(answers), 6)
            if answers
            else None
        )
        for key in (
            "raw_citation_coverage",
            "raw_citation_validity",
            "citation_coverage",
            "citation_validity",
        )
    }
    summary["tokens"] = {
        "answered_queries": len(answers),
        "prompt_total": sum(int(row.get("prompt_tokens", 0)) for row in answers),
        "output_total": sum(int(row.get("output_tokens", 0)) for row in answers),
        "prompt_distribution": _distribution(
            [int(row.get("prompt_tokens", 0)) for row in answers]
        ),
        "output_distribution": _distribution(
            [int(row.get("output_tokens", 0)) for row in answers]
        ),
    }
    summary["candidate_counts"] = {
        "min": min((row["candidate_count"] for row in rows), default=0),
        "max": max((row["candidate_count"] for row in rows), default=0),
    }
    for key in ("dense_search_ms", "bm25_search_ms", "fusion_ms"):
        summary["latency_ms"][key] = _distribution(
            [float(row[key]) for row in returned if key in row]
        )
    return summary


def pair_rows(records: dict[str, list[dict]]) -> list[dict]:
    """Align all strategies by the same fixed query IDs and labels."""
    if set(records) != set(STRATEGIES):
        raise ValueError("All three candidate strategies are required")
    first = records[STRATEGIES[0]]
    identity = [(row["id"], row["query"], row["expected_behavior"]) for row in first]
    if len(identity) != len(set(item[0] for item in identity)):
        raise ValueError("Duplicate query ID in paired run")
    for strategy in STRATEGIES[1:]:
        current = [
            (row["id"], row["query"], row["expected_behavior"])
            for row in records[strategy]
        ]
        if current != identity:
            raise ValueError(f"Paired query order or labels differ for {strategy}")
    paired = []
    for index, (identifier, query, expected) in enumerate(identity):
        row = {"id": identifier, "query": query, "expected_behavior": expected}
        for strategy in STRATEGIES:
            result = records[strategy][index]
            row[f"{strategy}_behavior"] = result["actual_behavior"]
            row[f"{strategy}_fallback"] = result.get("used_fallback")
            row[f"{strategy}_total_ms"] = result.get("total_ms")
            row[f"{strategy}_raw_answer"] = result.get("raw_answer")
            row[f"{strategy}_final_answer"] = result.get("answer")
            row[f"{strategy}_source_doc_ids"] = [
                source["doc_id"] for source in result.get("sources", [])
            ]
        paired.append(row)
    return paired


def write_paired_csv(path: Path, paired: list[dict]) -> None:
    if not paired:
        raise ValueError("No paired rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        for row in paired:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, list)
                    else value
                    for key, value in row.items()
                }
            )
