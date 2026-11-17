"""Traceable behavior evaluation for the local CodeAnswer RAG pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from codeanswer.provenance import sha256_file
from codeanswer.reranking import calibrate_answerability_threshold


@dataclass
class PrefixPCA:
    """Use the first N components of the saved B1 PCA basis."""

    base: Any
    dimensions: int

    @property
    def n_features_in_(self) -> int:
        return int(self.base.n_features_in_)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.base.transform(values)[:, : self.dimensions]


def load_labels(path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["labels_sha256"]:
        raise ValueError("Labels do not match the frozen D2 manifest")
    labels = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected_counts = manifest["counts"]
    ids: set[str] = set()
    families: dict[int, str] = {}
    for row in labels:
        split = row["split"]
        behavior = row["expected_behavior"]
        if split not in expected_counts or behavior not in {"answer", "abstain"}:
            raise ValueError("Invalid split or behavior label")
        if row["id"] in ids:
            raise ValueError(f"Duplicate query ID: {row['id']}")
        ids.add(row["id"])
        family = row.get("family_id")
        if family is not None and family in families and families[family] != split:
            raise ValueError("C2 family crosses D2 splits")
        if family is not None:
            families[family] = split
    for split, counts in expected_counts.items():
        for behavior, count in counts.items():
            if (
                sum(
                    row["split"] == split and row["expected_behavior"] == behavior
                    for row in labels
                )
                != count
            ):
                raise ValueError(f"Incorrect {split} {behavior} count")
    return labels, manifest


def calibrate(rows: list[dict], pipeline: Any) -> tuple[list[dict], dict]:
    """Select a threshold using calibration labels and scores only."""
    calibration = [row for row in rows if row["split"] == "calibration"]
    records = []
    for row in calibration:
        started = time.perf_counter()
        sources, timing = pipeline.retrieve_and_rerank(row["query"])
        if sources.empty:
            raise ValueError(f"No calibration sources for {row['id']}")
        records.append(
            {
                **row,
                "top_reranker_score": float(sources.iloc[0]["reranker_score"]),
                "sources": sources.to_dict(orient="records"),
                "timings_ms": {
                    **timing,
                    "calibration_total": round(
                        (time.perf_counter() - started) * 1000, 3
                    ),
                },
            }
        )
    labels = np.array(
        [int(row["expected_behavior"] == "answer") for row in records],
        dtype=np.int64,
    )
    scores = np.array([row["top_reranker_score"] for row in records], dtype=np.float64)
    selected = calibrate_answerability_threshold(labels, scores)
    return records, selected


@dataclass
class TracedGenerator:
    """Capture exact prompts and attempted generation for each query."""

    base: Any
    system_prompt: str | None = None
    user_prompt: str | None = None
    attempted: bool = False

    @property
    def model_name(self) -> str:
        return str(self.base.model_name)

    def generate(
        self, *, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> dict[str, Any]:
        self.attempted = True
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.base.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

    def clear(self) -> None:
        self.attempted = False
        self.system_prompt = None
        self.user_prompt = None


def evaluate_row(row: dict, pipeline: Any, generator: TracedGenerator) -> dict:
    """Evaluate one labeled question; keep errors visible as failed behavior."""
    generator.clear()
    started = time.perf_counter()
    record: dict[str, Any] = {
        **row,
        "answerability_threshold": float(pipeline.answerability_threshold),
    }
    try:
        result = pipeline.answer(row["query"])
    except Exception as exc:
        record.update(
            {
                "actual_behavior": "error",
                "behavior_correct": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "wall_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
    else:
        actual = "abstain" if result["abstained"] else "answer"
        record.update(
            {
                "actual_behavior": actual,
                "behavior_correct": actual == row["expected_behavior"],
                "error": None,
                "wall_ms": round((time.perf_counter() - started) * 1000, 3),
                **result,
            }
        )
        # The pipeline returns the same query; retain the frozen label copy.
        record["query"] = row["query"]
    record["generation_attempted"] = generator.attempted
    record["system_prompt"] = generator.system_prompt
    record["user_prompt"] = generator.user_prompt
    return record


def _percentiles(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate only outcomes traceable to the supplied per-query records."""
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    total = len(records)
    in_domain = [row for row in records if row["expected_behavior"] == "answer"]
    out_domain = [row for row in records if row["expected_behavior"] == "abstain"]
    answered = [row for row in records if row["actual_behavior"] == "answer"]
    returned = [row for row in records if row["actual_behavior"] != "error"]
    timing_keys = (
        "total_ms",
        "retrieval_ms",
        "encode_ms",
        "projection_ms",
        "search_ms",
        "fetch_ms",
        "rerank_model_ms",
        "generation_ms",
        "gate_ms",
    )
    return {
        "query_count": total,
        "behavior_accuracy": sum(row["behavior_correct"] for row in records) / total,
        "in_domain_answer_rate": (
            sum(row["actual_behavior"] == "answer" for row in in_domain)
            / len(in_domain)
            if in_domain
            else None
        ),
        "out_of_domain_abstention_rate": (
            sum(row["actual_behavior"] == "abstain" for row in out_domain)
            / len(out_domain)
            if out_domain
            else None
        ),
        "fallback_rate_among_answers": (
            sum(bool(row["used_fallback"]) for row in answered) / len(answered)
            if answered
            else None
        ),
        "error_rate": (total - len(returned)) / total,
        "counts": {
            "answer": len(answered),
            "abstain": total - len(answered) - (total - len(returned)),
            "error": total - len(returned),
            "fallback": sum(bool(row["used_fallback"]) for row in answered),
            "fallback_invalid_number": sum(
                "numeric_citation_validity_below_one" in row.get("fallback_reasons", [])
                for row in answered
            ),
            "fallback_low_coverage": sum(
                "long_line_coverage_below_half" in row.get("fallback_reasons", [])
                for row in answered
            ),
            "in_domain": len(in_domain),
            "out_of_domain": len(out_domain),
        },
        "latency_ms": {
            key: _percentiles([float(row[key]) for row in returned if key in row])
            for key in timing_keys
        },
        "latency_boundary": (
            "Per-query completed pipeline calls; errors excluded; "
            "total includes retrieval, fetch, rerank, gate, and generation"
        ),
        "behavior_definition": (
            "answer/abstain agreement with domain-control labels; "
            "does not measure answer correctness or source support"
        ),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")
