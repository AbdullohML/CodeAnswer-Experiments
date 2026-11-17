"""Run the complete B4 resource profile in isolated, sequential processes."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/evaluation/b4"
    output.mkdir(parents=True, exist_ok=True)
    plan = [("pca", "subset", None), ("pca", "full", None)]
    for scope in ("subset", "full"):
        for system in ("flat384", "flat128", "hnsw128"):
            plan.append(("build", scope, system))
    for system in ("flat384", "flat128", "hnsw128"):
        plan.append(("runtime", "full", system))

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(root / "src"),
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    for phase, scope, system in plan:
        command = [
            sys.executable,
            str(root / "scripts/profile_resources.py"),
            phase,
            "--scope",
            scope,
            "--output-dir",
            str(output),
        ]
        if system:
            command.extend(("--system", system))
        print(f"Running {phase}/{scope}/{system or 'pca'}", flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)

    rows = []
    for phase, scope, system in plan:
        name = f"{phase}_{scope}_{system or 'pca'}.json"
        result = json.loads((output / name).read_text())
        details = result["details"]
        row = {
            "phase": phase,
            "scope": scope,
            "system": system or "pca",
            "rows": details["rows"],
            "peak_rss_mib": details["memory"]["rss_peak_mib"],
            "peak_gpu_vram_mib": details["gpu_peak_vram_mib"],
            "build_or_fit_s": details.get(
                "build_seconds_add_only", details.get("fit_seconds")
            ),
            "index_file_mib": (
                round(details["file_bytes"] / 1024**2, 3)
                if "file_bytes" in details
                else None
            ),
            "first_load_s": details.get("first_load_seconds"),
            "second_load_s": details.get("second_load_seconds"),
            "search_p50_ms": None,
            "search_p95_ms": None,
            "fetch_p50_ms": None,
            "fetch_p95_ms": None,
        }
        if phase == "runtime":
            first_round = details["warm_rounds"][0]
            row.update(
                {
                    "search_p50_ms": first_round["search"]["p50_ms"],
                    "search_p95_ms": first_round["search"]["p95_ms"],
                    "fetch_p50_ms": first_round["fetch40"]["p50_ms"],
                    "fetch_p95_ms": first_round["fetch40"]["p95_ms"],
                }
            )
        rows.append(row)
    with (output / "resource_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": "B4 sequential isolated-process resource profile",
                "plan": [
                    {"phase": phase, "scope": scope, "system": system}
                    for phase, scope, system in plan
                ],
                "hardware_and_settings": "embedded in each phase JSON",
                "table": "resource_table.csv",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"B4 complete: {output / 'resource_table.csv'}")


if __name__ == "__main__":
    main()
