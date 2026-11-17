"""Run D1 synthetic citation-gate and fallback probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from codeanswer.rag_gate_audit import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation/d1/fixture_audit.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(root, args.output)
    print(f"D1 {result['fixture_counts']['total']} fixtures: {args.output}")


if __name__ == "__main__":
    main()
