"""Measure B4 build, load, memory, and retrieval-stage resource costs."""

from __future__ import annotations

import argparse
from pathlib import Path

from codeanswer.resource_profile import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pca", "build", "runtime"))
    parser.add_argument("--scope", choices=("full", "subset"), default="full")
    parser.add_argument("--system", choices=("flat384", "flat128", "hnsw128"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/evaluation/b4")
    )
    args = parser.parse_args()
    if (args.phase == "pca") != (args.system is None):
        parser.error("--system is required for build/runtime and omitted for pca")
    if args.phase == "runtime" and args.scope != "full":
        parser.error("runtime profiles the existing full indexes only")
    result = run(
        Path(__file__).resolve().parents[1],
        args.output_dir,
        phase=args.phase,
        scope=args.scope,
        system=args.system,
    )
    print(f"B4 {args.phase}/{args.scope}/{args.system or 'pca'}: {result['output']}")


if __name__ == "__main__":
    main()
