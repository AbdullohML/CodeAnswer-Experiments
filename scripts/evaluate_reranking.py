"""Run the A4 paired cross-encoder ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

from codeanswer.reranking_ablation import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/models/ms-marco-MiniLM-L-6-v2"),
    )
    parser.add_argument(
        "--settings", type=Path, default=Path("configs/reranking_ablation.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/evaluation/a4_test")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(
        root,
        args.split,
        args.output_dir,
        args.model_path,
        args.settings,
        args.smoke_limit,
        args.device,
    )
    print(f"A4 {result['query_count']} queries; results: {args.output_dir}")


if __name__ == "__main__":
    main()
