"""Run D3 paired dense, BM25, and hybrid RAG with fixed D2 settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeanswer.bm25 import open_index as open_bm25_index
from codeanswer.config import load_config
from codeanswer.embeddings import SentenceTransformerEncoder
from codeanswer.indexing import load_index
from codeanswer.optimization import load_pca_model
from codeanswer.provenance import sha256_file
from codeanswer.rag import CodeAnswerPipeline, OllamaGenerator
from codeanswer.rag_comparison import pair_rows, strategy_summary, write_paired_csv
from codeanswer.rag_evaluation import (
    PrefixPCA,
    TracedGenerator,
    evaluate_row,
    load_labels,
    write_jsonl,
)
from codeanswer.rag_retrievers import STRATEGIES, CandidateRetriever
from codeanswer.reranking import CrossEncoderScorer

ROOT = Path(__file__).resolve().parents[1]


def _check_d2_identity(config: object, d2: dict, labels_manifest: dict) -> float:
    if d2["labels_sha256"] != labels_manifest["labels_sha256"]:
        raise ValueError("D2 labels changed")
    required_hashes = {
        "config_sha256": ROOT / "configs/default.yaml",
        "corpus_sha256": ROOT / "artifacts/corpus/qa_corpus.parquet",
        "pca_sha256": ROOT / "artifacts/b1/pca_384_to_256.joblib",
        "index_sha256": ROOT / "artifacts/b2/hnsw_128.faiss",
        "reranker_model_sha256": (
            ROOT / "artifacts/models/ms-marco-MiniLM-L-6-v2/model.safetensors"
        ),
    }
    for key, path in required_hashes.items():
        if sha256_file(path) != d2[key]:
            raise ValueError(f"D2 artifact or setting changed: {key}")
    expected_generator = {
        "model": config.ollama.model_name,
        "base_url": config.ollama.base_url,
        "temperature": config.ollama.temperature,
        "context_length": config.ollama.context_length,
        "seed": config.dataset.random_seed,
        "max_tokens": 320,
        "maximum_answer_words": config.ollama.maximum_answer_words,
    }
    if d2["generator"] != expected_generator:
        raise ValueError("D2 generator settings changed")
    selected = json.loads(
        (
            ROOT / "results/evaluation/d2/run_local_b1b2/selected_threshold.json"
        ).read_text(encoding="utf-8")
    )
    if (
        selected["selected_from"] != "calibration"
        or selected["labels_sha256"] != labels_manifest["labels_sha256"]
        or selected["threshold"] != d2["selected_threshold"]
    ):
        raise ValueError("D2 calibration threshold changed")
    return float(selected["threshold"])


def _evaluate(
    rows: list[dict],
    *,
    pipeline: CodeAnswerPipeline,
    generator: TracedGenerator,
    retrievers: dict[str, CandidateRetriever],
) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {name: [] for name in STRATEGIES}
    for index, row in enumerate(rows):
        order = STRATEGIES[index % 3 :] + STRATEGIES[: index % 3]
        for strategy in order:
            retriever = retrievers[strategy]
            pipeline.candidate_retriever = retriever
            result = evaluate_row(row, pipeline, generator)
            result["strategy"] = strategy
            result["candidate_doc_ids"] = retriever.last_doc_ids.copy()
            result["candidate_scores"] = retriever.last_scores.copy()
            result["candidate_count"] = len(retriever.last_doc_ids)
            result["error_stage"] = (
                "generation"
                if result["actual_behavior"] == "error"
                and result["generation_attempted"]
                else "retrieval_or_rerank"
                if result["actual_behavior"] == "error"
                else None
            )
            records[strategy].append(result)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/evaluation/d3/run_local_b1b2",
    )
    args = parser.parse_args()
    labels_path = ROOT / "results/evaluation/d2/labels.jsonl"
    labels, labels_manifest = load_labels(
        labels_path, labels_path.with_name("manifest.json")
    )
    test = [row for row in labels if row["split"] == "test"]
    if len(test) != 28:
        raise ValueError("D3 requires all 28 frozen D2 test questions")
    config = load_config(ROOT / "configs/default.yaml")
    d2_path = ROOT / "results/evaluation/d2/run_local_b1b2/run_manifest.json"
    d2 = json.loads(d2_path.read_text(encoding="utf-8"))
    threshold = _check_d2_identity(config, d2, labels_manifest)
    corpus_path = ROOT / d2["corpus_path"]
    bm25_path = ROOT / "artifacts/indexes/bm25_fts5.sqlite"
    bm25_config = ROOT / "configs/bm25.json"
    connection, bm25_manifest = open_bm25_index(bm25_path, corpus_path, bm25_config)
    try:
        encoder = SentenceTransformerEncoder(
            model_name=config.embedding.model_name,
            max_sequence_length=config.embedding.max_sequence_length,
            device=args.device,
        )
        if encoder.model_revision != d2["encoder_revision"]:
            raise ValueError("Encoder revision differs from D2")
        scorer = CrossEncoderScorer(
            model_name=str(ROOT / d2["reranker_model_path"]), device=args.device
        )
        generator = TracedGenerator(
            OllamaGenerator(
                model_name=config.ollama.model_name,
                base_url=config.ollama.base_url,
                temperature=config.ollama.temperature,
                context_length=config.ollama.context_length,
                seed=config.dataset.random_seed,
            )
        )
        pca = PrefixPCA(
            load_pca_model(ROOT / d2["pca_path"]), dimensions=d2["pca_dimensions"]
        )
        index = load_index(
            ROOT / d2["index_path"], expected_objects=520000, expected_dimension=128
        )
        retrievers = {
            name: CandidateRetriever(name, encoder, pca, index, connection)
            for name in STRATEGIES
        }
        pipeline = CodeAnswerPipeline(
            corpus_path=corpus_path,
            pca=pca,
            index=index,
            encoder=encoder,
            reranker=scorer,
            generator=generator,
            answerability_threshold=threshold,
            retrieval_candidates=40,
            final_sources=3,
            reranker_batch_size=config.reranker.batch_size,
            ef_search=config.hnsw.ef_search,
            maximum_answer_words=config.ollama.maximum_answer_words,
        )
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        main_rows = _evaluate(
            test, pipeline=pipeline, generator=generator, retrievers=retrievers
        )
        repeat_ids = {row["id"] for row in test[:4] + test[-2:]}
        repeat_rows = [row for row in test if row["id"] in repeat_ids]
        repeated = _evaluate(
            repeat_rows, pipeline=pipeline, generator=generator, retrievers=retrievers
        )
        paired = pair_rows(main_rows)
        repeat_paired = pair_rows(repeated)
        for strategy in STRATEGIES:
            write_jsonl(output / f"{strategy}.jsonl", main_rows[strategy])
            write_jsonl(output / f"{strategy}_repeat.jsonl", repeated[strategy])
        write_paired_csv(output / "paired_per_query.csv", paired)
        write_paired_csv(output / "paired_repeat.csv", repeat_paired)
        summaries = {
            "main": {
                strategy: strategy_summary(main_rows[strategy])
                for strategy in STRATEGIES
            },
            "repeat": {
                strategy: strategy_summary(repeated[strategy])
                for strategy in STRATEGIES
            },
            "repeat_query_ids": [row["id"] for row in repeat_rows],
            "repeat_purpose": (
                "descriptive sequential runtime stability; no significance test"
            ),
        }
        (output / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
        result_files = [
            "summary.json",
            "paired_per_query.csv",
            "paired_repeat.csv",
            *[f"{name}.jsonl" for name in STRATEGIES],
            *[f"{name}_repeat.jsonl" for name in STRATEGIES],
        ]
        manifest = {
            "protocol": "D3 paired RAG; candidate strategy is the sole varied setting",
            "test_labels_sha256": labels_manifest["labels_sha256"],
            "d2_manifest_sha256": sha256_file(d2_path),
            "answerability_threshold": threshold,
            "candidate_depth": 40,
            "reranked_sources": 3,
            "rrf_constant": 60,
            "bm25_index_sha256": bm25_manifest["index_sha256"],
            "bm25_config_sha256": sha256_file(bm25_config),
            "d2_shared_settings": d2,
            "execution_order": "query major, strategies rotated by query index",
            "repeat_query_ids": summaries["repeat_query_ids"],
            "files_sha256": {name: sha256_file(output / name) for name in result_files},
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"D3 compared {len(test)} paired questions; results: {output}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
