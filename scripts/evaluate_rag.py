"""Run D2 calibration and local-Qwen RAG behavior evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeanswer.config import load_config
from codeanswer.embeddings import SentenceTransformerEncoder
from codeanswer.indexing import load_index
from codeanswer.optimization import load_pca_model
from codeanswer.provenance import sha256_file
from codeanswer.rag import CodeAnswerPipeline, OllamaGenerator
from codeanswer.rag_evaluation import (
    PrefixPCA,
    TracedGenerator,
    calibrate,
    evaluate_row,
    load_labels,
    summarize,
    write_jsonl,
)
from codeanswer.reranking import CrossEncoderScorer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "results/evaluation/d2/labels.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/evaluation/d2/run_local_b1b2",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    config = load_config(ROOT / "configs/default.yaml")
    labels, label_manifest = load_labels(
        args.labels, args.labels.with_name("manifest.json")
    )
    for split, expected_sha in label_manifest["source_c2_csv_sha256"].items():
        source = ROOT / "artifacts/evaluation/frozen_v1" / f"{split}.csv"
        if sha256_file(source) != expected_sha:
            raise ValueError(f"C2 {split} source changed after label preparation")

    pca_path = ROOT / "artifacts/b1/pca_384_to_256.joblib"
    index_path = ROOT / "artifacts/b2/hnsw_128.faiss"
    reranker_path = ROOT / "artifacts/models/ms-marco-MiniLM-L-6-v2"
    corpus_path = ROOT / "artifacts/corpus/qa_corpus.parquet"
    encoder = SentenceTransformerEncoder(
        model_name=config.embedding.model_name,
        max_sequence_length=config.embedding.max_sequence_length,
        device=args.device,
    )
    scorer = CrossEncoderScorer(model_name=str(reranker_path), device=args.device)
    generator = TracedGenerator(
        OllamaGenerator(
            model_name=config.ollama.model_name,
            base_url=config.ollama.base_url,
            temperature=config.ollama.temperature,
            context_length=config.ollama.context_length,
            seed=config.dataset.random_seed,
        )
    )
    pipeline = CodeAnswerPipeline(
        corpus_path=corpus_path,
        pca=PrefixPCA(load_pca_model(pca_path), dimensions=128),
        index=load_index(index_path, expected_objects=520000, expected_dimension=128),
        encoder=encoder,
        reranker=scorer,
        generator=generator,
        answerability_threshold=config.reranker.answerability_threshold,
        retrieval_candidates=config.reranker.retrieval_candidates,
        final_sources=config.reranker.final_sources,
        reranker_batch_size=config.reranker.batch_size,
        ef_search=config.hnsw.ef_search,
        maximum_answer_words=config.ollama.maximum_answer_words,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    calibration_rows, selected = calibrate(labels, pipeline)
    write_jsonl(output / "calibration.jsonl", calibration_rows)
    (output / "selected_threshold.json").write_text(
        json.dumps(
            {
                **selected,
                "selected_from": "calibration",
                "calibration_query_count": len(calibration_rows),
                "labels_sha256": label_manifest["labels_sha256"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pipeline.answerability_threshold = selected["threshold"]
    summaries = {}
    for split in ("development", "test"):
        records = [
            evaluate_row(row, pipeline, generator)
            for row in labels
            if row["split"] == split
        ]
        write_jsonl(output / f"{split}.jsonl", records)
        summaries[split] = summarize(records)
        print(
            f"{split}: {len(records)} queries, "
            f"behavior accuracy {summaries[split]['behavior_accuracy']:.3f}"
        )

    settings = {
        "protocol": "D2 new local run; historical RAG aggregates are unverified",
        "config_sha256": sha256_file(ROOT / "configs/default.yaml"),
        "pipeline_source_sha256": sha256_file(ROOT / "src/codeanswer/rag.py"),
        "c2_manifest_sha256": label_manifest["source_c2_manifest_sha256"],
        "labels_sha256": label_manifest["labels_sha256"],
        "calibration_source": "calibration only",
        "selected_threshold": selected["threshold"],
        "corpus_path": str(corpus_path.relative_to(ROOT)),
        "corpus_sha256": sha256_file(corpus_path),
        "pca_path": str(pca_path.relative_to(ROOT)),
        "pca_sha256": sha256_file(pca_path),
        "pca_dimensions": 128,
        "index_path": str(index_path.relative_to(ROOT)),
        "index_sha256": sha256_file(index_path),
        "encoder_model": config.embedding.model_name,
        "encoder_revision": encoder.model_revision,
        "reranker_model_path": str(reranker_path.relative_to(ROOT)),
        "reranker_model_sha256": sha256_file(reranker_path / "model.safetensors"),
        "generator": {
            "model": generator.model_name,
            "base_url": config.ollama.base_url,
            "temperature": config.ollama.temperature,
            "context_length": config.ollama.context_length,
            "seed": config.dataset.random_seed,
            "max_tokens": 320,
            "maximum_answer_words": config.ollama.maximum_answer_words,
        },
        "retrieval_candidates": config.reranker.retrieval_candidates,
        "final_sources": config.reranker.final_sources,
        "ef_search": config.hnsw.ef_search,
        "device": args.device,
        "files_sha256": {
            name: sha256_file(output / name)
            for name in (
                "calibration.jsonl",
                "selected_threshold.json",
                "development.jsonl",
                "test.jsonl",
            )
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved D2 traces and summary to {output}")


if __name__ == "__main__":
    main()
