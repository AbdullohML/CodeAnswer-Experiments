"""Tests for frozen known-item splits and artifact identity checks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import pytest

import scripts.evaluate_retrieval as evaluation_script
from codeanswer.evaluation_protocol import (
    freeze_known_item_splits,
    load_frozen_split,
    verify_flat_artifacts,
)
from codeanswer.provenance import sha256_file


def _corpus(path: Path, *, duplicate_id: bool = False) -> None:
    frame = pd.DataFrame(
        {
            "doc_id": range(12),
            "question_id": [100 + i for i in range(11)]
            + [100 if duplicate_id else 111],
            "title": ["same title", "same title"]
            + [f"Title {i}" for i in range(2, 12)],
            "question_body": ["body 0", "body 1", "shared body", "shared body"]
            + [f"body {i}" for i in range(4, 12)],
        }
    )
    frame.loc[10, "title"] = "NA"
    frame.loc[11, "title"] = "123"
    frame.to_parquet(path, index=False)


def test_frozen_splits_are_deterministic_and_family_disjoint(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.parquet"
    _corpus(corpus)
    first = freeze_known_item_splits(
        corpus,
        tmp_path / "first",
        query_count=12,
        seed=7,
        development_count=3,
        calibration_count=3,
    )
    second = freeze_known_item_splits(
        corpus,
        tmp_path / "second",
        query_count=12,
        seed=7,
        development_count=3,
        calibration_count=3,
    )
    assert first == second
    assert first["duplicates_in_selected_queries"]["exact_duplicate_title_rows"] == 1
    assert first["duplicates_in_selected_queries"]["exact_duplicate_body_rows"] == 1

    split_frames = [
        load_frozen_split(tmp_path / "first" / "manifest.json", corpus, split=name)[0]
        for name in ("development", "calibration", "test")
    ]
    assert sum(len(frame) for frame in split_frames) == 12
    assert all(not frame.empty for frame in split_frames)
    assert {"NA", "123"} <= set(pd.concat(split_frames)["query"])
    for left in range(3):
        for right in range(left + 1, 3):
            assert set(split_frames[left]["family_id"]).isdisjoint(
                split_frames[right]["family_id"]
            )


def test_frozen_split_rejects_tampering_and_duplicate_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.parquet"
    _corpus(corpus)
    folder = tmp_path / "frozen"
    freeze_known_item_splits(
        corpus,
        folder,
        query_count=9,
        seed=3,
        development_count=2,
        calibration_count=2,
    )
    with (folder / "test.csv").open("a", encoding="utf-8") as stream:
        stream.write("unexpected\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_frozen_split(folder / "manifest.json", corpus)

    other = tmp_path / "duplicate.parquet"
    _corpus(other, duplicate_id=True)
    with pytest.raises(ValueError, match="Duplicate question_id"):
        freeze_known_item_splits(
            other,
            tmp_path / "bad",
            query_count=9,
            seed=3,
            development_count=2,
            calibration_count=2,
        )


def test_flat_index_lineage_and_frozen_baseline_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.parquet"
    _corpus(corpus)
    vectors = np.eye(12, 2, dtype=np.float32)
    embedding = tmp_path / "embeddings.f32"
    vectors.tofile(embedding)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    index_path = tmp_path / "flat.faiss"
    faiss.write_index(index, str(index_path))
    Path(f"{embedding}.manifest.json").write_text(
        json.dumps(
            {
                "corpus_path": str(corpus),
                "output_path": str(embedding),
                "objects": 12,
                "dimension": 2,
                "model_name": "fixture",
                "max_sequence_length": 32,
            }
        ),
        encoding="utf-8",
    )
    Path(f"{index_path}.manifest.json").write_text(
        json.dumps(
            {
                "embedding_path": str(embedding),
                "index_path": str(index_path),
                "index_type": "IndexFlatIP",
                "metric": "inner_product",
                "objects": 12,
                "dimension": 2,
            }
        ),
        encoding="utf-8",
    )
    identity = verify_flat_artifacts(
        corpus,
        index_path,
        index,
        expected_model="fixture",
        expected_sequence_length=32,
    )
    assert identity["index_embedding_match"] == "all_vectors"
    assert identity["embedding_corpus_match"] == "legacy_path_only"

    frozen_dir = tmp_path / "frozen"
    freeze_known_item_splits(
        corpus,
        frozen_dir,
        query_count=12,
        seed=7,
        development_count=3,
        calibration_count=3,
    )

    class FakeQueryEncoder:
        device = "cpu"

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
            del batch_size
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    config = SimpleNamespace(
        paths=SimpleNamespace(artifacts_dir=tmp_path, results_dir=tmp_path),
        evaluation=SimpleNamespace(retrieval_queries=12, latency_queries=2),
        dataset=SimpleNamespace(target_objects=12, random_seed=7),
        embedding=SimpleNamespace(
            model_name="fixture",
            dimension=2,
            max_sequence_length=32,
            batch_size=4,
        ),
    )
    output_dir = tmp_path / "baseline"
    args = SimpleNamespace(
        config="unused",
        corpus=str(corpus),
        index=str(index_path),
        output_dir=str(output_dir),
        evaluation_manifest=str(frozen_dir / "manifest.json"),
        split="test",
        query_count=None,
        latency_queries=2,
        device="cpu",
    )
    monkeypatch.setattr(evaluation_script, "parse_arguments", lambda: args)
    monkeypatch.setattr(evaluation_script, "load_config", lambda path: config)
    monkeypatch.setattr(
        evaluation_script, "SentenceTransformerEncoder", FakeQueryEncoder
    )
    evaluation_script.main()
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    per_query = pd.read_csv(output_dir / "per_query_metrics.csv")
    assert result["quality"]["queries"] == len(per_query)
    assert result["evaluation_split"] == "test"
    assert result["artifact_identity"]["index_embedding_match"] == "all_vectors"
    assert "retrieved_doc_ids" in per_query

    changed = vectors.copy()
    changed[0] = [0.5, 0.5]
    changed.tofile(embedding)
    with pytest.raises(ValueError, match="Index vectors differ"):
        verify_flat_artifacts(
            corpus,
            index_path,
            index,
            expected_model="fixture",
            expected_sequence_length=32,
        )
    vectors.tofile(embedding)
    embedding_manifest_path = Path(f"{embedding}.manifest.json")
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    embedding_manifest["corpus_sha256"] = sha256_file(corpus)
    embedding_manifest_path.write_text(json.dumps(embedding_manifest), encoding="utf-8")
    identity = verify_flat_artifacts(
        corpus, index_path, index, expected_model="fixture", expected_sequence_length=32
    )
    assert identity["embedding_corpus_match"] == "sha256"
    changed_corpus = pd.read_parquet(corpus)
    changed_corpus.loc[0, "title"] = "Changed"
    changed_corpus.to_parquet(corpus, index=False)
    with pytest.raises(ValueError, match="corpus checksum"):
        verify_flat_artifacts(
            corpus, index_path, index, expected_model="fixture",
            expected_sequence_length=32,
        )
