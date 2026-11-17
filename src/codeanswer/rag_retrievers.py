"""Candidate adapters for paired dense, BM25, and hybrid RAG runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from codeanswer.bm25 import search as search_bm25
from codeanswer.fusion import fuse_rrf
from codeanswer.indexing import search_index
from codeanswer.optimization import transform_queries

STRATEGIES = ("dense", "bm25", "hybrid")


@dataclass
class CandidateRetriever:
    """Change only the top-40 candidate selection before shared reranking."""

    strategy: str
    encoder: Any
    pca: Any
    dense_index: Any
    bm25_connection: Any
    rrf_constant: int = 60
    last_doc_ids: list[int] = field(default_factory=list, init=False)
    last_scores: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown candidate strategy: {self.strategy}")
        if self.rrf_constant <= 0:
            raise ValueError("RRF constant must be positive")

    def retrieve(
        self, query: str, depth: int
    ) -> tuple[list[int], list[float], dict[str, float]]:
        if depth <= 0:
            raise ValueError("Candidate depth must be positive")
        self.last_doc_ids = []
        self.last_scores = []
        started = time.perf_counter()
        encode_ms = projection_ms = dense_search_ms = bm25_search_ms = fusion_ms = 0.0
        dense_ids: list[int] = []
        dense_scores: list[float] = []
        lexical_ids: list[int] = []
        lexical_scores: list[float] = []
        if self.strategy in {"dense", "hybrid"}:
            encoded = self.encoder.encode([query], batch_size=1)
            encoded_at = time.perf_counter()
            projected = transform_queries(self.pca, encoded)
            projected_at = time.perf_counter()
            values, ids = search_index(self.dense_index, projected, top_k=depth)
            searched_at = time.perf_counter()
            dense_ids = [int(value) for value in ids[0] if value >= 0]
            dense_scores = [float(value) for value in values[0][: len(dense_ids)]]
            encode_ms = (encoded_at - started) * 1000
            projection_ms = (projected_at - encoded_at) * 1000
            dense_search_ms = (searched_at - projected_at) * 1000
        if self.strategy in {"bm25", "hybrid"}:
            lexical_started_at = time.perf_counter()
            hits = search_bm25(self.bm25_connection, query, depth)
            lexical_finished_at = time.perf_counter()
            lexical_ids = [doc_id for doc_id, _ in hits]
            lexical_scores = [rank for _, rank in hits]
            bm25_search_ms = (lexical_finished_at - lexical_started_at) * 1000
        if self.strategy == "dense":
            selected_ids, selected_scores = dense_ids, dense_scores
        elif self.strategy == "bm25":
            selected_ids, selected_scores = lexical_ids, lexical_scores
        else:
            fusion_started_at = time.perf_counter()
            selected_ids, selected_scores = fuse_rrf(
                dense_ids + [-1] * (depth - len(dense_ids)),
                lexical_ids + [-1] * (depth - len(lexical_ids)),
                candidate_depth=depth,
                output_depth=depth,
                rrf_constant=self.rrf_constant,
            )
            selected_ids = [doc_id for doc_id in selected_ids if doc_id >= 0]
            fusion_ms = (time.perf_counter() - fusion_started_at) * 1000
        if len(selected_ids) != len(selected_scores):
            raise ValueError("Candidate IDs and scores differ in length")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("Candidate list contains duplicate IDs")
        self.last_doc_ids = selected_ids
        self.last_scores = selected_scores
        total_ms = (time.perf_counter() - started) * 1000
        return (
            selected_ids,
            selected_scores,
            {
                "retrieval_ms": round(total_ms, 3),
                "encode_ms": round(encode_ms, 3),
                "projection_ms": round(projection_ms, 3),
                "dense_search_ms": round(dense_search_ms, 3),
                "bm25_search_ms": round(bm25_search_ms, 3),
                "fusion_ms": round(fusion_ms, 3),
                "search_ms": round(dense_search_ms + bm25_search_ms + fusion_ms, 3),
            },
        )
