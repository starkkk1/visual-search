from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from qdrant_client import QdrantClient

from .config import QDRANT_PATH
from .embeddings import EmbeddingMethod, extract_embedding


def search_similar(
    query_image: Path,
    images_dir: Path,
    collection_name: str = "histogram",
    top_k: int = 5,
) -> list[tuple[str, float]]:
    client = QdrantClient(path=str(QDRANT_PATH))
    
    # We use the collection_name to determine the embedding method
    method = cast(EmbeddingMethod, collection_name)
    query_vec = extract_embedding(query_image, method=method)

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vec.tolist(),
        limit=top_k,
    )

    results: list[tuple[str, float]] = []
    for point in search_result:
        if point.payload and "path" in point.payload:
            image_rel_path = point.payload["path"]
            image_abs_path = str(images_dir / image_rel_path)
            results.append((image_abs_path, float(point.score)))

    return results


def hybrid_search_similar(
    query_image: Path,
    images_dir: Path,
    alpha: float = 0.5,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    client = QdrantClient(path=str(QDRANT_PATH))
    
    # Query CNN
    query_vec_cnn = extract_embedding(query_image, method="cnn_resnet50")
    # Fetch a large number of top results to combine
    search_result_cnn = client.search(
        collection_name="cnn_resnet50",
        query_vector=query_vec_cnn.tolist(),
        limit=1000,
    )
    
    # Query Swin
    query_vec_swin = extract_embedding(query_image, method="swin_tiny")
    search_result_swin = client.search(
        collection_name="swin_tiny",
        query_vector=query_vec_swin.tolist(),
        limit=1000,
    )

    scores_cnn = {point.payload["path"]: float(point.score) for point in search_result_cnn if point.payload}
    scores_swin = {point.payload["path"]: float(point.score) for point in search_result_swin if point.payload}
    
    all_paths = set(scores_cnn.keys()).union(set(scores_swin.keys()))
    
    hybrid_scores: list[tuple[str, float]] = []
    for path in all_paths:
        score_cnn = scores_cnn.get(path, 0.0)
        score_swin = scores_swin.get(path, 0.0)
        score = alpha * score_cnn + (1.0 - alpha) * score_swin
        hybrid_scores.append((str(images_dir / path), float(score)))

    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    return hybrid_scores[:top_k]
