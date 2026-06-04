from __future__ import annotations

from pathlib import Path
from typing import cast

from qdrant_client import QdrantClient

from .config import QDRANT_PATH
from .embeddings import EmbeddingMethod, extract_embedding, extract_clip_text_embedding


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

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vec.tolist(),
        limit=top_k,
    )

    results: list[tuple[str, float]] = []
    for point in search_result.points:
        if point.payload and "path" in point.payload:
            image_rel_path = point.payload["path"]
            image_abs_path = str(images_dir / image_rel_path)
            results.append((image_abs_path, float(point.score)))

    return results


def search_similar_by_text(
    query_text: str,
    images_dir: Path,
    collection_name: str = "clip",
    top_k: int = 5,
) -> list[tuple[str, float]]:
    if collection_name != "clip":
        raise ValueError("Text search is only supported with the 'clip' collection.")

    client = QdrantClient(path=str(QDRANT_PATH))
    query_vec = extract_clip_text_embedding(query_text)

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vec.tolist(),
        limit=top_k,
    )

    results: list[tuple[str, float]] = []
    for point in search_result.points:
        if point.payload and "path" in point.payload:
            image_rel_path = point.payload["path"]
            image_abs_path = str(images_dir / image_rel_path)
            results.append((image_abs_path, float(point.score)))

    return results

