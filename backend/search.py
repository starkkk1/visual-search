from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import cast

from .embeddings import EmbeddingMethod, extract_embedding
from .qdrant import get_qdrant_client
from .timing import SearchTiming


def search_similar(
    query_image: Path,
    images_dir: Path,
    collection_name: str = "histogram",
    top_k: int = 5,
) -> tuple[list[tuple[str, float]], SearchTiming]:
    client = get_qdrant_client()
    
    # We use the collection_name to determine the embedding method
    method = cast(EmbeddingMethod, collection_name)
    embedding_started_at = perf_counter()
    query_vec = extract_embedding(query_image, method=method)
    embedding_ms = (perf_counter() - embedding_started_at) * 1000

    qdrant_started_at = perf_counter()
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vec.tolist(),
        limit=top_k,
    )
    qdrant_query_ms = (perf_counter() - qdrant_started_at) * 1000

    results: list[tuple[str, float]] = []
    for point in search_result.points:
        if point.payload and "path" in point.payload:
            image_rel_path = point.payload["path"]
            image_abs_path = str(images_dir / image_rel_path)
            results.append((image_abs_path, float(point.score)))

    timing = SearchTiming(
        embedding_ms=embedding_ms,
        qdrant_query_ms=qdrant_query_ms,
    )
    return results, timing
