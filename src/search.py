from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import INDEX_FILE
from .embeddings import EmbeddingMethod, extract_embedding


def load_index(index_file: Path = INDEX_FILE) -> tuple[np.ndarray, list[str], EmbeddingMethod]:
    if not index_file.exists():
        raise FileNotFoundError(
            f"Index not found at {index_file}. Run the index command first."
        )

    data = np.load(index_file, allow_pickle=False)
    vectors = data["vectors"]
    paths = data["paths"].tolist()
    if "method" in data:
        method = cast(EmbeddingMethod, str(data["method"].item()))
    else:
        method = "histogram"
    return vectors, paths, method


def search_similar(
    query_image: Path,
    images_dir: Path,
    index_file: Path = INDEX_FILE,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    vectors, paths, method = load_index(index_file)

    query_vec = extract_embedding(query_image, method=method).reshape(1, -1)

    if query_vec.shape[1] != vectors.shape[1]:
        raise ValueError(
            "Query embedding dimension does not match index vectors. "
            "Rebuild the index with the same method."
        )

    n_neighbors = min(top_k, len(paths))

    nn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
    nn.fit(vectors)
    distances, indices = nn.kneighbors(query_vec)

    results: list[tuple[str, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = float(1.0 - dist)
        image_rel_path = paths[int(idx)]
        image_abs_path = str(images_dir / image_rel_path)
        results.append((image_abs_path, similarity))

    return results


def hybrid_search_similar(
    query_image: Path,
    images_dir: Path,
    index_file_cnn: Path,
    index_file_swin: Path,
    alpha: float = 0.5,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    vectors_cnn, paths_cnn, method_cnn = load_index(index_file_cnn)
    vectors_swin, paths_swin, method_swin = load_index(index_file_swin)

    if method_cnn != "cnn_resnet50" or method_swin != "swin_tiny":
        raise ValueError("Hybrid search requires cnn_resnet50 and swin_tiny indexes.")

    if len(paths_cnn) != len(paths_swin) or set(paths_cnn) != set(paths_swin):
        raise ValueError("Indexes must contain the exact same images for hybrid search.")

    query_vec_cnn = extract_embedding(query_image, method=method_cnn).reshape(1, -1)
    query_vec_swin = extract_embedding(query_image, method=method_swin).reshape(1, -1)

    # Compute cosine similarities using O(N) dot product (vectors are L2-normalized)
    similarities_cnn = (query_vec_cnn @ vectors_cnn.T)[0]
    similarities_swin = (query_vec_swin @ vectors_swin.T)[0]

    # Convert similarities to scores mapped by path
    scores_cnn = {paths_cnn[idx]: float(similarities_cnn[idx]) for idx in range(len(paths_cnn))}
    scores_swin = {paths_swin[idx]: float(similarities_swin[idx]) for idx in range(len(paths_swin))}

    hybrid_scores: list[tuple[str, float]] = []
    for path in paths_cnn:
        score = alpha * scores_cnn[path] + (1.0 - alpha) * scores_swin[path]
        hybrid_scores.append((str(images_dir / path), float(score)))

    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    return hybrid_scores[:top_k]
