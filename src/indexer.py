from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import INDEX_FILE, SUPPORTED_EXTENSIONS
from .embeddings import EmbeddingMethod, extract_embedding, extract_embeddings_batch, get_embedding_size


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def build_index(
    images_dir: Path,
    index_file: Path = INDEX_FILE,
    method: EmbeddingMethod = "histogram",
    batch_size: int = 32,
) -> tuple[int, Path]:
    image_paths = sorted(path for path in images_dir.rglob("*") if _is_image(path))
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    embedding_size = get_embedding_size(method)
    all_vectors = []
    rel_paths: list[str] = []

    num_batches = math.ceil(len(image_paths) / batch_size)
    for b in tqdm(range(num_batches), desc=f"Indexing ({method})"):
        batch_paths = image_paths[b * batch_size : (b + 1) * batch_size]
        batch_vectors, valid_paths = extract_embeddings_batch(batch_paths, method=method)
        
        if len(valid_paths) > 0:
            all_vectors.append(batch_vectors)
            for p in valid_paths:
                rel_paths.append(str(p.relative_to(images_dir)))

    if not all_vectors:
        raise ValueError("No valid images could be processed.")
        
    vectors = np.vstack(all_vectors)

    index_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_file,
        vectors=vectors,
        paths=np.array(rel_paths, dtype=str),
        method=np.array(method, dtype=str),
    )

    return len(rel_paths), index_file
