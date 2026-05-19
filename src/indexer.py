from __future__ import annotations

import math
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from .config import QDRANT_PATH, SUPPORTED_EXTENSIONS
from .embeddings import EmbeddingMethod, extract_embeddings_batch, get_embedding_size


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def build_index(
    images_dir: Path,
    collection_name: str = "histogram",
    method: EmbeddingMethod = "histogram",
    batch_size: int = 32,
) -> int:
    image_paths = sorted(path for path in images_dir.rglob("*") if _is_image(path))
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    embedding_size = get_embedding_size(method)
    
    # Initialize Qdrant Client in local mode
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    
    # Recreate collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )

    num_batches = math.ceil(len(image_paths) / batch_size)
    total_indexed = 0
    
    for b in tqdm(range(num_batches), desc=f"Indexing ({method}) into Qdrant"):
        batch_paths = image_paths[b * batch_size : (b + 1) * batch_size]
        batch_vectors, valid_paths = extract_embeddings_batch(batch_paths, method=method)
        
        if len(valid_paths) > 0:
            points = []
            for vec, p in zip(batch_vectors, valid_paths):
                rel_path = str(p.relative_to(images_dir))
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, rel_path))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vec.tolist(),
                        payload={"path": rel_path},
                    )
                )
            
            client.upload_points(
                collection_name=collection_name,
                points=points,
            )
            total_indexed += len(points)

    if total_indexed == 0:
        raise ValueError("No valid images could be processed.")

    return total_indexed
