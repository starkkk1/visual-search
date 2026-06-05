from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient

from .config import QDRANT_API_KEY, QDRANT_PATH, QDRANT_URL


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create one Qdrant client per process.

    In local development we can still use embedded storage via `QDRANT_PATH`.
    In containers and production we prefer `QDRANT_URL` so Qdrant runs as a
    separate service.
    """
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(path=str(QDRANT_PATH))
