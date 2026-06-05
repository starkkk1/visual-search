from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SearchTiming:
    total_ms: float = 0.0
    upload_read_ms: float = 0.0
    temp_write_ms: float = 0.0
    embedding_ms: float = 0.0
    qdrant_query_ms: float = 0.0
    result_format_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "total_ms": round(self.total_ms, 2),
            "upload_read_ms": round(self.upload_read_ms, 2),
            "temp_write_ms": round(self.temp_write_ms, 2),
            "embedding_ms": round(self.embedding_ms, 2),
            "qdrant_query_ms": round(self.qdrant_query_ms, 2),
            "result_format_ms": round(self.result_format_ms, 2),
        }

    def as_server_timing(self) -> str:
        return ", ".join(
            [
                f"upload_read;dur={self.upload_read_ms:.2f}",
                f"temp_write;dur={self.temp_write_ms:.2f}",
                f"embedding;dur={self.embedding_ms:.2f}",
                f"qdrant_query;dur={self.qdrant_query_ms:.2f}",
                f"result_format;dur={self.result_format_ms:.2f}",
                f"total;dur={self.total_ms:.2f}",
            ]
        )
