from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import DEFAULT_COLLECTION, IMAGES_DIR, LOG_SEARCH_TIMING, PROJECT_ROOT
from .search import search_similar
from .timing import SearchTiming

app = FastAPI(title="Image Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = PROJECT_ROOT / "backend" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

class SearchResult(BaseModel):
    path: str
    url: str
    score: float

@app.post("/search", response_model=list[SearchResult])
async def search(response: Response, image: UploadFile = File(...)):
    request_started_at = perf_counter()
    read_started_at = perf_counter()
    content = await image.read()
    upload_read_ms = (perf_counter() - read_started_at) * 1000

    write_started_at = perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    temp_write_ms = (perf_counter() - write_started_at) * 1000

    try:
        # Using the standard search, which targets the DEFAULT_COLLECTION
        results, timing = search_similar(
            query_image=tmp_path,
            images_dir=IMAGES_DIR,
            collection_name=DEFAULT_COLLECTION,
            top_k=20,
        )
        
        format_started_at = perf_counter()
        formatted_results = []
        for abs_path, score in results:
            rel_path = Path(abs_path).relative_to(IMAGES_DIR)
            formatted_results.append(
                SearchResult(
                    path=str(rel_path),
                    url=f"/images/{str(rel_path).replace(os.sep, '/')}",
                    score=score,
                )
            )
        result_format_ms = (perf_counter() - format_started_at) * 1000
        full_timing = SearchTiming(
            total_ms=(perf_counter() - request_started_at) * 1000,
            upload_read_ms=upload_read_ms,
            temp_write_ms=temp_write_ms,
            embedding_ms=timing.embedding_ms,
            qdrant_query_ms=timing.qdrant_query_ms,
            result_format_ms=result_format_ms,
        )
        response.headers["Server-Timing"] = full_timing.as_server_timing()
        response.headers["X-Search-Timing"] = json.dumps(full_timing.as_dict())
        if LOG_SEARCH_TIMING:
            print(f"Search timing: {full_timing.as_dict()}")
        return formatted_results
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
