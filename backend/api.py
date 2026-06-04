from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import DEFAULT_COLLECTION, IMAGES_DIR, PROJECT_ROOT
from .search import search_similar, search_similar_by_text

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
async def search(image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Using the standard search, which targets the DEFAULT_COLLECTION
        results = search_similar(
            query_image=tmp_path,
            images_dir=IMAGES_DIR,
            collection_name=DEFAULT_COLLECTION,
            top_k=20,
        )
        
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
        return formatted_results
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 20


@app.post("/search-text", response_model=list[SearchResult])
async def search_text(request: TextSearchRequest):
    try:
        # We search with DEFAULT_COLLECTION (which can be overriden via ENV)
        # Note: search_similar_by_text will raise a ValueError if collection is not 'clip'
        results = search_similar_by_text(
            query_text=request.query,
            images_dir=IMAGES_DIR,
            collection_name=DEFAULT_COLLECTION,
            top_k=request.top_k,
        )
        
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
        return formatted_results
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
