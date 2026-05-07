from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import IMAGES_DIR, INDEX_FILE, PROJECT_ROOT
from .search import search_similar

app = FastAPI(title="Image Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = PROJECT_ROOT / "src" / "static"
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
        # Using the standard search, which pulls the method from the index file metadata.
        # This keeps the hybrid or method complexity backstage.
        results = search_similar(
            query_image=tmp_path,
            images_dir=IMAGES_DIR,
            index_file=INDEX_FILE,
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

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
