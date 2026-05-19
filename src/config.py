from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("IMAGE_SEARCH_DATA_DIR", PROJECT_ROOT.parent / "data"))
IMAGES_DIR = Path(os.getenv("IMAGE_SEARCH_IMAGES_DIR", DATA_DIR / "images"))
QDRANT_PATH = Path(os.getenv("IMAGE_SEARCH_QDRANT_PATH", DATA_DIR / "qdrant"))

DEFAULT_COLLECTION = os.getenv("IMAGE_SEARCH_DEFAULT_COLLECTION", "histogram")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
