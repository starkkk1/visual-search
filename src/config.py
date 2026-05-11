from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("IMAGE_SEARCH_DATA_DIR", PROJECT_ROOT.parent / "data"))
IMAGES_DIR = Path(os.getenv("IMAGE_SEARCH_IMAGES_DIR", DATA_DIR / "images"))
INDEX_FILE = Path(os.getenv("IMAGE_SEARCH_INDEX_FILE", DATA_DIR / "index" / "image_index.npz"))

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
