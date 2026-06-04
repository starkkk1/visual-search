# Image Search System

A visual search engine that lets you index a folder of images into vector embeddings and search for visually similar images using an interactive, modern web interface.

It supports two embedding modes:
- `histogram` (fast baseline)
- `clip` (OpenAI CLIP ViT-B/32 features)

## Project Structure

```
visual-search/
  frontend/          # Next.js React web application
  backend/           # FastAPI backend and Python AI logic
  data/              # (Optional) Place your images in data/images
```

## Setup

1. Activate your Python virtual environment.
2. Install the backend dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Add images
Put your dataset images into the `data/images` directory, or specify your image path by creating a `.env` file in the root directory:
```env
IMAGE_SEARCH_IMAGES_DIR=D:\Path\To\Your\Images
```

### 2. Build the Index (Qdrant)
Run the indexer to process all images and load them into the local Qdrant vector database:

```bash
python -m backend.cli index
```

To use the **CLIP** model instead (downloads weights on first run):
```bash
python -m backend.cli index --method clip
```

### 3. Start the Web Interface
Start the backend server:

```bash
python -m uvicorn backend.api:app --reload
```

Then open your browser and navigate to: **http://localhost:8000**
You can drag and drop images into the UI to search for similar images instantly!

### 4. Optional: CLI Search
If you prefer searching via command line instead of the web interface:

```bash
python -m backend.cli search --query path/to/query_image.jpg --top-k 5
```

## Development
- **Backend:** Powered by `FastAPI` and `Qdrant`.
- **Frontend:** Built with `Next.js`, `TypeScript`, and `TailwindCSS v4`. When modifying frontend code, you must build and export the static files to `backend/static`.
