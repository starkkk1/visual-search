# Image Search System

A visual search engine that indexes a folder of images into vector embeddings and serves a drag-and-drop web UI for similar-image retrieval.

It supports two embedding modes:
- `histogram` for a lightweight baseline
- `clip` for higher-quality visual matching

## Project Structure

```text
visual-search/
  frontend/          # Next.js app, exported as static files
  backend/           # FastAPI API and indexing/search logic
  data/images/       # Your image dataset
```

## Recommended Docker Setup

This repo now ships with:
- `Dockerfile` that builds the frontend and serves it from FastAPI
- `docker-compose.yml` that runs `backend` and `qdrant`
- `Dockerfile.gpu` and `docker-compose.gpu.yml` for CLIP on NVIDIA GPU
- persistent volumes for Qdrant storage and Hugging Face model cache

### 1. Put images in `data/images`

By default, Docker mounts `./data/images` into the backend container at `/app/data/images`.

### 2. Choose your deployment mode

For CPU-only deployment:

```bash
docker compose up --build -d
```

For NVIDIA GPU deployment:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

This starts:
- `backend` on `http://localhost:8000`
- `qdrant` on `http://localhost:6333`

GPU requirements:
- NVIDIA GPU on the Docker host
- current NVIDIA driver installed on the host
- NVIDIA Container Toolkit installed so Docker can pass the GPU into the container
- The GPU image uses PyTorch with CUDA 12.1, which is suitable for RTX 2070 Super as long as the host driver is new enough for CUDA 12.x

### 3. Build the search index

For the faster baseline:

```bash
docker compose run --rm backend python -m backend.cli index --method histogram
```

For CLIP search quality:

```bash
docker compose run --rm backend python -m backend.cli index --method clip
```

If you are using the GPU stack, use the same override files during indexing:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm backend python -m backend.cli index --method clip
```

Notes:
- The first CLIP indexing run downloads `openai/clip-vit-base-patch32`.
- That model cache is stored in the `huggingface_cache` Docker volume so it is reused later.
- The compose file defaults `IMAGE_SEARCH_DEFAULT_COLLECTION` to `clip`. Change it if you want the web app to search the histogram collection instead.
- The GPU stack intentionally does not reinstall `torch` from `requirements.txt`; it uses the CUDA-enabled PyTorch already bundled in `Dockerfile.gpu`.

### 4. Open the web app

Visit `http://localhost:8000` and upload an image to search.

## Timing and Profiling

Each `/search` request now records timing for:
- upload read
- temp file write
- embedding extraction
- Qdrant query
- result formatting
- total request time

Where to see it:
- backend logs print a `Search timing: {...}` line for every query
- the HTTP response includes a `Server-Timing` header
- the CLI search command prints `Timing (ms): {...}`

Example:

```bash
python -m backend.cli search --query path/to/query.jpg --collection clip --top-k 5
```

If you want browser-side inspection, open DevTools and inspect the `Server-Timing` header on the `/search` request.

## Local Python Workflow

If you want to run without Docker, install dependencies:

```bash
pip install -r requirements.txt
```

Then you can index locally with embedded Qdrant storage:

```bash
python -m backend.cli index --method histogram
python -m backend.cli index --method clip
python -m uvicorn backend.api:app --reload
```

Local mode uses `IMAGE_SEARCH_QDRANT_PATH` unless `QDRANT_URL` is set.

If your local Python environment has CUDA-enabled PyTorch and a supported NVIDIA GPU, the CLIP path will use `cuda` automatically.

## Environment Variables

Useful settings:

```env
IMAGE_SEARCH_IMAGES_DIR=./data/images
IMAGE_SEARCH_QDRANT_PATH=./data/qdrant
IMAGE_SEARCH_DEFAULT_COLLECTION=clip
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
LOG_SEARCH_TIMING=1
```

## Deploy Checklist

For a correct CLIP deployment:

1. Put your image corpus in `data/images`.
2. Start `qdrant` and `backend` with either CPU or GPU compose commands.
3. Run indexing against the same stack you will serve from.
4. Set `IMAGE_SEARCH_DEFAULT_COLLECTION=clip` if the web app should query the CLIP collection.
5. Confirm backend logs show timing data and that CLIP is not re-downloading weights on every run.
6. For GPU mode, verify the container can see the GPU before large indexing jobs.

Useful GPU verification command:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec backend python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## Notes

- In Docker, Qdrant runs as a separate service instead of embedded local storage.
- The backend now reuses a single Qdrant client per process, which avoids reopening the vector store on every search request.
- The frontend uses same-origin requests to `/search`, so serving the exported frontend from FastAPI keeps deployment simple.
