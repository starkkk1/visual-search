# Image Search System (Starter)

This is an image search project that lets you:
- index a folder of images into vector embeddings
- query with an image and retrieve the most similar images

It now supports three embedding modes:
- `histogram` (fast baseline)
- `cnn_resnet50` (CNN features)
- `swin_tiny` (Swin Transformer features)

## Project Structure

```
image_search_system/
  data/
    images/          # put your dataset images here
    index/           # generated index file is stored here
  src/
    config.py
    embeddings.py
    indexer.py
    search.py
    cli.py
    api.py           # FastAPI backend
    static/          # Web UI assets
  requirements.txt
```

## Setup

1. Activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1) Add images

Put your images in `data/images` (subfolders are supported).

### 2) Build index

```bash
python -m src.cli index
```

Optional:

```bash
python -m src.cli index --images data/images --index-file data/index/image_index.npz
```

Use deep models:

```bash
python -m src.cli index --method cnn_resnet50
python -m src.cli index --method swin_tiny
```

### 3) Search

```bash
python -m src.cli search --query data/images/example.jpg --top-k 5
```

Search automatically uses the embedding method saved in the index.

### 4) Web Interface (New)

You can now search using a beautiful drag-and-drop web application.

To start the backend server:

```bash
python -m uvicorn src.api:app --reload
```

Then, open your web browser and navigate to: **http://localhost:8000**

## Notes

- Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- The index is saved as a compressed NumPy file (`.npz`)
- Similarity is cosine similarity in range approximately `[0, 1]`
- First run with CNN/Swin downloads pretrained weights from the internet
- Ongoing project changes are tracked in [CHANGELOG.md](CHANGELOG.md)

## Next Upgrades

- Replace `extract_embedding` with CLIP embeddings
- Store index metadata in SQLite for filtering and tags
