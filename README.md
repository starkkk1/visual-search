# Image Search System (Starter)

This is an image search project that lets you:

- index a folder of images into vector embeddings
- query with an image and retrieve the most similar images

It now supports four embedding modes:

- `histogram` (fast baseline)
- `cnn_resnet50` (CNN features)
- `swin_tiny` (Swin Transformer features)
- `clip` (OpenAI CLIP ViT-B/32 features)

## Project Structure

```
data/
  images/          # put your dataset images here
  qdrant/          # Qdrant vector database is stored here
visual-search/
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

Put your images in `../data/images` (subfolders are supported).

### 2) Build index

```bash
python -m src.cli index
```

Optional:

```bash
python -m src.cli index --images ../data/images --method clip
```

Use deep models:

```bash
python -m src.cli index --method cnn_resnet50
python -m src.cli index --method swin_tiny
python -m src.cli index --method clip
```

### 3) Search

**Standard Search:**

```bash
python -m src.cli search --query ../data/images/example.jpg --top-k 5
```

Search uses the `histogram` collection by default. You can specify a different one:

```bash
python -m src.cli search --query ../data/images/example.jpg --collection clip --top-k 5
```

**Hybrid Search (ResNet50 + Swin):**

To use hybrid search, first build two separate indexes into their respective collections:

```bash
python -m src.cli index --method cnn_resnet50
python -m src.cli index --method swin_tiny
```

Then trigger the hybrid search using the `--hybrid` flag. You can optionally use the `--alpha` flag to adjust the weight between the two models (defaults to 0.5):

```bash
python -m src.cli search --query ../data/images/example.jpg \
  --hybrid \
  --alpha 0.5 \
  --top-k 5
```

### 4) Web Interface (New)

You can now search using a beautiful drag-and-drop web application.

To start the backend server:

```bash
python -m uvicorn src.api:app --reload
```

Then, open your web browser and navigate to: **http://localhost:8000**

## Notes

- Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- The index is now stored locally in a **Qdrant Vector Database**
- Similarity is cosine similarity in range approximately `[0, 1]`
- First run with CNN/Swin downloads pretrained weights from the internet
- Ongoing project changes are tracked in [CHANGELOG.md](CHANGELOG.md)

- [x] Migrate index to Qdrant Vector Database
- [ ] Store index metadata in SQLite for filtering and tags
