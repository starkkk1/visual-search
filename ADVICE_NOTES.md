# System Improvement Notes

This file records practical recommendations for improving the current image search system.

## 1) Add Evaluation First
- Create a small labeled validation set.
- Track retrieval metrics:
  - Recall@1
  - Recall@5
  - Recall@10
  - mAP
- Use these metrics to verify whether each new change improves quality.

## 2) Improve Scalability of Search
- Current nearest-neighbor approach is fine for small datasets.
- For larger collections, use approximate nearest neighbor indexing (for example FAISS or HNSW) to improve speed and latency.

## 3) Speed Up Feature Extraction
- Embed images in batches for the CLIP backbone.
- Add GPU support when available, with CPU fallback.
- Cache embeddings so re-indexing only processes new or modified files.

## 4) Strengthen Metadata and Versioning
- Save these details in index metadata:
  - model/backbone name
  - preprocessing config
  - embedding dimension
  - creation timestamp
- This helps reproducibility and prevents model/index mismatch issues.

## 5) Add Metadata-Aware Retrieval
- Store image metadata (label, source, tags) in CSV or SQLite.
- Support filtered search after similarity ranking (example: by class or dataset source).

## 6) Improve Query Robustness
- Add consistent query preprocessing (resize/crop pipeline).
- Handle EXIF orientation explicitly.
- Detect and skip corrupt images during indexing.

## 7) Retrieval Algorithms To Use

### Recommended Options
- Cosine KNN (exact): best baseline for small datasets and debugging.
- HNSW (ANN): best default for medium and large datasets with strong speed/accuracy tradeoff.
- IVF + PQ (FAISS): best when dataset is very large and memory must be reduced.

### Recommended Stack For This Project
- Keep exact cosine KNN as the offline quality baseline.
- Use HNSW + cosine as the primary fast retrieval index.
- Use two-stage retrieval when scale grows:
  - Stage 1: retrieve top candidates quickly with ANN.
  - Stage 2: rerank candidates with full exact cosine similarity score.
