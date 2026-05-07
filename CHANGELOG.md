# Changelog

All notable project changes are documented here.

## 2026-05-06

### Added
- Added Apple Silicon (MPS) acceleration support for deep embedding extraction.

### Changed
- **BREAKING**: Changed index `.npz` file format to save metadata natively (`dtype=str`) instead of as python objects. This allows `allow_pickle=False` when loading indexes, greatly improving security. Existing index files must be rebuilt.
- Optimized `hybrid_search_similar` to use a direct O(N) dot product instead of a NearestNeighbors search, drastically speeding up queries for large datasets.

### Fixed
- Fixed a bug where a single corrupted image would crash the entire indexing process. The indexer now gracefully skips corrupted or unreadable images.

## 2026-04-29

### Added
- Added automatic GPU detection and execution to speed up embedding extraction.
- Added batch processing in `build_index` to significantly speed up indexing for large datasets.
- Added a `tqdm` progress bar to the indexing CLI command.
- Added `hybrid_search_similar` function to fuse CNN and Swin index scores.
- Added `--index-swin` and `--alpha` flags to the `search` CLI command to trigger hybrid search.
- Added `tqdm` to `requirements.txt`.


## 2026-04-15

### Added
- Initialized project scaffold for image search in [image_search_system](README.md):
  - [src/config.py](src/config.py), [src/embeddings.py](src/embeddings.py), [src/indexer.py](src/indexer.py), [src/search.py](src/search.py), [src/cli.py](src/cli.py)
  - [data/images/.gitkeep](data/images/.gitkeep), [data/index/.gitkeep](data/index/.gitkeep)
  - [requirements.txt](requirements.txt), [README.md](README.md), [.gitignore](.gitignore), [.env.example](.env.example)
- Added project advice notes file: [ADVICE_NOTES.md](ADVICE_NOTES.md)

### Changed
- Added deep embedding backbones and model selection:
  - `histogram` baseline retained
  - `cnn_resnet50` added
  - `swin_tiny` added
- Added method-aware index metadata storage and method-aware query search.
- Extended CLI with `--method {histogram,cnn_resnet50,swin_tiny}` for indexing.
- Updated dependencies with `torch` and `timm`.
- Updated usage docs in [README.md](README.md).
- Expanded [ADVICE_NOTES.md](ADVICE_NOTES.md) with retrieval algorithm guidance (exact cosine KNN, HNSW, IVF+PQ) and a practical hybrid fusion stack.
