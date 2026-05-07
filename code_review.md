# Image Search System - Code Review

I have reviewed the current state of your image search system, including the recent additions for hybrid search and GPU support mentioned in the changelog. 

Overall, the project is well-structured. The decision to lazy-load `torch` and `timm` in `embeddings.py` is excellent for keeping the CLI responsive when using the baseline `histogram` method.

Here are my findings, highlighting a few critical performance bottlenecks and robustness issues that you should address next:

### 1. Severe Performance Bottleneck in Hybrid Search
In `src/search.py`, your new `hybrid_search_similar` function calculates distances using `NearestNeighbors`:
```python
nn_cnn = NearestNeighbors(metric="cosine", n_neighbors=len(paths_cnn))
nn_cnn.fit(vectors_cnn)
distances_cnn, indices_cnn = nn_cnn.kneighbors(query_vec_cnn)
```
**The Issue:** Setting `n_neighbors` to the total number of paths forces `scikit-learn` to calculate and sort the distance to *every single image* in the dataset. This O(N log N) operation will be extremely slow and memory-intensive for large indexes.
**The Fix:** Since your embeddings are already L2-normalized in `extract_embeddings_batch`, the cosine similarity is perfectly equivalent to a simple dot product. You can replace the NearestNeighbors block with a blazingly fast matrix multiplication:
```python
# query_vec is (1, D), vectors is (N, D). Dot product gives (1, N) similarities.
similarities_cnn = (query_vec_cnn @ vectors_cnn.T)[0]
scores_cnn = {paths_cnn[i]: similarities_cnn[i] for i in range(len(paths_cnn))}
```

### 2. Indexing Crashes on Corrupted Images
In `src/embeddings.py` -> `_extract_deep_embeddings_batch`, the code iterates over paths and opens them:
```python
for image_path in image_paths:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
```
**The Issue:** If even one image in your dataset is corrupted or unreadable, `Image.open` will raise an `UnidentifiedImageError` (or `OSError`), crashing the entire indexing process and losing all progress.
**The Fix:** You need to wrap the image opening in a `try...except` block to gracefully skip bad images. *Note: If you skip images, you must also update `src/indexer.py` to dynamically adjust the size of the pre-allocated `vectors` array, as it will no longer perfectly match `len(image_paths)`.*

### 3. Missing Apple Silicon (MPS) Acceleration
In `src/embeddings.py`, the device detection logic currently ignores Macs:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
**The Fix:** Add support for Apple's Metal Performance Shaders (MPS) to grant a massive speedup to Mac users:
```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

### 4. Security Risk in Index Loading
In `src/search.py`, `load_index` uses:
```python
data = np.load(index_file, allow_pickle=True)
```
**The Issue:** `allow_pickle=True` allows arbitrary Python code execution if a user loads a malicious `.npz` file. You currently need it because `np.savez_compressed` in `indexer.py` saves `paths` and `method` as `dtype=object`.
**The Fix:** Save the arrays as strings using `dtype=str` instead. This will allow you to safely load the index with `allow_pickle=False`.

---

Would you like me to go ahead and implement these fixes in the codebase?
