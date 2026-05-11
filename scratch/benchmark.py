import sys
import time
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.embeddings import SUPPORTED_METHODS, extract_embedding
from src.indexer import build_index
from src.search import search_similar, hybrid_search_similar

def get_class(path: Path) -> str:
    # Extracts 'ashmolean' from 'ashmolean_000000.jpg'
    return path.stem.rsplit('_', 1)[0]

def setup_small_dataset():
    images_dir = Path("data/images")
    small_dir = Path("data/images_small")
    small_dir.mkdir(parents=True, exist_ok=True)
    
    # Grab images from 5 different classes (20 per class) to ensure diversity
    from collections import defaultdict
    class_counts = defaultdict(int)
    diverse_images = []
    
    for img in images_dir.glob("*.jpg"):
        cls = get_class(img)
        if class_counts[cls] < 20 and len(class_counts) <= 5:
            diverse_images.append(img)
            class_counts[cls] += 1
            
        if len(diverse_images) >= 100:
            break
            
    for img in diverse_images:
        dest = small_dir / img.name
        if not dest.exists():
            shutil.copy(img, dest)
            
    # Pick 5 unique queries representing different categories
    queries = []
    seen_classes = set()
    for img in diverse_images:
        cls = get_class(img)
        if cls not in seen_classes:
            queries.append(img)
            seen_classes.add(cls)
        if len(queries) == 5:
            break
            
    return small_dir, queries

def main():
    print("Setting up small dataset for benchmark...")
    small_dir, queries = setup_small_dataset()
    
    indices = {
        "histogram": Path("data/images_small/index_hist_bench.npz"),
        "cnn_resnet50": Path("data/images_small/index_cnn_bench.npz"),
        "swin_tiny": Path("data/images_small/index_swin_bench.npz"),
        "clip": Path("data/images_small/index_clip_bench.npz")
    }
    
    print("\nBuilding indices...")
    for method, index_path in indices.items():
        if not index_path.exists():
            print(f"Building {method} index...")
            build_index(small_dir, index_path, method)
    
    # Warmup
    print("\nWarming up models...")
    for method in SUPPORTED_METHODS:
        extract_embedding(queries[0], method)
        
    results_dict = {}
    
    print(f"\nBenchmarking {len(queries)} queries (measuring Time and Precision@5)...")
    print("-" * 65)
    print(f"{'Method':<15} | {'Latency':<18} | {'Precision@5':<15}")
    print("-" * 65)
    
    # Helper to evaluate a specific method
    def evaluate(method_name, query_func):
        start_time = time.time()
        total_precision = 0.0
        
        for q in queries:
            q_class = get_class(q)
            results = query_func(q)
            
            # calculate precision@5
            hits = sum(1 for res_path, _ in results if get_class(Path(res_path)) == q_class)
            precision = hits / len(results) if results else 0
            total_precision += precision
            
        avg_time = (time.time() - start_time) / len(queries)
        avg_precision = total_precision / len(queries)
        results_dict[method_name] = {"time": avg_time, "precision": avg_precision}
        
        print(f"{method_name:<15} | {avg_time:.4f} secs/query | {avg_precision*100:>5.1f}%")
    
    # Standalone benchmarks
    for method, index_path in indices.items():
        evaluate(
            method,
            lambda q: search_similar(q, small_dir, index_path, top_k=5)
        )
        
    # Hybrid benchmark
    evaluate(
        "hybrid",
        lambda q: hybrid_search_similar(
            q, small_dir, indices["cnn_resnet50"], indices["swin_tiny"], alpha=0.5, top_k=5
        )
    )
    print("-" * 65)
    
if __name__ == "__main__":
    main()
