from __future__ import annotations

import argparse
from pathlib import Path

from .config import IMAGES_DIR, INDEX_FILE
from .embeddings import SUPPORTED_METHODS
from .indexer import build_index
from .search import hybrid_search_similar, search_similar


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image search system CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build or rebuild image index")
    index_parser.add_argument("--images", type=Path, default=IMAGES_DIR, help="Directory containing images")
    index_parser.add_argument("--index-file", type=Path, default=INDEX_FILE, help="Where to write index")
    index_parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default="histogram",
        help="Embedding method: histogram, cnn_resnet50, or swin_tiny",
    )

    search_parser = subparsers.add_parser("search", help="Search similar images")
    search_parser.add_argument("--query", type=Path, required=True, help="Path to query image")
    search_parser.add_argument("--images", type=Path, default=IMAGES_DIR, help="Directory containing indexed images")
    search_parser.add_argument("--index-file", type=Path, default=INDEX_FILE, help="Path to index file (or CNN index for hybrid)")
    search_parser.add_argument("--index-swin", type=Path, default=None, help="Path to Swin index file (enables hybrid search)")
    search_parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid search CNN weight (0.0 to 1.0)")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "index":
        count, index_path = build_index(
            images_dir=args.images,
            index_file=args.index_file,
            method=args.method,
        )
        print(f"Indexed {count} images using {args.method} -> {index_path}")
        return

    if args.command == "search":
        if args.index_swin is not None:
            results = hybrid_search_similar(
                query_image=args.query,
                images_dir=args.images,
                index_file_cnn=args.index_file,
                index_file_swin=args.index_swin,
                alpha=args.alpha,
                top_k=args.top_k,
            )
        else:
            results = search_similar(
                query_image=args.query,
                images_dir=args.images,
                index_file=args.index_file,
                top_k=args.top_k,
            )
        print("Top results:")
        for rank, (path, score) in enumerate(results, start=1):
            print(f"{rank:>2}. score={score:.4f} path={path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
