from __future__ import annotations

import argparse
from pathlib import Path

from .config import IMAGES_DIR
from .embeddings import SUPPORTED_METHODS
from .indexer import build_index
from .search import search_similar, search_similar_by_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image search system CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build or rebuild image index")
    index_parser.add_argument("--images", type=Path, default=IMAGES_DIR, help="Directory containing images")
    index_parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default="histogram",
        help="Embedding method: histogram or clip",
    )

    search_parser = subparsers.add_parser("search", help="Search similar images")
    search_parser.add_argument("--query", type=Path, help="Path to query image")
    search_parser.add_argument("--text", type=str, help="Text query to search for similar images")
    search_parser.add_argument("--images", type=Path, default=IMAGES_DIR, help="Directory containing indexed images")
    search_parser.add_argument("--collection", type=str, default="histogram", help="Name of the Qdrant collection to search")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "index":
        count = build_index(
            images_dir=args.images,
            collection_name=args.method,
            method=args.method,
        )
        print(f"Indexed {count} images using {args.method} into collection '{args.method}'")
        return

    if args.command == "search":
        if not args.query and not args.text:
            parser.error("At least one of --query (image path) or --text (search text) must be provided.")
        if args.query and args.text:
            parser.error("Only one of --query or --text can be provided at a time.")

        if args.text:
            if args.collection != "clip":
                parser.error("Text search is only supported with the 'clip' collection. Set --collection clip.")
            results = search_similar_by_text(
                query_text=args.text,
                images_dir=args.images,
                collection_name=args.collection,
                top_k=args.top_k,
            )
            timing = None
        else:
            results, timing = search_similar(
                query_image=args.query,
                images_dir=args.images,
                collection_name=args.collection,
                top_k=args.top_k,
            )

        print("Top results:")
        for rank, (path, score) in enumerate(results, start=1):
            print(f"{rank:>2}. score={score:.4f} path={path}")
        if timing is not None:
            print(f"Timing (ms): {timing.as_dict()}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
