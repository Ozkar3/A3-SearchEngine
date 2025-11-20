"""Command-line interface for the search engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from search_engine import SearchEngine


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the indexed corpus.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing the index files.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query string to search for. If not provided, enters interactive mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5).",
    )
    return parser.parse_args(argv)


def interactive_mode(search_engine: SearchEngine, top_k: int = 5) -> None:
    """Run search engine in interactive mode."""
    print("\n" + "=" * 70)
    print("Search Engine - Interactive Mode")
    print("=" * 70)
    print("Enter search queries (type 'quit' or 'exit' to stop)")
    print("-" * 70 + "\n")

    while True:
        try:
            query = input("Query: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print(f"\nSearching for: '{query}'")
            print("-" * 70)

            results = search_engine.search(query, top_k=top_k)

            if not results:
                print("No results found.\n")
                continue

            print(f"\nFound {len(results)} result(s):\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result.score:.4f}] {result.url}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def single_query_mode(
    search_engine: SearchEngine, query: str, top_k: int = 5
) -> None:
    """Run search engine for a single query."""
    print(f"Searching for: '{query}'")
    results = search_engine.search(query, top_k=top_k)

    if not results:
        print("No results found.")
        return

    print(f"\nTop {len(results)} result(s):\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.score:.4f}] {result.url}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.index_dir.exists():
        print(f"ERROR: Index directory does not exist: {args.index_dir}")
        return 1

    print("Initializing search engine...")
    search_engine = SearchEngine(args.index_dir)

    try:
        if args.query:
            single_query_mode(search_engine, args.query, args.top_k)
        else:
            interactive_mode(search_engine, args.top_k)
    finally:
        search_engine.close()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

