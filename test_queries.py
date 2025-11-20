"""Test script to run all required queries for Milestone 2."""

from __future__ import annotations

import sys
from pathlib import Path

from search_engine import SearchEngine


def main() -> int:
    index_dir = Path("index_output")
    if not index_dir.exists():
        print(f"ERROR: Index directory does not exist: {index_dir}")
        return 1

    print("Initializing search engine...")
    search_engine = SearchEngine(index_dir)

    # Required queries for Milestone 2
    queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering",
    ]

    print("\n" + "=" * 80)
    print("MILESTONE 2 QUERY RESULTS")
    print("=" * 80)

    all_results = {}

    try:
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Query: '{query}'")
            print(f"{'='*80}")
            
            results = search_engine.search(query, top_k=5)
            all_results[query] = results

            if not results:
                print("No results found.")
                continue

            print(f"\nTop 5 Results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result.score:.4f}] {result.url}")

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        for query in queries:
            results = all_results[query]
            print(f"\n{query}:")
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.url}")
            else:
                print("  No results found")

    finally:
        search_engine.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

