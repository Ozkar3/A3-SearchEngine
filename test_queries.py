"""Test script to run all required queries for Milestone 2."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from search_engine import SearchEngine


def main() -> int:
    index_dir = Path("index_output")
    if not index_dir.exists():
        print(f"ERROR: Index directory does not exist: {index_dir}")
        return 1

    print("Initializing search engine...")
    search_engine = SearchEngine(index_dir)

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
    timings = {}

    try:
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Query: '{query}'")
            print(f"{'='*80}")

            start = time.perf_counter()
            results = search_engine.search(query, top_k=5)
            elapsed = time.perf_counter() - start
            timings[query] = elapsed
            all_results[query] = results

            print(f"\nSearch time: {elapsed*1000:.2f} ms")

            if not results:
                print("No results found.")
                continue

            print(f"\nTop 5 Results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result.score:.4f}] {result.url}")

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        for query in queries:
            results = all_results[query]
            print(f"\n{query}:")
            print(f"  Time: {timings[query]*1000:.2f} ms")
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

