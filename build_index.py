"""Command-line interface for building the inverted index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from index_builder import IndexBuilder
from zip_extractor import extract_first_n_folders, is_zip_file


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Milestone 1 inverted index."
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        required=True,
        help="Path to the corpus (zip file or extracted DEV folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will contain the generated index files.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Directory to extract zip file to (if corpus-root is a zip file). "
             "If not specified, extracts to a temporary directory.",
    )
    parser.add_argument(
        "--max-folders",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of folders to process (default: process all folders).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    
    corpus_root = args.corpus_root
    extract_dir = args.extract_dir
    
    # Check if corpus_root is a zip file
    if is_zip_file(corpus_root):
        print(f"Detected zip file: {corpus_root}")
        
        # Determine extraction directory
        if extract_dir is None:
            # Extract to a temporary directory next to the zip file
            extract_dir = corpus_root.parent / "extracted_corpus"
        else:
            extract_dir = extract_dir
        
        if args.max_folders is None:
            print(f"Extracting all folders to: {extract_dir}")
        else:
            print(f"Extracting first {args.max_folders} folders to: {extract_dir}")
        
        try:
            # Extract folders (all if max_folders is None)
            corpus_root = extract_first_n_folders(
                zip_path=corpus_root,
                output_dir=extract_dir,
                max_folders=args.max_folders,
                zip_internal_path="DEV/",
            )
            print(f"Extraction complete. Using corpus root: {corpus_root}")
        except Exception as e:
            print(f"ERROR: Failed to extract zip file: {e}")
            return 1
    else:
        # Use the provided path directly
        if not corpus_root.exists():
            print(f"ERROR: Corpus root does not exist: {corpus_root}")
            return 1
        if not corpus_root.is_dir():
            print(f"ERROR: Corpus root is not a directory: {corpus_root}")
            return 1
    
    # Build the index
    builder = IndexBuilder(
        corpus_root=corpus_root,
        output_dir=args.output_dir,
    )
    builder.build()
    print(f"Index successfully built at {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


