"""Index construction utilities for Milestone 1."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator

from parser import DocumentParser

@dataclass
class Posting:
    """Represents an occurrence of a token within a document."""
    doc_id: int
    weighted_tf: float
    raw_tf: int
    avg_position: float

class IndexBuilder:
    """Coordinates parsing documents and writing the on-disk inverted index. Processes all domain folders from the corpus."""
    def __init__(
        self,
        corpus_root: Path,
        output_dir: Path,
    ) -> None:
        self.corpus_root = corpus_root
        self.output_dir = output_dir
        self.parser = DocumentParser()
        self.doc_lookup: Dict[int, str] = {}
        self.doc_lengths: Dict[int, float] = {}
        self.doc_seen_urls: Dict[str, int] = {}
        self._postings: Dict[str, list[Posting]] = defaultdict(list)

    def build(self) -> None:
        """Build the on-disk inverted index."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        doc_id = 0
        files_processed = 0
        files_skipped = 0

        print("Building index...")
        for file_path in self._iter_corpus_files():
            files_processed += 1
            if files_processed % 100 == 0:
                print(f"  Processed {files_processed} files, indexed {doc_id} documents...")
            
            payload = self._load_document(file_path)
            if payload is None:
                files_skipped += 1
                continue

            url = payload["url"]
            html = payload.get("content", "")

            if url in self.doc_seen_urls:
                files_skipped += 1
                continue  # skip duplicates that share a URL

            self.doc_lookup[doc_id] = url
            self.doc_seen_urls[url] = doc_id
            token_stats = self.parser.parse(html)

            length = 0.0
            for term, stats in token_stats.items():
                posting = Posting(
                    doc_id=doc_id,
                    weighted_tf=stats.weighted_tf,
                    raw_tf=stats.raw_tf,
                    avg_position=self._average(stats.positions),
                )
                self._postings[term].append(posting)

                term_tf = 1.0 + math.log(1.0 + stats.weighted_tf)
                length += term_tf * term_tf

            self.doc_lengths[doc_id] = max(length, 1e-9)
            doc_id += 1

        print(f"  Completed processing: {files_processed} files processed, {doc_id} documents indexed, {files_skipped} skipped")
        print("  Writing index...")
        self._write_index()
        print("  Writing metadata...")
        self._write_metadata()

    def _iter_corpus_files(self) -> Iterator[Path]:
        """Iterate through JSON files in all domain folders.
        
        Structure: corpus_root/DEV/domain_folder/*.json
        Each domain folder contains multiple JSON files (one per web page).
        Processes all folders in the corpus.
        """
        # Get all immediate subdirectories (domain folders like aiclub_ics_uci_edu)
        domain_folders = []
        try:
            for item in self.corpus_root.iterdir():
                if item.is_dir():
                    domain_folders.append(item)
        except PermissionError:
            print(f"ERROR: Permission denied accessing: {self.corpus_root}")
            return
        except OSError as e:
            print(f"ERROR: Cannot access directory: {self.corpus_root}")
            print(f"  Error: {e}")
            return
        
        # Sort for consistent ordering
        domain_folders.sort(key=lambda x: x.name)
        
        # Process ALL folders (no limit)
        selected_folders = domain_folders
        
        print(f"Found {len(domain_folders)} total domain folders")
        print(f"Processing all {len(selected_folders)} folders (alphabetically sorted)")
        
        # Count total JSON files across all folders
        total_json_files = 0
        for folder in selected_folders:
            try:
                json_count = sum(1 for f in folder.iterdir() 
                                if f.is_file() and f.suffix.lower() == ".json")
                total_json_files += json_count
            except (OSError, PermissionError):
                continue
        
        print(f"Found {total_json_files} JSON files across {len(selected_folders)} domain folders")
        
        # Iterate through JSON files in all folders
        for folder in selected_folders:
            try:
                for filename in folder.iterdir():
                    if filename.is_file() and filename.suffix.lower() == ".json":
                        yield filename
            except (OSError, PermissionError):
                # Skip folders we can't read
                continue

    def _load_document(self, path: Path) -> Dict[str, str] | None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None

    def _write_index(self) -> None:
        """Write the inverted index directly from in-memory postings."""
        lexicon_path = self.output_dir / "lexicon.jsonl"
        postings_path = self.output_dir / "postings.jsonl"

        with open(lexicon_path, "w", encoding="utf-8") as lexicon_file, open(
            postings_path, "w", encoding="utf-8"
        ) as postings_file:
            # Sort terms alphabetically for consistent ordering
            for term in sorted(self._postings.keys()):
                postings = self._postings[term]
                
                # Sort postings by doc_id
                postings.sort(key=lambda posting: posting.doc_id)
                
                # Calculate doc_freq: number of unique documents containing this term
                doc_freq = len(postings)
                
                # Format: [{"doc_id": 0, "weighted_tf": 2.5, "raw_tf": 3}, ...]
                postings_data = [
                    {
                        "doc_id": posting.doc_id,
                        "weighted_tf": posting.weighted_tf,
                        "raw_tf": posting.raw_tf,
                        "avg_position": posting.avg_position,
                    }
                    for posting in postings
                ]
                postings_json = json.dumps(postings_data)
                offset = postings_file.tell()
                postings_file.write(postings_json + "\n")
                length = postings_file.tell() - offset

                record = {
                    "term": term,
                    "doc_freq": doc_freq,  # Number of unique documents containing this term
                    "offset": offset,  # Character offset in postings.jsonl file
                    "length": length,  # Length in characters
                }
                lexicon_file.write(json.dumps(record) + "\n")

    def _write_metadata(self) -> None:
        lexicon_path = self.output_dir / "lexicon.jsonl"
        postings_path = self.output_dir / "postings.jsonl"
        doc_lookup_path = self.output_dir / "doc_lookup.json"
        doc_lengths_path = self.output_dir / "doc_lengths.json"
        stats_path = self.output_dir / "stats.json"

        doc_lookup_path.write_text(
            json.dumps(self.doc_lookup, indent=2),
            encoding="utf-8",
        )
        doc_lengths_path.write_text(
            json.dumps(self.doc_lengths, indent=2),
            encoding="utf-8",
        )

        index_size_bytes = (
            lexicon_path.stat().st_size if lexicon_path.exists() else 0
        ) + (postings_path.stat().st_size if postings_path.exists() else 0)
        
        # Also include doc_lookup and doc_lengths in total size
        doc_lookup_size = doc_lookup_path.stat().st_size if doc_lookup_path.exists() else 0
        doc_lengths_size = doc_lengths_path.stat().st_size if doc_lengths_path.exists() else 0
        total_size_bytes = index_size_bytes + doc_lookup_size + doc_lengths_size

        stats = {
            "num_documents": len(self.doc_lookup),
            "num_unique_terms": self._count_lines(lexicon_path),
            "index_size_bytes": total_size_bytes,
            "index_size_kb": round(total_size_bytes / 1024.0, 2),
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        
        print(f"  Index statistics:")
        print(f"    Documents indexed: {stats['num_documents']}")
        print(f"    Unique terms: {stats['num_unique_terms']}")
        print(f"    Index size: {stats['index_size_kb']} KB ({stats['index_size_bytes']} bytes)")

    def _average(self, values: Iterable[int]) -> float:
        total = 0.0
        count = 0
        for value in values:
            total += value
            count += 1
        return total / max(count, 1)

    def _count_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)


