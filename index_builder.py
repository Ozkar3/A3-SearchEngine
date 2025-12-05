"""Index construction utilities for Milestone 1."""
from __future__ import annotations

import json
import os
import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Set

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
        self.partial_index_count = 0    # Counter used keep track on when to dump built index to disk before it gets too big in RAM

        # Deduplication helpers
        # Map content hash -> representative doc_id for exact duplicate detection
        self._content_hash_to_doc_id: Dict[str, int] = {}
        # Map doc_id -> set of tokens for near-duplicate detection
        self._doc_tokens: Dict[int, Set[str]] = {}
        # Statistics for reporting
        self._exact_duplicate_count: int = 0
        self._near_duplicate_count: int = 0
        # Jaccard similarity threshold for near-duplicates
        self._near_duplicate_threshold: float = 0.9

    def build(self) -> None:
        """Build the on-disk inverted index using partial dums to save memory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        doc_id = 0
        files_processed = 0
        files_skipped = 0

        MEMORY_THRESHOLD = 5000
        # FOR DEBUGGING; BUILDS SMALL INDEX INSTEAD OF ALL =======
        # MEMORY_THRESHOLD = 100
        # DEBUG_LIMIT = 1000
        #======================

        print("Building index...")
        for file_path in self._iter_corpus_files():
            # === DEBUGGING DELETE LATER ===
            # if files_processed >= DEBUG_LIMIT:
            #     print(f"DEBUG: Reached {DEBUG_LIMIT} files. Stopping early.")
            #     break
            # ========================
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

            # QUALITY FILTERS
            if url.endswith("~") or url.endswith(".tmp") or url.endswith(".bak"):
                files_skipped += 1
                continue

            # Calendar type sites
            if "?" in url:
                files_skipped += 1
                continue
            if "calendar" in url.lower() or "events" in url.lower():
                files_skipped += 1
                continue

            # Error pages with no useful info
            html_head = html[:2000].lower()
            if "404 not found" in html_head or "page not found" in html_head:
                files_skipped += 1
                continue
            if "whoops" in html_head and "trouble locating" in html_head:
                files_skipped += 1
                continue
            if "no news items found" in html_head:
                files_skipped += 1
                continue

            # ---------- Exact duplicate detection (content-based) ----------
            # Hash the raw HTML content; if we've seen this exact content before,
            # treat this page as an exact duplicate and skip it.
            content_hash = hashlib.md5(html.encode("utf-8", errors="ignore")).hexdigest()
            if content_hash in self._content_hash_to_doc_id:
                self._exact_duplicate_count += 1
                files_skipped += 1
                continue

            # Parse and extract token statistics once (used for both indexing
            # and near-duplicate detection).
            token_stats = self.parser.parse(html)
            tokens: Set[str] = set(token_stats.keys())

            # Skips sites with not many info
            if len(tokens) < 20:
                files_skipped += 1
                continue
            # ---------- Near-duplicate detection (token-based Jaccard) ----------
            # Compare this document's token set against already-indexed docs.
            # If Jaccard similarity is very high, treat it as a near-duplicate.
            is_near_duplicate = False
            if tokens:
                for existing_tokens in self._doc_tokens.values():
                    if not existing_tokens:
                        continue
                    intersection_size = len(tokens & existing_tokens)
                    union_size = len(tokens | existing_tokens)
                    if union_size == 0:
                        continue
                    jaccard = intersection_size / union_size
                    if jaccard >= self._near_duplicate_threshold:
                        is_near_duplicate = True
                        break

            if is_near_duplicate:
                self._near_duplicate_count += 1
                files_skipped += 1
                continue

            # At this point, the document is unique enough to index.
            self.doc_lookup[doc_id] = url
            self.doc_seen_urls[url] = doc_id

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

            # Record deduplication metadata only for documents we actually index.
            self._content_hash_to_doc_id[content_hash] = doc_id
            self._doc_tokens[doc_id] = tokens
            doc_id += 1
            
            # Everytime its hits a multiple of the threshold it dumps the current index to disk to save memory
            if doc_id % MEMORY_THRESHOLD == 0:
                self._dump_partial_index()

            # Dumps to disk anything left in memory
        if self._postings:
            self._dump_partial_index()

        print("===== Partial indexing COMPLETE =====")
        print(f"      Created {self.partial_index_count} partial index files.")
        self._merge_partial_indexes()

        print(
            "  Completed processing: "
            f"{files_processed} files processed, "
            f"{doc_id} documents indexed, "
            f"{files_skipped} skipped "
            f"({self._exact_duplicate_count} exact duplicates, "
            f"{self._near_duplicate_count} near-duplicates)"
        )
        ''' 
        _dump_partial_index replaces write_index 
        its used to implement partial index vs having the whole index in RAM
          '''
        #print("  Writing index...")
        # self._write_index()
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

    ''' 
        _dump_partial_index replaces write_index 
        its used to implement partial index vs having the whole index in RAM
    '''
    # def _write_index(self) -> None:
    #     """Write the inverted index directly from in-memory postings."""
    #     lexicon_path = self.output_dir / "lexicon.jsonl"
    #     postings_path = self.output_dir / "postings.jsonl"

    #     with open(lexicon_path, "w", encoding="utf-8") as lexicon_file, open(
    #         postings_path, "w", encoding="utf-8"
    #     ) as postings_file:
    #         # Sort terms alphabetically for consistent ordering
    #         for term in sorted(self._postings.keys()):
    #             postings = self._postings[term]
                
    #             # Sort postings by doc_id
    #             postings.sort(key=lambda posting: posting.doc_id)
                
    #             # Calculate doc_freq: number of unique documents containing this term
    #             doc_freq = len(postings)
                
    #             # Format: [{"doc_id": 0, "weighted_tf": 2.5, "raw_tf": 3}, ...]
    #             postings_data = [
    #                 {
    #                     "doc_id": posting.doc_id,
    #                     "weighted_tf": posting.weighted_tf,
    #                     "raw_tf": posting.raw_tf,
    #                     "avg_position": posting.avg_position,
    #                 }
    #                 for posting in postings
    #             ]
    #             postings_json = json.dumps(postings_data)
    #             offset = postings_file.tell()
    #             postings_file.write(postings_json + "\n")
    #             length = postings_file.tell() - offset

    #             record = {
    #                 "term": term,
    #                 "doc_freq": doc_freq,  # Number of unique documents containing this term
    #                 "offset": offset,  # Character offset in postings.jsonl file
    #                 "length": length,  # Length in characters
    #             }
    #             lexicon_file.write(json.dumps(record) + "\n")

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
        
    def _dump_partial_index(self) -> None:
        """Saves current index to file and clears memory"""
        # Creates the filename for the partial indexes
        filename = self.output_dir / f"partial_{self.partial_index_count}.jsonl"
        print(f"  Dumping partial index to {filename}...")

        # Sorts the terms in memory first. Makes merging faster
        sorted_terms = sorted(self._postings.keys())

        with open(filename, "w", encoding="utf-8") as f:
            for term in sorted_terms:
                postings_list = self._postings[term]
                # Formats the data
                record = {
                    "term": term,
                    "postings": [
                        {
                            "doc_id": p.doc_id,
                            "weighted_tf": p.weighted_tf,
                            "raw_tf": p.raw_tf,
                            "avg_position": p.avg_position,
                        }
                        for p in postings_list
                    ]
                }
                # One line per term to the disk
                f.write(json.dumps(record) + "\n")

        # Deletes the main index in RAM to continue loading the next partial index to avoid memory leak
        self._postings.clear()
        # Deletes the current comparison list used for similarity detection to avoid memory leak
        self._doc_tokens.clear()
        # Used to produce file names for the partial index with different #s
        self.partial_index_count += 1
    
    def _merge_partial_indexes(self) -> None:
        """
        Merges all partial index files into a final one.
        Opens all of the partial indexes, but only loads the first line of each one instead of loading the whole file for each
        Then Merges in order as indexes are presorted
        """
        print("Merging partial indexes...")
        
        # Opens all Files
        open_files = []
        for i in range(self.partial_index_count):
            filename = self.output_dir / f"partial_{i}.jsonl"
            open_files.append(open(filename, "r", encoding="utf-8"))

        # Holds the top lines from each file
        current_lines = []
        
        # Read first line from evry file
        for i, file_handle in enumerate(open_files):
            line = file_handle.readline()
            if line:
                data = json.loads(line)
                # store the term and the data
                current_lines.append([data["term"], data, i])

        # Output files
        with open(self.output_dir / "lexicon.jsonl", "w", encoding="utf-8") as lexicon_file, \
             open(self.output_dir / "postings.jsonl", "w", encoding="utf-8") as postings_file:

            # dumps everything to output files
            while current_lines:
                
                # Finds which one has the smallest term to place in new index
                current_lines.sort(key=lambda x: x[0])
                winner_term = current_lines[0][0]

                # Holds all who have the lowest term found as well 
                combined_postings = []
                
                # Holds files that need to grab next line  
                indices_to_update = []

                # searches through all first lines
                for card in current_lines:
                    term = card[0]
                    data = card[1]
                    file_index = card[2]
                    
                    if term == winner_term:
                        combined_postings.extend(data["postings"])
                        indices_to_update.append(file_index)
                    else:
                        break

                # sorts by doc id
                combined_postings.sort(key=lambda x: x["doc_id"])
                
                # postings.jsonl
                postings_str = json.dumps(combined_postings)
                offset = postings_file.tell() 
                postings_file.write(postings_str + "\n")
                length = postings_file.tell() - offset

                # lexicon.jsonl
                lexicon_record = {
                    "term": winner_term,
                    "doc_freq": len(combined_postings),
                    "offset": offset,
                    "length": length
                }
                lexicon_file.write(json.dumps(lexicon_record) + "\n")

                # gets new lines/terms from each file
                current_lines = [card for card in current_lines if card[0] != winner_term]

                # Reads new line only from the one that had the lowest term
                for index in indices_to_update:
                    line = open_files[index].readline()
                    if line:
                        new_data = json.loads(line)
                        current_lines.append([new_data["term"], new_data, index])

        # closes all the opened partial indexes
        for f in open_files:
            f.close()
            
        # deletes all the partial indexes to save space
        for i in range(self.partial_index_count):
           os.remove(self.output_dir / f"partial_{i}.jsonl")


