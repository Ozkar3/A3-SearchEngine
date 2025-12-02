"""Efficient index reader that reads from disk without loading entire index into memory."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from parser import DocumentParser


@dataclass
class LexiconEntry:
    """Represents a lexicon entry for a term."""

    term: str
    doc_freq: int
    offset: int
    length: int


@dataclass
class Posting:
    """Represents an occurrence of a token within a document."""

    doc_id: int
    weighted_tf: float
    raw_tf: int
    avg_position: float


class IndexReader:
    """Efficiently reads from the inverted index stored on disk."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.lexicon_path = self.index_dir / "lexicon.jsonl"
        self.postings_path = self.index_dir / "postings.jsonl"
        self.doc_lookup_path = self.index_dir / "doc_lookup.json"
        self.doc_lengths_path = self.index_dir / "doc_lengths.json"
        self.stats_path = self.index_dir / "stats.json"

        # Cache for lexicon entries (term -> LexiconEntry)
        # This is small compared to postings, so we can load it
        self._lexicon_cache: Optional[Dict[str, LexiconEntry]] = None

        # Cache for doc_lookup and doc_lengths (loaded once)
        self._doc_lookup: Optional[Dict[int, str]] = None
        self._doc_lengths: Optional[Dict[int, float]] = None
        self._num_documents: Optional[int] = None

        # File handle for postings file (opened on demand)
        self._postings_file_handle: Optional = None

    def _load_lexicon(self) -> Dict[str, LexiconEntry]:
        """Load the entire lexicon into memory for fast lookups."""
        if self._lexicon_cache is not None:
            return self._lexicon_cache

        lexicon: Dict[str, LexiconEntry] = {}
        print("Loading lexicon...")
        with open(self.lexicon_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if line_num % 50000 == 0 and line_num > 0:
                    print(f"  Loaded {line_num} lexicon entries...")
                record = json.loads(line)
                entry = LexiconEntry(
                    term=record["term"],
                    doc_freq=record["doc_freq"],
                    offset=record["offset"],
                    length=record["length"],
                )
                lexicon[entry.term] = entry

        self._lexicon_cache = lexicon
        print(f"Lexicon loaded: {len(lexicon)} terms")
        return lexicon

    def _load_doc_lookup(self) -> Dict[int, str]:
        """Load document ID to URL mapping."""
        if self._doc_lookup is not None:
            return self._doc_lookup

        print("Loading document lookup...")
        with open(self.doc_lookup_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert string keys to int keys
            self._doc_lookup = {int(doc_id): url for doc_id, url in data.items()}
        print(f"Document lookup loaded: {len(self._doc_lookup)} documents")
        return self._doc_lookup

    def _load_doc_lengths(self) -> Dict[int, float]:
        """Load document lengths for normalization."""
        if self._doc_lengths is not None:
            return self._doc_lengths

        print("Loading document lengths...")
        with open(self.doc_lengths_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert string keys to int keys
            self._doc_lengths = {int(doc_id): length for doc_id, length in data.items()}
        print(f"Document lengths loaded: {len(self._doc_lengths)} documents")
        return self._doc_lengths

    def get_num_documents(self) -> int:
        """Get total number of documents in the index."""
        if self._num_documents is not None:
            return self._num_documents

        with open(self.stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            self._num_documents = stats["num_documents"]
        return self._num_documents

    def get_postings(self, term: str) -> List[Posting]:
        """Get postings list for a given term from disk."""
        lexicon = self._load_lexicon()

        if term not in lexicon:
            return []

        entry = lexicon[term]

        # Read postings from disk using offset and length
        # Note: offset and length are in bytes/characters (UTF-8, ASCII-safe)
        # Each term's postings are on a single line in the JSONL file
        with open(self.postings_path, "r", encoding="utf-8") as f:
            f.seek(entry.offset)
            # Read exactly one line (each term's postings are on one line)
            # Use readline() to read until newline, then verify we got the right amount
            postings_json = f.readline()
            # Remove trailing newline
            postings_json = postings_json.rstrip('\n\r')

        postings_data = json.loads(postings_json)
        postings = [
            Posting(
                doc_id=posting["doc_id"],
                weighted_tf=posting["weighted_tf"],
                raw_tf=posting["raw_tf"],
                avg_position=posting["avg_position"],
            )
            for posting in postings_data
        ]

        return postings

    def get_doc_freq(self, term: str) -> int:
        """Get document frequency (number of documents containing the term)."""
        lexicon = self._load_lexicon()
        if term not in lexicon:
            return 0
        return lexicon[term].doc_freq

    def get_url(self, doc_id: int) -> Optional[str]:
        """Get URL for a given document ID."""
        doc_lookup = self._load_doc_lookup()
        return doc_lookup.get(doc_id)

    def get_doc_length(self, doc_id: int) -> float:
        """Get document length for normalization."""
        doc_lengths = self._load_doc_lengths()
        return doc_lengths.get(doc_id, 1.0)

    def compute_idf(self, term: str) -> float:
        """Compute inverse document frequency (IDF) for a term."""
        doc_freq = self.get_doc_freq(term)
        if doc_freq == 0:
            return 0.0
        num_docs = self.get_num_documents()
        return math.log(num_docs / doc_freq)

    def close(self) -> None:
        """Close any open file handles."""
        if self._postings_file_handle is not None:
            self._postings_file_handle.close()
            self._postings_file_handle = None

