"""Search engine implementation with boolean AND queries and tf-idf scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from index_reader import IndexReader, Posting
from parser import DocumentParser


@dataclass
class SearchResult:
    """Represents a search result with score and URL."""

    doc_id: int
    url: str
    score: float


class SearchEngine:
    """Search engine that handles boolean AND queries with tf-idf scoring."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.reader = IndexReader(self.index_dir)
        self.parser = DocumentParser()

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for documents matching the query.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        # Parse and stem query terms
        query_terms = self._parse_query(query)
        if not query_terms:
            return []

        # Get postings for each query term
        term_postings = {}
        for term in query_terms:
            postings = self.reader.get_postings(term)
            if not postings:
                # If any term has no postings, AND query returns no results
                return []
            term_postings[term] = postings

        # Boolean AND: find documents that contain all terms
        candidate_docs = self._boolean_and(term_postings)
        if not candidate_docs:
            return []

        # Score documents using tf-idf
        scored_results = self._score_documents(candidate_docs, term_postings, query_terms)

        # Sort by score (descending) and return top K
        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:top_k]

    def _parse_query(self, query: str) -> List[str]:
        """Parse query string into stemmed terms."""
        # Tokenize and stem query (same as document parsing)
        terms = []
        for token in self.parser._tokenize(query):
            stemmed = self.parser._stemmer.stem(token)
            terms.append(stemmed)
        return terms

    def _boolean_and(self, term_postings: dict[str, List[Posting]]) -> set[int]:
        """Find documents that contain all query terms (boolean AND)."""
        if not term_postings:
            return set()

        # Start with documents from the first term
        term_list = list(term_postings.keys())
        first_term = term_list[0]
        candidate_docs = {posting.doc_id for posting in term_postings[first_term]}

        # Intersect with documents from remaining terms
        for term in term_list[1:]:
            term_docs = {posting.doc_id for posting in term_postings[term]}
            candidate_docs &= term_docs  # Set intersection

        return candidate_docs

    def _score_documents(
        self,
        candidate_docs: set[int],
        term_postings: dict[str, List[Posting]],
        query_terms: List[str],
    ) -> List[SearchResult]:
        """Score documents using tf-idf."""
        results = []

        # Build posting maps for faster lookup
        term_posting_maps = {}
        for term, postings in term_postings.items():
            posting_map = {posting.doc_id: posting for posting in postings}
            term_posting_maps[term] = posting_map

        # Compute IDF for each term
        term_idfs = {}
        for term in query_terms:
            term_idfs[term] = self.reader.compute_idf(term)

        # Score each candidate document
        for doc_id in candidate_docs:
            score = 0.0

            # Compute tf-idf for each query term
            for term in query_terms:
                posting = term_posting_maps[term].get(doc_id)
                if posting is None:
                    continue

                # TF: Use weighted_tf (which includes importance weights)
                # Normalize by document length
                doc_length = self.reader.get_doc_length(doc_id)
                if doc_length > 0:
                    # Normalized TF: weighted_tf / sqrt(doc_length)
                    # This follows cosine similarity normalization
                    normalized_tf = posting.weighted_tf / math.sqrt(doc_length)
                else:
                    normalized_tf = posting.weighted_tf

                # IDF
                idf = term_idfs[term]

                # TF-IDF score component
                score += normalized_tf * idf

            # Get URL for this document
            url = self.reader.get_url(doc_id)
            if url:
                results.append(SearchResult(doc_id=doc_id, url=url, score=score))

        return results

    def close(self) -> None:
        """Close the index reader."""
        if self.reader:
            self.reader.close()

