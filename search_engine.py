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
        """Search for documents matching the query with enhanced features.
        """
        # Parse and stem query terms
        query_terms = self._parse_query(query)
        if not query_terms:
            return []

        term_postings = {}
        for term in query_terms:
            postings = self.reader.get_postings(term)
            term_postings[term] = postings

        # Get n-gram postings
        bigram_postings = {}
        trigram_postings = {}
        if len(query_terms) >= 2:
            # Generate bigrams from query
            for i in range(len(query_terms) - 1):
                bigram = f"{query_terms[i]}_{query_terms[i+1]}"
                postings = self.reader.get_bigram_postings(bigram)
                if postings:
                    bigram_postings[bigram] = postings

        if len(query_terms) >= 3:
            # Generate trigrams from query
            for i in range(len(query_terms) - 2):
                trigram = f"{query_terms[i]}_{query_terms[i+1]}_{query_terms[i+2]}"
                postings = self.reader.get_trigram_postings(trigram)
                if postings:
                    trigram_postings[trigram] = postings

        # Get anchor text postings
        anchor_postings = {}
        for term in query_terms:
            postings = self.reader.get_anchor_postings(term)
            if postings:
                anchor_postings[term] = postings

        candidate_docs = self._boolean_and(term_postings)
        
        for postings in bigram_postings.values():
            candidate_docs |= {p.doc_id for p in postings}
        for postings in trigram_postings.values():
            candidate_docs |= {p.doc_id for p in postings}

        if not candidate_docs:
            return []

        scored_results = self._score_documents_enhanced(
            candidate_docs,
            term_postings,
            bigram_postings,
            trigram_postings,
            anchor_postings,
            query_terms,
        )

        # Sort by score (descending) and return top K
        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:top_k]

    def _parse_query(self, query: str) -> List[str]:
        """Parse query string into stemmed terms."""
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

    def _score_documents_enhanced(
        self,
        candidate_docs: set[int],
        term_postings: dict[str, List[Posting]],
        bigram_postings: dict[str, List[Posting]],
        trigram_postings: dict[str, List[Posting]],
        anchor_postings: dict[str, List[Posting]],
        query_terms: List[str],
    ) -> List[SearchResult]:
        """Score documents using enhanced features: TF-IDF, n-grams, positions, anchor text."""
        results = []

        # Build posting maps for faster lookup
        term_posting_maps = {}
        for term, postings in term_postings.items():
            term_posting_maps[term] = {p.doc_id: p for p in postings}

        bigram_posting_maps = {}
        for bigram, postings in bigram_postings.items():
            bigram_posting_maps[bigram] = {p.doc_id: p for p in postings}

        trigram_posting_maps = {}
        for trigram, postings in trigram_postings.items():
            trigram_posting_maps[trigram] = {p.doc_id: p for p in postings}

        anchor_posting_maps = {}
        for term, postings in anchor_postings.items():
            anchor_posting_maps[term] = {p.doc_id: p for p in postings}

        # Compute IDF for each term
        term_idfs = {}
        for term in query_terms:
            term_idfs[term] = self.reader.compute_idf(term)

        # Score each candidate document
        for doc_id in candidate_docs:
            score = 0.0

            for term in query_terms:
                posting = term_posting_maps.get(term, {}).get(doc_id)
                if posting is None:
                    continue

                doc_length = self.reader.get_doc_length(doc_id)
                if doc_length > 0:
                    normalized_tf = posting.weighted_tf / math.sqrt(doc_length)
                else:
                    normalized_tf = posting.weighted_tf

                idf = term_idfs[term]
                score += normalized_tf * idf

            for bigram, posting_map in bigram_posting_maps.items():
                if doc_id in posting_map:
                    posting = posting_map[doc_id]
                    score += posting.weighted_tf * 1.5

            for trigram, posting_map in trigram_posting_maps.items():
                if doc_id in posting_map:
                    posting = posting_map[doc_id]
                    score += posting.weighted_tf * 2.0 
            for term in query_terms:
                if term in anchor_posting_maps and doc_id in anchor_posting_maps[term]:
                    score += 3.0

            if len(query_terms) > 1:
                positions = []
                for term in query_terms:
                    posting = term_posting_maps.get(term, {}).get(doc_id)
                    if posting and posting.avg_position > 0:
                        positions.append(posting.avg_position)
                
                if len(positions) > 1:
                    positions.sort()
                    # Reward documents where terms appear close together
                    min_gap = min(positions[i+1] - positions[i] for i in range(len(positions)-1))
                    proximity_bonus = 1.0 / (1.0 + min_gap / 10.0)  # Decay with distance
                    score += proximity_bonus * 0.5

            # Get URL for this document
            url = self.reader.get_url(doc_id)
            if url:
                results.append(SearchResult(doc_id=doc_id, url=url, score=score))

        return results

    def close(self) -> None:
        """Close the index reader."""
        if self.reader:
            self.reader.close()

