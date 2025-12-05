"""HTML parsing and token extraction utilities for the search engine indexer."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Set, Tuple

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk import download as nltk_download
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
try:
    word_tokenize("test")
except LookupError:
    nltk_download("punkt", quiet=True)
    nltk_download("punkt_tab", quiet=True)

# Filter out XMLParsedAsHTMLWarning since we're intentionally parsing HTML
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class TokenStats:
    """Container for statistics about a token within a document."""

    weighted_tf: float = 0.0
    raw_tf: int = 0
    positions: List[int] = field(default_factory=list)


class DocumentParser:
    """Extracts tokens from HTML documents while applying field-specific weights."""

    IMPORTANT_WEIGHTS = {
        "title": 4.0,
        "h1": 3.0,
        "h2": 2.5,
        "h3": 2.0,
        "strong": 1.75,
        "b": 1.75,
    }
    BASE_WEIGHT = 1.0

    def __init__(self) -> None:
        self._stemmer = PorterStemmer()

    def parse(self, html: str) -> Dict[str, TokenStats]:
        """Return a dictionary of stemmed tokens mapped to their statistics."""
        soup = BeautifulSoup(html or "", "lxml")
        stats: Dict[str, TokenStats] = {}
        position = 0

        def process_text(text: str, boost: float) -> None:
            nonlocal position
            for token in self._tokenize(text):
                stem = self._stemmer.stem(token)
                position += 1
                entry = stats.setdefault(stem, TokenStats())
                entry.raw_tf += 1
                entry.weighted_tf += boost
                entry.positions.append(position)

        for element in soup.find_all(string=True):
            parent_name = element.parent.name if element.parent else None
            if parent_name in {"script", "style", "noscript"}:
                continue
            process_text(element, self.BASE_WEIGHT)

        for tag, weight in self.IMPORTANT_WEIGHTS.items():
            for node in soup.find_all(tag):
                process_text(node.get_text(" ", strip=True), weight)

        return stats

    def parse_with_extras(
        self, html: str
    ) -> Tuple[Dict[str, TokenStats], Dict[str, TokenStats], Dict[str, TokenStats], List[str], Set[str]]:
        """Parse document and return tokens, n-grams, anchor text, links, and outbound URLs.
        
        Returns:
            Tuple of (token_stats, bigram_stats, trigram_stats, anchor_text_tokens, outbound_urls)
        """
        soup = BeautifulSoup(html or "", "lxml")
        stats: Dict[str, TokenStats] = {}
        bigram_stats: Dict[str, TokenStats] = {}
        trigram_stats: Dict[str, TokenStats] = {}
        position = 0
        anchor_text_tokens: List[str] = []
        outbound_urls: Set[str] = set()

        def process_text(text: str, boost: float) -> List[str]:
            """Process text and return list of stemmed tokens."""
            nonlocal position
            tokens = []
            for token in self._tokenize(text):
                stem = self._stemmer.stem(token)
                position += 1
                entry = stats.setdefault(stem, TokenStats())
                entry.raw_tf += 1
                entry.weighted_tf += boost
                entry.positions.append(position)
                tokens.append(stem)
            return tokens

        # Extract visible text nodes
        all_tokens: List[str] = []
        for element in soup.find_all(string=True):
            parent_name = element.parent.name if element.parent else None
            if parent_name in {"script", "style", "noscript"}:
                continue
            tokens = process_text(element, self.BASE_WEIGHT)
            all_tokens.extend(tokens)

        # Boost important fields
        for tag, weight in self.IMPORTANT_WEIGHTS.items():
            for node in soup.find_all(tag):
                tokens = process_text(node.get_text(" ", strip=True), weight)
                all_tokens.extend(tokens)

        # Extract anchor text and links
        for link in soup.find_all("a", href=True):
            href = link.get("href", "").strip()
            if href:
                outbound_urls.add(href)
            # Extract anchor text
            anchor_text = link.get_text(" ", strip=True)
            if anchor_text:
                # Tokenize and stem anchor text
                for token in self._tokenize(anchor_text):
                    stem = self._stemmer.stem(token)
                    anchor_text_tokens.append(stem)

        # Generate n-grams from all tokens
        # Bigrams (2-grams)
        for i in range(len(all_tokens) - 1):
            bigram = f"{all_tokens[i]}_{all_tokens[i+1]}"
            entry = bigram_stats.setdefault(bigram, TokenStats())
            entry.raw_tf += 1
            entry.weighted_tf += 1.0
            entry.positions.append(i)

        # Trigrams (3-grams)
        for i in range(len(all_tokens) - 2):
            trigram = f"{all_tokens[i]}_{all_tokens[i+1]}_{all_tokens[i+2]}"
            entry = trigram_stats.setdefault(trigram, TokenStats())
            entry.raw_tf += 1
            entry.weighted_tf += 1.0
            entry.positions.append(i)

        return stats, bigram_stats, trigram_stats, anchor_text_tokens, outbound_urls

    def _tokenize(self, text: str) -> Iterator[str]:
        """Tokenize a string into alphanumeric tokens using NLTK."""
        tokens = word_tokenize(text.lower())
        alphanumeric_pattern = re.compile(r"^[a-zA-Z0-9]+$")
        for token in tokens:
            if alphanumeric_pattern.match(token):
                yield token


