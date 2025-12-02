"""HTML parsing and token extraction utilities for the search engine indexer."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

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

        # Extract visible text nodes (excluding scripts/styles)
        for element in soup.find_all(string=True):
            parent_name = element.parent.name if element.parent else None
            if parent_name in {"script", "style", "noscript"}:
                continue
            process_text(element, self.BASE_WEIGHT)

        # Boost important fields after baseline processing to ensure higher weights
        for tag, weight in self.IMPORTANT_WEIGHTS.items():
            for node in soup.find_all(tag):
                process_text(node.get_text(" ", strip=True), weight)

        return stats

    def _tokenize(self, text: str) -> Iterator[str]:
        """Tokenize a string into alphanumeric tokens using NLTK."""
        # Use NLTK's word_tokenize for better tokenization
        tokens = word_tokenize(text.lower())
        # Filter to only alphanumeric sequences (as per assignment requirements)
        alphanumeric_pattern = re.compile(r"^[a-zA-Z0-9]+$")
        for token in tokens:
            if alphanumeric_pattern.match(token):
                yield token


