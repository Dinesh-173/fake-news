"""TruthLens Claim Extraction and Proposition Decomposition Engine.

Decomposes raw article text or user submissions into atomic, verifiable claim propositions.
Computes character spans, salience scores, and search queries for each claim.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

# Opinion / subjective markers to filter out non-verifiable statements
SUBJECTIVE_MARKERS = {
    "i think",
    "i believe",
    "in my opinion",
    "i feel",
    "from my perspective",
    "personally",
    "it seems to me",
    "who knows",
    "i hope",
    "i wish",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "once",
}


@dataclass
class ClaimProposition:
    claim_id: str
    claim_text: str
    source_sentence: str
    start_char: int
    end_char: int
    salience_score: float
    search_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "source_sentence": self.source_sentence,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "salience_score": round(self.salience_score, 3),
            "search_queries": self.search_queries,
        }


class ClaimExtractor:
    """Extracts verifiable claims and generates search queries."""

    @classmethod
    def extract_sentences(cls, text: str) -> list[tuple[str, int, int]]:
        """
        Splits text into sentences while tracking exact character offsets (start, end).
        """
        sentences = []
        # Pattern splits on sentence terminals (. ! ?) followed by whitespace or end of string
        pattern = re.compile(r'([^.!?]+[.!?]*)(?:\s+|$)', re.MULTILINE)
        for match in pattern.finditer(text):
            sent = match.group(1).strip()
            if len(sent) > 0 and re.search(r"[a-zA-Z0-9]", sent):
                start = match.start(1)
                end = match.end(1)
                sentences.append((sent, start, end))
        return sentences

    @classmethod
    def is_verifiable_claim(cls, sentence: str) -> bool:
        """
        Evaluates whether a sentence contains verifiable factual assertions rather than pure subjective opinion.
        """
        lower = sentence.lower().strip()
        # Too short or is a question
        if len(lower.split()) < 4 or lower.endswith("?"):
            return False

        # Filter out obvious opinion expressions
        for marker in SUBJECTIVE_MARKERS:
            if lower.startswith(marker) or f", {marker}" in lower:
                return False

        # Check for factual assertion indicators: numbers, dates, organizations, or declarative verbs
        has_number_or_date = bool(re.search(r'\b\d+(?:,\d+)*(?:\.\d+)?\b|\b(?:19|20)\d{2}\b', lower))
        has_capitalized_entities = bool(re.search(r'\b[A-Z][a-z]+\b', sentence))
        has_declarative_verbs = bool(
            re.search(r'\b(?:announced|stated|reported|confirmed|claimed|signed|passed|voted|discovered|killed|found|increased|decreased|signed)\b', lower)
        )

        return has_number_or_date or has_capitalized_entities or has_declarative_verbs

    @classmethod
    def calculate_salience(cls, sentence: str, position_index: int, total_sentences: int) -> float:
        """
        Computes salience score [0.0, 1.0] representing importance for verification.
        Sentences in headlines / lead paragraphs and those with numerical figures get higher salience.
        """
        score = 0.50

        # Position bonus (lead sentences often contain core thesis)
        if position_index == 0:
            score += 0.25
        elif position_index < 3:
            score += 0.15

        # Numerical factual density bonus
        if re.search(r'\b\d+\b', sentence):
            score += 0.15

        # Named entity density bonus
        cap_words = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        if len(cap_words) >= 2:
            score += 0.10

        return min(1.0, round(score, 3))

    @classmethod
    def generate_search_queries(cls, claim_text: str) -> list[str]:
        """
        Generates targeted search queries for a claim.
        Produces:
        1. Entity/keyword query with fact-check intent.
        2. Cleaned concise query.
        """
        words = re.findall(r'\b[a-zA-Z0-9_\-]+\b', claim_text)
        filtered = [w for w in words if w.lower() not in STOPWORDS]

        if not filtered:
            return [claim_text]

        concise_query = " ".join(filtered[:8])
        factcheck_query = f"{concise_query} fact check"

        return [factcheck_query, concise_query]

    @classmethod
    def extract_claims(cls, text: str, max_claims: int = 5) -> list[ClaimProposition]:
        """
        Decomposes input text into ranked claim propositions.
        """
        sentences = cls.extract_sentences(text)
        candidates: list[ClaimProposition] = []

        for idx, (sent, start, end) in enumerate(sentences):
            if cls.is_verifiable_claim(sent):
                salience = cls.calculate_salience(sent, idx, len(sentences))
                queries = cls.generate_search_queries(sent)
                claim = ClaimProposition(
                    claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                    claim_text=sent,
                    source_sentence=sent,
                    start_char=start,
                    end_char=end,
                    salience_score=salience,
                    search_queries=queries,
                )
                candidates.append(claim)

        # If no strict verifiable claims found, fallback to top sentences
        if not candidates and sentences:
            top_sent, start, end = sentences[0]
            candidates.append(
                ClaimProposition(
                    claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                    claim_text=top_sent,
                    source_sentence=top_sent,
                    start_char=start,
                    end_char=end,
                    salience_score=0.50,
                    search_queries=cls.generate_search_queries(top_sent),
                )
            )

        # Sort by salience descending and cap at max_claims
        candidates.sort(key=lambda c: c.salience_score, reverse=True)
        return candidates[:max_claims]
