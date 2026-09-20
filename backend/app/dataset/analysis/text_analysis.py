"""Text length distribution and linguistic quality profiling module."""

import re
import string
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from backend.app.dataset.schemas import CanonicalNewsRecord

# Documented deterministic word-tokenization pattern:
# Matches alphanumeric word tokens (including standard contractions like don't)
# and standalone punctuation symbols, avoiding opaque third-party tokenizer dependencies.
TOKENIZER_REGEX = re.compile(r"\b\w+(?:'\w+)?\b|[^\w\s]", re.UNICODE)
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


class TextQualityThresholds(BaseModel):
    """Documented thresholds for identifying abnormal or low-quality textual records."""

    min_char_length: int = Field(
        default=15,
        description="Statements shorter than 15 characters typically lack propositional claim structure.",
    )
    max_char_length: int = Field(
        default=5000,
        description="Statements exceeding 5000 characters deviate from single-statement fact-checking format.",
    )
    max_punct_ratio: float = Field(
        default=0.25,
        description="Non-whitespace text containing >25% punctuation characters indicates spam or encoding artifacts.",
    )
    max_url_count: int = Field(
        default=2,
        description="Statements with >2 URLs often represent link spam rather than verifiable textual claims.",
    )
    repetition_max_ratio: float = Field(
        default=0.50,
        description="If unique words constitute <50% of tokens (for length >= 10), repetitive text is detected.",
    )


class TextQualityFlags(BaseModel):
    """Aggregate counts of records triggering specific textual anomalies."""

    empty_count: int = 0
    whitespace_only_count: int = 0
    extremely_short_count: int = 0
    extremely_long_count: int = 0
    excessive_punctuation_count: int = 0
    excessive_urls_count: int = 0
    repetitive_text_count: int = 0
    total_flagged_records: int = 0


class MetricDistribution(BaseModel):
    """Summary statistics and percentile distributions for a continuous metric."""

    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float


class TextAnalysisResult(BaseModel):
    """Comprehensive text length distributions, class-conditional metrics, and quality audit."""

    total_records: int
    char_length_stats: MetricDistribution
    token_length_stats: MetricDistribution
    class_conditional_char_stats: dict[str, MetricDistribution]
    class_conditional_token_stats: dict[str, MetricDistribution]
    quality_flags: TextQualityFlags
    quality_thresholds: dict[str, Any]
    tokenizer_name: str = "Regex-based Word Tokenizer (\\b\\w+(?:'\\w+)?\\b|[^\\w\\s])"


class TextAnalyzer:
    """Computes distribution statistics for character/token counts and assesses textual quality."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenizes text using the documented deterministic regex tokenizer."""
        if not text:
            return []
        return TOKENIZER_REGEX.findall(text)

    @classmethod
    def _compute_stats(cls, values: list[int | float]) -> MetricDistribution:
        if not values:
            return MetricDistribution(
                count=0,
                min=0.0,
                max=0.0,
                mean=0.0,
                median=0.0,
                std=0.0,
                p5=0.0,
                p25=0.0,
                p50=0.0,
                p75=0.0,
                p95=0.0,
                p99=0.0,
            )
        arr = np.array(values, dtype=float)
        return MetricDistribution(
            count=len(arr),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=round(float(np.mean(arr)), 2),
            median=round(float(np.median(arr)), 2),
            std=round(float(np.std(arr)), 2),
            p5=round(float(np.percentile(arr, 5)), 2),
            p25=round(float(np.percentile(arr, 25)), 2),
            p50=round(float(np.percentile(arr, 50)), 2),
            p75=round(float(np.percentile(arr, 75)), 2),
            p95=round(float(np.percentile(arr, 95)), 2),
            p99=round(float(np.percentile(arr, 99)), 2),
        )

    @classmethod
    def analyze(
        cls,
        records: list[CanonicalNewsRecord],
        thresholds: TextQualityThresholds | None = None,
    ) -> TextAnalysisResult:
        thresh = thresholds or TextQualityThresholds()
        if not records:
            empty_dist = cls._compute_stats([])
            return TextAnalysisResult(
                total_records=0,
                char_length_stats=empty_dist,
                token_length_stats=empty_dist,
                class_conditional_char_stats={},
                class_conditional_token_stats={},
                quality_flags=TextQualityFlags(),
                quality_thresholds=thresh.model_dump(),
            )

        char_lengths: list[int] = []
        token_lengths: list[int] = []
        class_chars: dict[str, list[int]] = {}
        class_tokens: dict[str, list[int]] = {}

        q_empty = 0
        q_ws = 0
        q_short = 0
        q_long = 0
        q_punct = 0
        q_urls = 0
        q_rep = 0
        flagged_indices: set[int] = set()

        punct_set = set(string.punctuation)

        for idx, r in enumerate(records):
            txt = r.text
            norm_label = r.normalized_label or "UNRESOLVED"

            if norm_label not in class_chars:
                class_chars[norm_label] = []
                class_tokens[norm_label] = []

            # 1. Empty / Whitespace check
            if not txt:
                q_empty += 1
                flagged_indices.add(idx)
                char_lengths.append(0)
                token_lengths.append(0)
                class_chars[norm_label].append(0)
                class_tokens[norm_label].append(0)
                continue

            if not txt.strip():
                q_ws += 1
                flagged_indices.add(idx)
                char_lengths.append(len(txt))
                token_lengths.append(0)
                class_chars[norm_label].append(len(txt))
                class_tokens[norm_label].append(0)
                continue

            # Length calculation
            c_len = len(txt)
            tokens = cls.tokenize(txt)
            t_len = len(tokens)

            char_lengths.append(c_len)
            token_lengths.append(t_len)
            class_chars[norm_label].append(c_len)
            class_tokens[norm_label].append(t_len)

            # Quality anomalies
            is_flagged = False
            if c_len < thresh.min_char_length:
                q_short += 1
                is_flagged = True
            elif c_len > thresh.max_char_length:
                q_long += 1
                is_flagged = True

            # Punctuation check (ratio of punctuation in non-whitespace chars)
            non_ws = [ch for ch in txt if not ch.isspace()]
            if non_ws:
                p_count = sum(1 for ch in non_ws if ch in punct_set)
                if (p_count / len(non_ws)) > thresh.max_punct_ratio:
                    q_punct += 1
                    is_flagged = True

            # URL check
            url_matches = URL_REGEX.findall(txt)
            if len(url_matches) > thresh.max_url_count:
                q_urls += 1
                is_flagged = True

            # Repetitive check (vocabulary density on longer statements)
            if t_len >= 10:
                lower_tokens = [tok.lower() for tok in tokens if tok[0].isalnum()]
                if lower_tokens:
                    unique_ratio = len(set(lower_tokens)) / len(lower_tokens)
                    if unique_ratio < thresh.repetition_max_ratio:
                        q_rep += 1
                        is_flagged = True

            if is_flagged:
                flagged_indices.add(idx)

        total_stats_char = cls._compute_stats(char_lengths)
        total_stats_token = cls._compute_stats(token_lengths)

        cond_chars = {k: cls._compute_stats(v) for k, v in sorted(class_chars.items())}
        cond_tokens = {k: cls._compute_stats(v) for k, v in sorted(class_tokens.items())}

        flags = TextQualityFlags(
            empty_count=q_empty,
            whitespace_only_count=q_ws,
            extremely_short_count=q_short,
            extremely_long_count=q_long,
            excessive_punctuation_count=q_punct,
            excessive_urls_count=q_urls,
            repetitive_text_count=q_rep,
            total_flagged_records=len(flagged_indices),
        )

        return TextAnalysisResult(
            total_records=len(records),
            char_length_stats=total_stats_char,
            token_length_stats=total_stats_token,
            class_conditional_char_stats=cond_chars,
            class_conditional_token_stats=cond_tokens,
            quality_flags=flags,
            quality_thresholds=thresh.model_dump(),
        )
