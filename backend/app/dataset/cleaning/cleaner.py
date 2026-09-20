"""Minimally destructive data cleaner for textual and multimodal records."""

import re
import unicodedata
from datetime import datetime, timezone

from dateutil import parser as date_parser

from backend.app.dataset.schemas import CanonicalNewsRecord


class DataCleaner:
    """Performs deterministic, minimally destructive cleaning preserving NLP features."""

    # Unprintable control characters except \t, \n, \r
    CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

    # Consecutive horizontal whitespace (spaces, tabs) but not newlines
    HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\r\n]+")

    # Excessive consecutive newlines (more than 2 -> 2)
    EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")

    @classmethod
    def clean_text(cls, text: str | None) -> str:
        """
        Cleans text minimally:
        - Normalizes Unicode to standard NFC
        - Strips control characters
        - Collapses excessive horizontal whitespace without deleting newlines, casing, or punctuation
        """
        if not text:
            return ""

        # 1. Unicode NFC normalization
        normalized = unicodedata.normalize("NFC", text)

        # 2. Strip unprintable control characters
        cleaned = cls.CONTROL_CHAR_RE.sub("", normalized)

        # 3. Collapse multiple horizontal spaces to single space
        cleaned = cls.HORIZONTAL_WHITESPACE_RE.sub(" ", cleaned)

        # 4. Limit consecutive blank lines to at most two
        cleaned = cls.EXCESSIVE_NEWLINES_RE.sub("\n\n", cleaned)

        return cleaned.strip()

    @classmethod
    def normalize_timestamp(cls, raw_ts: str | None) -> tuple[str | None, datetime | None]:
        """Parses arbitrary timestamp string into UTC ISO string and datetime object."""
        if not raw_ts or not str(raw_ts).strip():
            return None, None

        cleaned_str = str(raw_ts).strip()
        try:
            dt = date_parser.parse(cleaned_str)
            # Ensure timezone awareness (default to UTC if naive)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat(), dt
        except (ValueError, OverflowError, TypeError):
            return cleaned_str, None

    @classmethod
    def clean_record(cls, record: CanonicalNewsRecord) -> CanonicalNewsRecord:
        """Applies cleaning to all fields of a canonical record and updates modality flags."""
        cleaned_text = cls.clean_text(record.text)
        cleaned_title = cls.clean_text(record.title) if record.title else None

        # Clean timestamp
        raw_ts, norm_dt = cls.normalize_timestamp(record.published_at_raw)

        # Update modality presence flags
        has_text = bool(cleaned_text and len(cleaned_text) >= 5)
        has_image = bool(record.image_path or record.image_url)
        has_metadata = bool(record.metadata or record.author or record.source_name_raw)
        has_timestamp = norm_dt is not None
        has_source = bool(
            record.source_name_normalized or record.source_domain or record.source_url
        )

        return record.model_copy(
            update={
                "text": cleaned_text,
                "title": cleaned_title,
                "published_at_raw": raw_ts,
                "published_at_normalized": norm_dt,
                "has_text": has_text,
                "has_image": has_image,
                "has_metadata": has_metadata,
                "has_timestamp": has_timestamp,
                "has_source": has_source,
            }
        )
