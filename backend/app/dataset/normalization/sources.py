"""Source and domain normalization module."""

import re
import urllib.parse


class SourceNormalizer:
    """Extracts and standardizes publisher identities and domains without loss of provenance."""

    DOMAIN_CLEAN_RE = re.compile(r"^(?:https?:\/\/)?(?:www\.)?([^\/\?\#]+)", re.IGNORECASE)

    @classmethod
    def extract_domain(cls, url_or_domain: str | None) -> str | None:
        """Extracts canonical lowercased domain without 'www.' prefix."""
        if not url_or_domain or not str(url_or_domain).strip():
            return None

        clean_str = str(url_or_domain).strip()
        # Ensure scheme for urlparse if not present
        candidate = (
            clean_str if clean_str.startswith(("http://", "https://")) else "http://" + clean_str
        )
        try:
            parsed = urllib.parse.urlparse(candidate)
            netloc = parsed.netloc or ""
            if ":" in netloc:
                netloc = netloc.split(":")[0]
            if netloc.lower().startswith("www."):
                netloc = netloc[4:]
            # A valid domain must contain at least one dot with valid TLD length >= 2
            if "." in netloc and len(netloc.split(".")[-1]) >= 2:
                return netloc.lower()
            return None
        except Exception:
            return None

    @classmethod
    def normalize_source_name(cls, raw_source: str | None) -> str | None:
        """Standardizes common publisher and source name strings."""
        if not raw_source or not str(raw_source).strip():
            return None

        cleaned = str(raw_source).strip()
        # Remove common trailing suffixes and extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned
