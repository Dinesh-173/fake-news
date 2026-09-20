"""Unit tests for domain extraction and source normalization."""

from backend.app.dataset.normalization.sources import SourceNormalizer


def test_domain_extraction():
    """Verify clean domain parsing from diverse URL formats."""
    urls = {
        "https://www.reuters.com/world/us": "reuters.com",
        "http://apnews.com/article/123": "apnews.com",
        "https://sub.domain.bbc.co.uk/news": "sub.domain.bbc.co.uk",
        "https://edition.cnn.com:8080/path": "edition.cnn.com",
        "not_a_valid_url": None,
        "": None,
        None: None,
    }
    for url, expected in urls.items():
        assert SourceNormalizer.extract_domain(url) == expected


def test_normalize_source_name():
    """Verify source name whitespace cleanup."""
    raw = "   Associated   Press   "
    assert SourceNormalizer.normalize_source_name(raw) == "Associated Press"
    assert SourceNormalizer.normalize_source_name("") is None
