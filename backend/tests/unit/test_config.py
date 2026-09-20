"""Unit tests for configuration management."""

from backend.app.core.config import Settings


def test_default_settings():
    """Verify default configuration parameters."""
    cfg = Settings()
    assert cfg.PROJECT_NAME == "TruthLens"
    assert cfg.VERSION == "0.1.0"
    assert cfg.API_V1_STR == "/api/v1"
    assert cfg.MAX_ARTICLE_CHARS >= 10000
    assert cfg.DEFAULT_RANDOM_SEED == 42
    assert "cpu" in cfg.MODEL_DEVICE


def test_cors_origin_parsing():
    """Verify CORS origins parsed from list and JSON formats."""
    # List format
    cfg1 = Settings(CORS_ORIGINS=["http://localhost:3000", "http://example.com"])
    assert "http://localhost:3000" in cfg1.CORS_ORIGINS
    assert "http://example.com" in cfg1.CORS_ORIGINS

    # String comma-separated format
    cfg2 = Settings(CORS_ORIGINS="http://foo.com,http://bar.com")
    assert "http://foo.com" in cfg2.CORS_ORIGINS
    assert "http://bar.com" in cfg2.CORS_ORIGINS

    # JSON array string format
    cfg3 = Settings(CORS_ORIGINS='["http://json1.com", "http://json2.com"]')
    assert "http://json1.com" in cfg3.CORS_ORIGINS
    assert "http://json2.com" in cfg3.CORS_ORIGINS
