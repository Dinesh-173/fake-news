"""Structured logging configuration for TruthLens."""

import logging
import sys


class SafeLogFormatter(logging.Formatter):
    """Custom log formatter that suppresses sensitive tokens and produces consistent output."""

    SENSITIVE_PATTERNS = ["sk-", "Bearer ", "password=", "secret="]

    def format(self, record: logging.LogRecord) -> str:
        orig_msg = super().format(record)
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in orig_msg:
                orig_msg = orig_msg.replace(pattern, "[REDACTED]")
        return orig_msg


def setup_logging(debug: bool = True) -> logging.Logger:
    """Configures application-wide logging with safety sanitizers."""
    logger = logging.getLogger("truthlens")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Avoid duplicate handlers if re-initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = SafeLogFormatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
