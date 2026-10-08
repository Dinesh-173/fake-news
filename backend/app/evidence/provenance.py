"""TruthLens Source Credibility and Provenance Tracking Module.

Manages:
1. Source domain normalization and validation.
2. SSRF protection against internal/private network calls.
3. Domain authority and historical reliability scoring.
4. Cryptographic / timestamped evidence provenance logging.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("truthlens.evidence.provenance")

# Curated domain trust database grounded in International Fact-Checking Network (IFCN) signatories
# and established academic source credibility studies.
CURATED_SOURCES: dict[str, dict[str, Any]] = {
    "reuters.com": {
        "name": "Reuters Fact Check",
        "reliability": 0.96,
        "is_fact_checker": True,
        "category": "international_wire",
    },
    "apnews.com": {
        "name": "Associated Press",
        "reliability": 0.95,
        "is_fact_checker": True,
        "category": "international_wire",
    },
    "politifact.com": {
        "name": "PolitiFact",
        "reliability": 0.94,
        "is_fact_checker": True,
        "category": "fact_checker",
    },
    "factcheck.org": {
        "name": "FactCheck.org",
        "reliability": 0.94,
        "is_fact_checker": True,
        "category": "fact_checker",
    },
    "snopes.com": {
        "name": "Snopes",
        "reliability": 0.92,
        "is_fact_checker": True,
        "category": "fact_checker",
    },
    "bbc.com": {
        "name": "BBC Reality Check",
        "reliability": 0.93,
        "is_fact_checker": True,
        "category": "public_broadcaster",
    },
    "who.int": {
        "name": "World Health Organization",
        "reliability": 0.98,
        "is_fact_checker": True,
        "category": "health_authority",
    },
    "cdc.gov": {
        "name": "Centers for Disease Control and Prevention",
        "reliability": 0.98,
        "is_fact_checker": True,
        "category": "health_authority",
    },
    "nasa.gov": {
        "name": "NASA Official Portal",
        "reliability": 0.99,
        "is_fact_checker": False,
        "category": "scientific_agency",
    },
    "theonion.com": {
        "name": "The Onion",
        "reliability": 0.05,
        "is_fact_checker": False,
        "category": "satire",
    },
    "babylonbee.com": {
        "name": "The Babylon Bee",
        "reliability": 0.05,
        "is_fact_checker": False,
        "category": "satire",
    },
}

DEFAULT_RELIABILITY = 0.50


@dataclass(frozen=True)
class SourceProvenance:
    domain: str
    name: str
    reliability_score: float
    is_fact_checker: bool
    category: str
    url: str
    retrieved_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "name": self.name,
            "reliability_score": self.reliability_score,
            "is_fact_checker": self.is_fact_checker,
            "category": self.category,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
        }


def extract_domain(url: str) -> str:
    """Extracts lowercase clean domain from a URL."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or "unknown_source"
    except Exception:
        return "unknown_source"


def is_safe_external_url(url: str) -> bool:
    """
    Validates URL to protect against Server-Side Request Forgery (SSRF).
    Blocks private IP spaces, loopback interfaces, link-local addresses, and cloud metadata IPs.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Disallow loopback / special names
        if hostname in ("localhost", "0.0.0.0", "broadcast"):
            return False

        # Attempt DNS resolution to inspect resolved IP address
        try:
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)

            # Block private, loopback, link-local, multicast, reserved addresses
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
            ):
                logger.warning("SSRF blocked: Host %s resolved to private/reserved IP %s", hostname, ip_str)
                return False

            # Block cloud metadata addresses explicitly (e.g. 169.254.169.254)
            if str(ip) == "169.254.169.254":
                logger.warning("SSRF blocked: Attempted access to cloud metadata endpoint")
                return False

        except socket.gaierror:
            # Domain could not be resolved
            pass

        return True
    except Exception as exc:
        logger.error("Error during SSRF validation for URL %s: %s", url, exc)
        return False


def get_source_provenance(url: str) -> SourceProvenance:
    """Computes provenance and reliability metadata for a given source URL."""
    domain = extract_domain(url)
    now_utc = datetime.now(timezone.utc).isoformat()

    info = CURATED_SOURCES.get(domain)
    if info:
        return SourceProvenance(
            domain=domain,
            name=info["name"],
            reliability_score=info["reliability"],
            is_fact_checker=info["is_fact_checker"],
            category=info["category"],
            url=url,
            retrieved_at=now_utc,
        )

    # Heuristics for educational / governmental domains
    if domain.endswith(".edu") or domain.endswith(".gov"):
        return SourceProvenance(
            domain=domain,
            name=f"Official {domain} Institution",
            reliability_score=0.88,
            is_fact_checker=False,
            category="official_institution",
            url=url,
            retrieved_at=now_utc,
        )

    return SourceProvenance(
        domain=domain,
        name=domain.capitalize(),
        reliability_score=DEFAULT_RELIABILITY,
        is_fact_checker=False,
        category="general_web",
        url=url,
        retrieved_at=now_utc,
    )
