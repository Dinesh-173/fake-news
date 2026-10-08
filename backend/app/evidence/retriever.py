"""TruthLens Multi-Source Evidence Retrieval and Reranking Engine.

Executes:
1. Multi-source evidence querying (Local Verified Knowledge Base + Safe Live Search).
2. SSRF-safe URL validation.
3. Evidence snippet deduplication and relevance reranking.
4. Automatic stance labeling and source provenance assignment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from backend.app.evidence.provenance import (
    get_source_provenance,
    is_safe_external_url,
)
from backend.app.evidence.stance_classifier import StanceClassifier
from backend.app.models.entities import StanceType

logger = logging.getLogger("truthlens.evidence.retriever")

# Curated offline verified knowledge base for reproducible testing and benchmark evaluation
LOCAL_FACT_CHECK_ARCHIVE = [
    {
        "title": "Fact Check: NASA has not discovered alien cities or structures on Mars",
        "url": "https://www.reuters.com/fact-check/nasa-mars-alien-cities-false-claim",
        "snippet": "Claims on social media asserting NASA rovers discovered ancient alien cities on Mars are false. NASA scientists confirmed no evidence of extraterrestrial civilizations or structures has been found.",
        "keywords": ["nasa", "mars", "alien", "cities", "civilization", "rover"],
        "provider": "Reuters Fact Check",
    },
    {
        "title": "PolitiFact: Vaccines do not contain microchips or alter human DNA",
        "url": "https://www.politifact.com/factchecks/2021/coronavirus-vaccine-microchips-dna-false",
        "snippet": "Medical experts and official CDC documentation confirm vaccines do not contain microchips, tracking devices, or alter human genetics. Such claims are Pants on Fire false.",
        "keywords": ["vaccine", "vaccines", "microchip", "dna", "cdc", "covid"],
        "provider": "PolitiFact",
    },
    {
        "title": "AP Fact Check: Congress passed bipartisan infrastructure bill signed into law",
        "url": "https://www.apnews.com/article/infrastructure-bill-signed-bipartisan-law",
        "snippet": "Official congressional records confirm the bipartisan Infrastructure Investment and Jobs Act was passed by Congress and signed into law by President Biden, allocating funding for roads, bridges, and broadband.",
        "keywords": ["infrastructure", "congress", "bipartisan", "law", "president", "funding", "roads"],
        "provider": "Associated Press",
    },
    {
        "title": "FactCheck.org: Social security benefits were not cancelled by executive order",
        "url": "https://www.factcheck.org/2023/social-security-executive-order-rumor-debunked",
        "snippet": "Viral posts claiming the president signed an executive order cancelling Social Security payments are completely false. Social Security is governed by federal statute, and no such order exists.",
        "keywords": ["social security", "executive order", "cancelled", "benefits", "president"],
        "provider": "FactCheck.org",
    },
    {
        "title": "WHO Statement: WHO officially declared end to global COVID-19 emergency phase",
        "url": "https://www.who.int/news/item/05-05-2023-statement-on-fifteenth-meeting-of-ihr-covid-19",
        "snippet": "The World Health Organization confirmed that COVID-19 no longer constitutes a public health emergency of international concern, while emphasizing ongoing disease surveillance remains essential.",
        "keywords": ["who", "covid", "emergency", "health", "world health organization", "pandemic"],
        "provider": "World Health Organization",
    },
]


@dataclass
class RetrievedEvidence:
    evidence_id: str
    claim_id: str
    source_domain: str
    source_url: str
    title: str
    snippet: str
    relevance_score: float
    stance: StanceType
    stance_confidence: float
    retrieval_provider: str
    reliability_score: float
    published_at: str | None = None
    retrieved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "source_domain": self.source_domain,
            "source_url": self.source_url,
            "title": self.title,
            "snippet": self.snippet,
            "relevance_score": round(self.relevance_score, 3),
            "stance": self.stance.value,
            "stance_confidence": round(self.stance_confidence, 3),
            "retrieval_provider": self.retrieval_provider,
            "reliability_score": round(self.reliability_score, 3),
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
        }


class EvidenceRetriever:
    """Multi-source evidence retriever with relevance reranking and deduplication."""

    @classmethod
    def compute_relevance(cls, claim_text: str, snippet: str, title: str = "") -> float:
        """
        Computes lexical and semantic overlap relevance between claim and candidate passage.
        """
        c_words = set(re.findall(r'\b\w{3,}\b', claim_text.lower()))
        combined_text = f"{title} {snippet}".lower()
        t_words = set(re.findall(r'\b\w{3,}\b', combined_text))

        if not c_words or not t_words:
            return 0.0

        intersection = c_words.intersection(t_words)
        coverage = len(intersection) / len(c_words)

        # Bonus for exact keyphrase matches
        phrases = [p for p in claim_text.lower().split() if len(p) >= 5]
        phrase_hits = sum(1 for p in phrases if p in combined_text)
        phrase_bonus = min(0.30, phrase_hits * 0.10)

        raw_score = 0.50 * coverage + phrase_bonus + 0.20 * (len(intersection) / len(t_words))
        return min(1.0, max(0.0, round(raw_score, 3)))

    @classmethod
    def deduplicate_evidence(cls, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Removes exact and near-duplicate snippets using snippet text fingerprinting.
        """
        seen_hashes = set()
        unique = []
        for item in items:
            text = item.get("snippet", "").strip().lower()
            h = hashlib.md5(text[:120].encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(item)
        return unique

    @classmethod
    def query_local_archive(cls, claim_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieves matching candidates from the verified fact-checking local archive.
        """
        c_lower = claim_text.lower()
        results = []
        for entry in LOCAL_FACT_CHECK_ARCHIVE:
            # Check keyword match
            kw_hits = sum(1 for kw in entry["keywords"] if kw in c_lower)
            if kw_hits > 0:
                rel = cls.compute_relevance(claim_text, entry["snippet"], entry["title"])
                results.append({
                    "title": entry["title"],
                    "url": entry["url"],
                    "snippet": entry["snippet"],
                    "provider": entry["provider"],
                    "relevance": max(rel, 0.45 + (kw_hits * 0.10)),
                })

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

    @classmethod
    def query_live_web(cls, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """
        Queries live search engine safely using DuckDuckGo API with SSRF protection and timeout.
        """
        encoded_q = urllib.parse.quote_plus(query)
        api_url = f"https://api.duckduckgo.com/?q={encoded_q}&format=json&no_html=1&skip_disambig=1"

        if not is_safe_external_url(api_url):
            logger.warning("Live web query URL failed SSRF check: %s", api_url)
            return []

        results = []
        try:
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "TruthLens-Research-Agent/1.0 (academic-verification)"},
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    # Extract Abstract
                    if data.get("Abstract") and data.get("AbstractURL"):
                        results.append({
                            "title": data.get("Heading", "Search Abstract"),
                            "url": data.get("AbstractURL"),
                            "snippet": data.get("Abstract"),
                            "provider": "DuckDuckGo Web Search",
                            "relevance": 0.70,
                        })
                    # Extract Related Topics
                    for topic in data.get("RelatedTopics", [])[:top_k]:
                        if isinstance(topic, dict) and topic.get("Text") and topic.get("FirstURL"):
                            results.append({
                                "title": topic.get("Text")[:60] + "...",
                                "url": topic.get("FirstURL"),
                                "snippet": topic.get("Text"),
                                "provider": "DuckDuckGo Web Search",
                                "relevance": 0.65,
                            })
        except Exception as exc:
            logger.debug("Live web retrieval returned no results or timed out: %s", exc)

        return results[:top_k]

    @classmethod
    def retrieve_evidence_for_claim(
        cls,
        claim_id: str,
        claim_text: str,
        search_queries: list[str] | None = None,
        max_evidence: int = 5,
        enable_live_web: bool = True,
    ) -> list[RetrievedEvidence]:
        """
        Executes end-to-end evidence retrieval, deduplication, stance labeling, and provenance tracking.
        """
        candidates: list[dict[str, Any]] = []

        # 1. Query local verified archive first
        local_results = cls.query_local_archive(claim_text, top_k=max_evidence)
        candidates.extend(local_results)

        # 2. Query live search if enabled and queries provided
        if enable_live_web and search_queries:
            for q in search_queries[:2]:
                web_results = cls.query_live_web(q, top_k=3)
                candidates.extend(web_results)

        # 3. Deduplicate
        deduped = cls.deduplicate_evidence(candidates)

        # 4. Filter by relevance threshold and construct final evaluated evidence objects
        evidence_items: list[RetrievedEvidence] = []
        for item in deduped:
            url = item["url"]
            if not is_safe_external_url(url):
                continue

            rel_score = cls.compute_relevance(claim_text, item["snippet"], item.get("title", ""))
            if rel_score < 0.25:
                continue

            prov = get_source_provenance(url)
            stance_res = StanceClassifier.classify_stance(claim_text, item["snippet"])

            ev = RetrievedEvidence(
                evidence_id=f"evid_{uuid.uuid4().hex[:8]}",
                claim_id=claim_id,
                source_domain=prov.domain,
                source_url=url,
                title=item.get("title", f"Report from {prov.domain}"),
                snippet=item["snippet"],
                relevance_score=rel_score,
                stance=stance_res.stance,
                stance_confidence=stance_res.confidence,
                retrieval_provider=item.get("provider", "Verified Knowledge Base"),
                reliability_score=prov.reliability_score,
                published_at=None,
                retrieved_at=prov.retrieved_at,
            )
            evidence_items.append(ev)

        # Sort by relevance * reliability descending
        evidence_items.sort(key=lambda e: e.relevance_score * e.reliability_score, reverse=True)
        return evidence_items[:max_evidence]
