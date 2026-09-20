# TruthLens: Evidence Retrieval Strategy

## 1. Multi-Stage Evidence Retrieval Pipeline

Evidence retrieval in TruthLens transforms an extracted claim into authoritative, verifiable real-world facts through a four-stage process:

```text
Claim Statement: "NASA confirmed discovery of alien cities on Mars."
                             │
                             ▼
              [Stage 1: Query Formulation]
     - Keyword extraction (NER: "NASA", "Mars", "alien cities")
     - Semantic reformulation ("NASA Mars rover alien structures verification")
                             │
                             ▼
              [Stage 2: Hybrid Retrieval]
     - Dense Semantic Search (pgvector with sentence-transformers)
     - Sparse BM25 Search (exact entity match over indexed fact-check archives)
     - Dynamic Live Search (DuckDuckGo News / Web search fallback)
                             │
                             ▼
              [Stage 3: Cross-Encoder Reranking]
     - Cross-encoder scoring query-passage relevance
     - Domain authority & recency weighting
     - Filter out candidate passages below threshold theta_rel = 0.60
                             │
                             ▼
              [Stage 4: Stance & NLI Evaluation]
     - Pair (Claim, Top-K Evidence Passages)
     - NLI model predicts: SUPPORTS, CONTRADICTS, or NEUTRAL
     - Compute Evidence Sufficiency & Conflict Score
```

---

## 2. Stance Taxonomy & NLI Formulations

Given claim $c$ and retrieved evidence passage $e$:
- **SUPPORTS ($P_{\text{supp}}$)**: The passage asserts factual premises that confirm or validate the claim.
- **CONTRADICTS ($P_{\text{contra}}$)**: The passage asserts factual premises that directly refute, debunk, or contradict the claim.
- **CONTEXTUALIZES / NEUTRAL ($P_{\text{neut}}$)**: The passage discusses the subject matter but neither proves nor disproves the specific claim.

### Mathematical Conflict Index
When multiple evidence items are returned, TruthLens explicitly measures evidentiary consensus vs. conflict:

$$\text{Conflict Score} = 2 \times \min(S_{\text{supp}}, S_{\text{contra}})$$

Where $S_{\text{supp}} = \frac{1}{K} \sum_{k=1}^K w_k P_{\text{supp}}^{(k)}$ and $S_{\text{contra}} = \frac{1}{K} \sum_{k=1}^K w_k P_{\text{contra}}^{(k)}$.
If $\text{Conflict Score} > 0.40$, the system explicitly flags **Conflicting Evidence** rather than silently averaging probabilities.

---

## 3. Provenance Data Model

Every evidence object maintains strict cryptographic and network traceability:

```json
{
  "evidence_id": "evid_98f12a",
  "claim_id": "claim_01",
  "source_domain": "reuters.com",
  "source_url": "https://www.reuters.com/fact-check/nasa-mars-claims-debunked",
  "title": "Fact Check: NASA did not discover alien cities on Mars",
  "author": "Reuters Fact Check Team",
  "published_at": "2024-05-12T14:30:00Z",
  "retrieved_at": "2026-09-20T18:30:00Z",
  "snippet": "Claims circulating on social media that NASA rovers uncovered alien cities are false. NASA officials confirmed...",
  "relevance_score": 0.94,
  "stance": "CONTRADICTS",
  "stance_confidence": 0.98,
  "retrieval_provider": "duckduckgo_search"
}
```

If an external search provider is unreachable or returns 0 results, the system outputs:
```json
{
  "status": "EVIDENCE_UNAVAILABLE",
  "message": "External retrieval returned no corroborating sources for the given claims."
}
```
Fabricating fake sources or simulated URLs is strictly prohibited.
