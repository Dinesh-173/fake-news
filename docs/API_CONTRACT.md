# TruthLens: Versioned REST API Contract (v1)

## 1. Global Conventions

- **Base URL**: `/api/v1`
- **Content Negotiation**: `application/json` (UTF-8)
- **Error Format**: RFC 7807 compliant problem details JSON.
- **Request Tracing**: Every request is tagged with an `X-Request-ID` UUID header echoed in responses.

---

## 2. Core Endpoints

### 2.1 Health & Diagnostics
#### `GET /api/v1/health`
Evaluates system health, database readiness, and machine learning runtime.

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-09-20T18:40:00Z",
  "components": {
    "database": {
      "status": "up",
      "dialect": "sqlite",
      "latency_ms": 1.2
    },
    "ml_runtime": {
      "status": "available",
      "torch_version": "2.13.0+cpu",
      "device": "cpu",
      "tensor_ops": "pass"
    }
  }
}
```

---

### 2.2 Article Ingestion & Analysis
#### `POST /api/v1/analyze`
Submits an article for full multimodal, evidence-grounded verification.

**Request**:
```json
{
  "text": "Breaking: Scientists in Antarctica confirm melting ice has revealed ancient pyramid.",
  "title": "Ancient Pyramids Uncovered in Antarctica",
  "url": "https://example-unverified-blog.org/pyramids-antarctica",
  "image_url": null,
  "options": {
    "retrieve_evidence": true,
    "generate_explanation": true,
    "max_claims": 5
  }
}
```

**Response (200 OK)**:
```json
{
  "analysis_id": "8f3b2c14-5231-4e78-9e12-b2f5674a9123",
  "assessment": {
    "state": "LIKELY_FALSE",
    "prediction_confidence": 0.94,
    "calibrated_confidence": 0.91,
    "evidence_sufficiency": 0.85,
    "conflict_score": 0.04
  },
  "claims": [
    {
      "claim_id": "c1",
      "text": "Scientists in Antarctica confirm melting ice has revealed ancient pyramid.",
      "source_sentence": "Breaking: Scientists in Antarctica confirm melting ice has revealed ancient pyramid.",
      "start_char": 10,
      "end_char": 83,
      "salience_score": 0.95
    }
  ],
  "evidence": [
    {
      "evidence_id": "e1",
      "claim_id": "c1",
      "source_domain": "livescience.com",
      "source_url": "https://www.livescience.com/antarctica-pyramid-mountain.html",
      "title": "No, There Aren't Pyramids in Antarctica",
      "snippet": "Geologists have repeatedly clarified that the pyramid-shaped peak is a natural nunatak, formed by glacial erosion over hundreds of millions of years.",
      "relevance_score": 0.93,
      "stance": "CONTRADICTS",
      "stance_confidence": 0.97
    }
  ],
  "explanation": {
    "summary": "The claim is LIKELY FALSE. Geological authorities confirm that the photographed formation is a natural nunatak (eroded mountain peak), not an artificial structure.",
    "evidence_breakdown": "LiveScience and Antarctic geological surveys confirm natural glacial erosion. No scientific expeditions report artificial archaeological discoveries."
  },
  "execution_stats": {
    "latency_ms": 340.5,
    "model_version": "truthlens-v1.0"
  }
}
```

---

### 2.3 Standalone Claim Extraction
#### `POST /api/v1/claims`
Extracts atomic factual propositions from arbitrary text without running the full verification pipeline.

---

### 2.4 Evidence Search
#### `POST /api/v1/evidence/search`
Queries retrieved external evidence for a given claim proposition.
