# TruthLens

> **An Evidence-Aware Multimodal AI Framework for Fake News Detection and Explainable News Verification**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TruthLens is an academic research system and software prototype designed to overcome the critical failure modes of conventional fake news detection: binary oversimplification, ungrounded Large Language Model hallucinations, and opacity. Rather than querying an LLM as an unquestionable arbiter of truth, TruthLens grounds every assessment in discrete claim extraction, task-specific deep neural representations, external evidence retrieval, cross-modal consistency analysis, and calibrated uncertainty estimation.

---

## 1. System Architecture

```text
               News Input (Text, URL, Headline, Image)
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │     Preprocessing     │
                     │  & Input Normalization│
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │    Claim Extraction   │
                     │  (Span & Salience)    │
                     └───────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐   ┌───────────────────┐
│  Text Analysis   │    │  Image Analysis  │   │Evidence Retrieval │
│(RoBERTa/DeBERTa) │    │(OCR + Vision-Enc)│   │  & Query Engine   │
└────────┬─────────┘    └────────┬─────────┘   └─────────┬─────────┘
         │                       │                       │
         │                       ▼                       ▼
         │             ┌──────────────────┐    ┌───────────────────┐
         │             │Cross-Modal Check │    │ Stance & Relevance│
         │             │(Semantic Match)  │    │  (NLI Inference)  │
         │             └─────────┬────────┘    └─────────┬─────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Multimodal & Evidence │
                     │   Fusion Classifier   │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │Uncertainty Estimation │
                     │ & Temperature Scaling │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Structured Assessment │
                     │  (State + Provenance) │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  Guarded LLM Explainer│
                     │ (Evidence-Constrained)│
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  API Response / UI    │
                     │ (Full Traceability)   │
                     └───────────────────────┘
```

---

## 2. Credibility State Taxonomy

TruthLens rejects forced binary ("Real" vs. "Fake") labels in favor of four calibrated states:

1. **LIKELY TRUE**: Corroborated by high-authority external sources with minimal contradiction and consistent cross-modal context.
2. **LIKELY FALSE**: Directly contradicted by verified reporting or demonstrates fabricated/tampered modal context.
3. **MISLEADING / PARTIALLY SUPPORTED**: Contains genuine factual elements embedded in false framing or non-matching imagery.
4. **INSUFFICIENT EVIDENCE**: Retrieved external evidence is missing, low-relevance, or conflicting. The system refuses to produce an ungrounded classification.

---

## 3. Project Structure

```text
fake-news/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # Versioned API endpoints (Health, Analyze, Claims)
│   │   ├── core/            # Configuration, structured logging, security guards
│   │   ├── db/              # SQLAlchemy 2.0 session and base
│   │   ├── models/          # Relational entities (Article, Claim, Evidence, etc.)
│   │   ├── schemas/         # Pydantic v2 validation models
│   │   ├── services/        # Business logic & health check services
│   │   └── main.py          # FastAPI application entry point
│   └── tests/
│       ├── unit/            # Unit test suites (Config, Schemas, Security, ML)
│       ├── integration/     # Integration suites (Health API, DB sessions)
│       └── conftest.py      # Async fixtures and in-memory test database
├── docs/                    # Complete Phase 0 research specifications (19 documents)
├── data/                    # Versioned datasets (raw, interim, processed, external)
├── models/                  # Serialized model checkpoints and artifacts
├── experiments/             # Experiment runs, metrics, and master registry
├── scripts/                 # Environment validation and automation scripts
├── docker/                  # Multi-stage Dockerfiles
├── .env.example             # Environment configuration template
├── pyproject.toml           # Project dependencies, ruff, and pytest config
├── docker-compose.yml       # Production stack (PostgreSQL + pgvector, Redis, Backend)
└── README.md
```

---

## 4. Quickstart & Installation

### 4.1 Prerequisites
- Python 3.10 or newer (Verified on Python 3.14.7)
- Git

### 4.2 Local Environment Setup
```powershell
# 1. Clone repository
git clone <repo-url>
cd fake-news

# 2. Configure environment
copy .env.example .env

# 3. Verify ML & System Environment
python scripts/verify_env.py
```

### 4.3 Running the Automated Test Suite
```powershell
pytest
```

### 4.4 Launching the Backend Server
```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at `http://127.0.0.1:8000/docs`.

---

## 5. Implementation Status

| Component | Status | Details |
| :--- | :--- | :--- |
| **Phase 0: Specifications** | **VERIFIED** | 19 detailed specifications authored in `docs/` |
| **Phase 1: Environment & Health** | **VERIFIED** | FastAPI, async SQLAlchemy, PyTorch tensor checks, Pytest suites pass |
| **Phase 2: Dataset Pipeline** | **VERIFIED** | Validation, cleaning, exact/near deduplication, label mapping, zero-leakage splitting, manifests |
| **Phase 3: Dataset Quality Analysis** | *PENDING* | Planned in Phase 3 |
| **Phase 4–10: ML & Evidence Models** | *PENDING* | Planned in subsequent phases |
| **Phase 12: Web Interface** | *PENDING* | Planned in Phase 12 |

---

## 6. Research Integrity Notice
No metrics, benchmark accuracies, or datasets reported in TruthLens are fabricated. If an experiment has not yet executed on real data, it is marked as `PENDING` or `NOT IMPLEMENTED`.
