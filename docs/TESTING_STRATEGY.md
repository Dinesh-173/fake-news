# TruthLens: Testing Strategy & Verification Policy

## 1. Testing Pyramid

TruthLens enforces a multi-tier test pyramid:

```text
               ┌────────────────────────┐
               │ End-to-End (E2E) Tests │ (5%)
               │ (Full API & Pipeline)  │
               ├────────────────────────┴──┐
               │   Integration Test Suite  │ (25%)
               │ (DB, Retrieval, Inference)│
         ┌─────┴───────────────────────────┴────┐
         │          Unit Test Suite             │ (70%)
         │ (Schemas, Preprocessors, Normalizers)│
         └──────────────────────────────────────┘
```

---

## 2. Test Suites Breakdown

### 2.1 Unit Tests (`backend/tests/unit/` and `tests/unit/`)
- Configuration validation (valid values, missing mandatory keys, type coercion).
- Pydantic schema validation and serialization.
- Text cleaning, tokenization, and sentence boundary detection.
- Metric calculation functions (ECE, Macro-F1, Precision, Recall).
- ML runtime sanity tests (Torch tensor creation, Scikit-learn fit/predict).

### 2.2 Integration Tests (`backend/tests/integration/`)
- Database session lifecycle, model relationship cascades, and transaction rollbacks.
- FastAPI endpoint test requests via `httpx.AsyncClient`.
- Health endpoint status reporting under operational and degraded states.
- Mocked external search provider failure handling (verifying graceful fallback to `INSUFFICIENT EVIDENCE`).

### 2.3 Boundary & Failure Tests
- Empty input strings, whitespace-only inputs, strings exceeding 50,000 characters.
- Corrupted images and unsupported MIME types.
- Malformed URLs and private IP ranges (SSRF validation).
- Unreachable database fallback and error handling.

---

## 3. Zero-Regression Policy

1. Every identified bug requires a minimal reproducing test case committed before fixing.
2. No existing assertions may be weakened or removed without documented peer review.
3. Automated CI/CD runs test suite with `pytest --cov` enforcing high coverage on business logic.
