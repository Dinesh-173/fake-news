"""Unit tests for Pydantic request and response schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.app.models.entities import CredibilityState
from backend.app.schemas.assessment import (
    AnalyzeRequest,
    AssessmentResult,
)
from backend.app.schemas.health import (
    DatabaseHealth,
    HealthResponse,
    MLRuntimeHealth,
)


def test_health_response_schema():
    """Verify health schema serialization."""
    db_h = DatabaseHealth(status="up", dialect="sqlite", latency_ms=2.5)
    ml_h = MLRuntimeHealth(
        status="available",
        torch_version="2.13.0",
        device="cpu",
        tensor_ops="pass",
    )
    hr = HealthResponse(
        status="healthy",
        application="TruthLens",
        version="0.1.0",
        environment="test",
        timestamp=datetime.now(timezone.utc),
        components={"database": db_h.model_dump(), "ml_runtime": ml_h.model_dump()},
    )
    assert hr.status == "healthy"
    assert hr.components["database"]["status"] == "up"
    assert hr.components["ml_runtime"]["tensor_ops"] == "pass"


def test_assessment_result_validation():
    """Verify probability bounds on assessment results."""
    valid = AssessmentResult(
        state=CredibilityState.LIKELY_TRUE,
        prediction_confidence=0.85,
        calibrated_confidence=0.82,
        evidence_sufficiency=0.78,
        conflict_score=0.10,
    )
    assert valid.state == CredibilityState.LIKELY_TRUE

    # Invalid confidence > 1.0
    with pytest.raises(ValidationError):
        AssessmentResult(
            state=CredibilityState.LIKELY_TRUE,
            prediction_confidence=1.5,  # Exceeds max 1.0
            calibrated_confidence=0.82,
            evidence_sufficiency=0.78,
            conflict_score=0.10,
        )


def test_analyze_request_length_bounds():
    """Verify minimum and maximum text length enforcement."""
    # Text too short (< 20 chars)
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="Short text")

    # Valid text
    valid = AnalyzeRequest(
        text="This is a sufficiently long news article body for testing validation boundaries.",
        title="Test Article",
    )
    assert valid.title == "Test Article"
