"""Integration tests for FastAPI health endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_api_v1_health_endpoint(client: AsyncClient):
    """Verify GET /api/v1/health returns 200 OK and valid health report."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["application"] == "TruthLens"
    assert "version" in data
    assert "timestamp" in data

    components = data["components"]
    assert "database" in components
    assert "ml_runtime" in components

    # Check database section
    assert components["database"]["status"] in ("up", "degraded")
    assert components["database"]["latency_ms"] >= 0.0

    # Check ML runtime section
    assert components["ml_runtime"]["status"] == "available"
    assert components["ml_runtime"]["tensor_ops"] == "pass"

    # Check custom security & tracing headers
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


@pytest.mark.asyncio
async def test_root_health_endpoint(client: AsyncClient):
    """Verify root GET /health alias returns identical status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["application"] == "TruthLens"
