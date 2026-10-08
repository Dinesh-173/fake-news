"""Integration tests for web UI and credibility analysis endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_web_app_endpoint(client: AsyncClient):
    """Verify GET / returns 200 OK and serves TruthLens HTML."""
    response = await client.get("/")
    assert response.status_code == 200
    assert "TruthLens" in response.text
    assert "Live Credibility Verifier" in response.text


@pytest.mark.asyncio
async def test_analyze_endpoint_success(client: AsyncClient):
    """Verify POST /api/v1/analyze returns 200 OK and complete assessment."""
    payload = {
        "text": "NASA launched the exploration rover to Mars successfully from Cape Canaveral."
    }
    response = await client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "analysis_id" in data
    assert "assessment" in data
    assert data["assessment"]["state"] in [
        "LIKELY_TRUE",
        "LIKELY_FALSE",
        "MISLEADING_OR_PARTIALLY_SUPPORTED",
        "INSUFFICIENT_EVIDENCE",
    ]
    assert "claims" in data
    assert len(data["claims"]) > 0
    assert "evidence" in data
    assert "explanation" in data
    assert "execution_stats" in data


@pytest.mark.asyncio
async def test_benchmarks_endpoint(client: AsyncClient):
    """Verify GET /api/v1/benchmarks returns 200 OK and benchmark list."""
    response = await client.get("/api/v1/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
