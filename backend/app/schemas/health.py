"""Health check schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatabaseHealth(BaseModel):
    status: str = Field(..., description="'up', 'degraded', or 'down'")
    dialect: str = Field(..., description="Database dialect (e.g. postgresql, sqlite)")
    latency_ms: float = Field(..., description="Ping latency in milliseconds")
    error: str | None = Field(None, description="Error message if unhealthy")


class MLRuntimeHealth(BaseModel):
    status: str = Field(..., description="'available' or 'unavailable'")
    torch_version: str
    device: str
    tensor_ops: str = Field(..., description="'pass' or 'fail'")
    scikit_learn_status: str = Field("available")
    transformers_status: str = Field("available")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall status: 'healthy', 'degraded', or 'unhealthy'")
    application: str = Field("TruthLens")
    version: str
    environment: str
    timestamp: datetime
    components: dict[str, Any]
