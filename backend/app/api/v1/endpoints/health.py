"""Health check API endpoints."""

from fastapi import APIRouter, status

from backend.app.schemas.health import HealthResponse
from backend.app.services.health_service import get_system_health

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health & Runtime Status",
    description="Returns verified live status for database connectivity, PyTorch ML runtime, and system metadata.",
)
async def health_check() -> HealthResponse:
    """Executes live component checks and returns system health status."""
    return await get_system_health()
