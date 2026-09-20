"""Service performing live health verification across system dependencies."""

from datetime import datetime, timezone

import sklearn
import torch
import transformers

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.db.session import check_database_connectivity
from backend.app.schemas.health import (
    DatabaseHealth,
    HealthResponse,
    MLRuntimeHealth,
)


def verify_ml_runtime() -> MLRuntimeHealth:
    """Verifies PyTorch runtime capability and tensor math operations."""
    try:
        # Perform live tensor calculation
        x = torch.ones((2, 2), dtype=torch.float32)
        y = torch.ones((2, 2), dtype=torch.float32)
        z = torch.matmul(x, y)
        ops_pass = bool(z.shape == (2, 2) and torch.allclose(z, 2.0 * torch.ones((2, 2))))
        tensor_status = "pass" if ops_pass else "fail"
    except Exception as exc:
        logger.error("ML runtime tensor verification failed: %s", exc)
        tensor_status = "fail"

    return MLRuntimeHealth(
        status="available" if tensor_status == "pass" else "unavailable",
        torch_version=torch.__version__,
        device=settings.MODEL_DEVICE,
        tensor_ops=tensor_status,
        scikit_learn_status=sklearn.__version__,
        transformers_status=transformers.__version__,
    )


async def get_system_health() -> HealthResponse:
    """Aggregates real-time health metrics from DB, ML, and application core."""
    db_info = await check_database_connectivity()
    db_health = DatabaseHealth(
        status=db_info["status"],
        dialect=db_info["dialect"],
        latency_ms=db_info["latency_ms"],
        error=db_info["error"],
    )

    ml_health = verify_ml_runtime()

    # Determine overall status
    if db_health.status == "up" and ml_health.status == "available":
        overall_status = "healthy"
    elif db_health.status == "up" or ml_health.status == "available":
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        application=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
        components={
            "database": db_health.model_dump(),
            "ml_runtime": ml_health.model_dump(),
        },
    )
