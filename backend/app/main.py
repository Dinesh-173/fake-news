"""FastAPI Application Entry Point for TruthLens."""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1.router import api_router
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.db.base import Base
from backend.app.db.session import get_engine
from backend.app.services.health_service import get_system_health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: manages database initialization and teardown."""
    logger.info("TruthLens API starting up in '%s' mode...", settings.ENVIRONMENT)
    try:
        engine = get_engine()
        # Initialize schema tables asynchronously on startup
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schemas verified and initialized.")
    except Exception as exc:
        logger.error("Startup database initialization error: %s", exc)

    yield

    logger.info("TruthLens API shutting down...")
    try:
        engine = get_engine()
        await engine.dispose()
        logger.info("Database engine disposed.")
    except Exception as exc:
        logger.error("Error disposing database engine: %s", exc)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="An Evidence-Aware Multimodal AI Framework for Fake News Detection and Explainable News Verification",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context_and_security_headers(request: Request, call_next):
    """Assigns unique Request ID, tracks latency, and sets security headers."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    response: Response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Provides structured RFC 7807 problem details for validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "type": "https://truthlens.ai/errors/validation-error",
            "title": "Request Validation Failed",
            "status": 422,
            "detail": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Prevents stack trace leaks to client and returns standardized error response."""
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.exception("Unhandled server exception for request %s: %s", request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "type": "https://truthlens.ai/errors/internal-server-error",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred. Please contact the administrator.",
            "request_id": request_id,
        },
    )


# Root health alias for load balancers
@app.get("/health", tags=["Health & Diagnostics"])
async def root_health():
    """Root health check endpoint."""
    return await get_system_health()


# Mount versioned API routes
app.include_router(api_router, prefix=settings.API_V1_STR)
