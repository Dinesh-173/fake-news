"""Central API v1 Router."""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import analyze, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health & Diagnostics"])
api_router.include_router(analyze.router, tags=["Credibility & Analysis"])
