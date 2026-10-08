"""API endpoints for credibility analysis, claim extraction, and benchmarks."""

from typing import Any

from fastapi import APIRouter, Query, status

from backend.app.schemas.assessment import AnalyzeRequest, AnalyzeResponse
from backend.app.services.analysis_service import AnalysisService

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze news text credibility",
    description="Processes text through trained multimodal ML baselines, extracts factual claims, and queries evidence.",
)
async def analyze_article(
    request: AnalyzeRequest,
    model: str = Query(
        default="logistic_regression",
        description="ML baseline model to evaluate: 'logistic_regression' or 'svm'",
    ),
) -> AnalyzeResponse:
    return await AnalysisService.analyze_content(request, model_type=model)


@router.get(
    "/benchmarks",
    response_model=list[dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Get baseline model benchmarks",
    description="Returns empirical benchmark performance metrics from Phase 4 experiments.",
)
def get_benchmarks() -> list[dict[str, Any]]:
    return AnalysisService.get_benchmarks()
