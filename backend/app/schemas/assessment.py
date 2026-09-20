"""Pydantic schemas for verification, claims, and analysis."""

from pydantic import BaseModel, Field

from backend.app.models.entities import CredibilityState, StanceType


class ClaimBase(BaseModel):
    claim_text: str = Field(..., min_length=5, description="Extracted claim text")
    source_sentence: str
    start_char: int
    end_char: int
    salience_score: float = Field(default=1.0, ge=0.0, le=1.0)


class ClaimResponse(ClaimBase):
    claim_id: str


class EvidenceResponse(BaseModel):
    evidence_id: str
    claim_id: str
    source_domain: str
    source_url: str
    title: str | None = None
    snippet: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    stance: StanceType
    stance_confidence: float = Field(..., ge=0.0, le=1.0)
    retrieval_provider: str


class AssessmentResult(BaseModel):
    state: CredibilityState
    prediction_confidence: float = Field(..., ge=0.0, le=1.0)
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_sufficiency: float = Field(..., ge=0.0, le=1.0)
    conflict_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ExplanationResponse(BaseModel):
    summary: str
    evidence_breakdown: str
    faithfulness_score: float | None = None


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=50000, description="Article content or body")
    title: str | None = Field(None, max_length=1024)
    url: str | None = None
    image_url: str | None = None


class AnalyzeResponse(BaseModel):
    analysis_id: str
    assessment: AssessmentResult
    claims: list[ClaimResponse]
    evidence: list[EvidenceResponse]
    explanation: ExplanationResponse
    execution_stats: dict
