"""SQLAlchemy ORM models for TruthLens entities."""

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class CredibilityState(str, enum.Enum):
    LIKELY_TRUE = "LIKELY_TRUE"
    LIKELY_FALSE = "LIKELY_FALSE"
    MISLEADING_OR_PARTIALLY_SUPPORTED = "MISLEADING_OR_PARTIALLY_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class StanceType(str, enum.Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid, index=True)
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    body_text: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(String(256), nullable=True)
    publisher: Mapped[str | None] = mapped_column(String(256), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    sha256_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    # Relationships
    claims: Mapped[list["Claim"]] = relationship(
        "Claim", back_populates="article", cascade="all, delete-orphan", lazy="selectin"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="article", cascade="all, delete-orphan", lazy="selectin"
    )


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid, index=True)
    article_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_sentence: Mapped[str] = mapped_column(Text, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    salience_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    # Relationships
    article: Mapped["Article"] = relationship("Article", back_populates="claims", lazy="selectin")
    evidence_items: Mapped[list["Evidence"]] = relationship(
        "Evidence", back_populates="claim", cascade="all, delete-orphan", lazy="selectin"
    )


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid, index=True)
    domain: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    historical_reliability_score: Mapped[float] = mapped_column(Float, default=0.50, nullable=False)
    is_fact_checker: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    country: Mapped[str | None] = mapped_column(String(64), nullable=True)

    evidence_items: Mapped[list["Evidence"]] = relationship(
        "Evidence", back_populates="source", lazy="selectin"
    )


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid, index=True)
    claim_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("sources.id", ondelete="SET NULL"), nullable=True, index=True
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    stance: Mapped[StanceType] = mapped_column(
        Enum(StanceType), nullable=False, default=StanceType.UNKNOWN
    )
    stance_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    retrieval_provider: Mapped[str] = mapped_column(String(64), nullable=False)

    # Relationships
    claim: Mapped["Claim"] = relationship("Claim", back_populates="evidence_items", lazy="selectin")
    source: Mapped[Optional["Source"]] = relationship(
        "Source", back_populates="evidence_items", lazy="selectin"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid, index=True)
    article_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    assessment_state: Mapped[CredibilityState] = mapped_column(
        Enum(CredibilityState), nullable=False
    )
    prediction_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    calibrated_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_sufficiency: Mapped[float] = mapped_column(Float, nullable=False)
    conflict_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    article: Mapped["Article"] = relationship(
        "Article", back_populates="predictions", lazy="selectin"
    )
    explanation: Mapped[Optional["Explanation"]] = relationship(
        "Explanation",
        back_populates="prediction",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Explanation(Base):
    __tablename__ = "explanations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid, index=True)
    prediction_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_breakdown_text: Mapped[str] = mapped_column(Text, nullable=False)
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_model_name: Mapped[str] = mapped_column(String(128), default="offline", nullable=False)

    # Relationships
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="explanation")
