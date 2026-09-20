"""Integration tests for database models, relationships, and transactions."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.entities import (
    Article,
    Claim,
    CredibilityState,
    Evidence,
    Prediction,
    Source,
    StanceType,
)


@pytest.mark.asyncio
async def test_article_claim_evidence_persistence(db_session: AsyncSession):
    """Verify entity insertion, foreign key linking, and cascading relationships."""
    # 1. Create Article
    article = Article(
        title="Breaking: Mars Discoveries Confirmed",
        body_text="Scientific team announces findings from the rover exploratory mission today.",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        author="Research Desk",
        publisher="Science Journal",
    )
    db_session.add(article)
    await db_session.flush()

    assert article.id is not None

    # 2. Add Claim
    claim = Claim(
        article_id=article.id,
        claim_text="Scientific team announces findings from rover mission.",
        source_sentence="Scientific team announces findings from the rover exploratory mission today.",
        start_char=0,
        end_char=75,
        salience_score=0.92,
    )
    db_session.add(claim)
    await db_session.flush()

    # 3. Add Source and Evidence
    source = Source(
        domain="nasa.gov",
        name="NASA JPL",
        historical_reliability_score=0.98,
        is_fact_checker=False,
    )
    db_session.add(source)
    await db_session.flush()

    evidence = Evidence(
        claim_id=claim.id,
        source_id=source.id,
        url="https://nasa.gov/news/rover-update",
        title="NASA Rover Mission Official Statement",
        snippet="NASA rover team published preliminary geologic analysis today.",
        relevance_score=0.95,
        stance=StanceType.SUPPORTS,
        stance_confidence=0.96,
        retrieval_provider="internal_index",
    )
    db_session.add(evidence)

    # 4. Add Prediction
    prediction = Prediction(
        article_id=article.id,
        assessment_state=CredibilityState.LIKELY_TRUE,
        prediction_confidence=0.91,
        calibrated_confidence=0.88,
        evidence_sufficiency=0.92,
        conflict_score=0.02,
        latency_ms=120.5,
    )
    db_session.add(prediction)
    await db_session.commit()

    # 5. Query and verify relations
    query = select(Article).where(Article.id == article.id)
    result = await db_session.execute(query)
    fetched_article = result.scalar_one_or_none()

    assert fetched_article is not None
    assert fetched_article.title == "Breaking: Mars Discoveries Confirmed"
    assert len(fetched_article.claims) == 1
    assert fetched_article.claims[0].claim_text == claim.claim_text
    assert len(fetched_article.claims[0].evidence_items) == 1
    assert fetched_article.claims[0].evidence_items[0].stance == StanceType.SUPPORTS
