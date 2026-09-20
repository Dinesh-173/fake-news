"""Database engine and asynchronous session management with automatic fallback."""

import asyncio
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.core.config import settings
from backend.app.core.logging import logger

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_active_dialect: str = "unknown"


def get_engine() -> AsyncEngine:
    """Returns the singleton async engine or initializes it."""
    global _engine, _session_factory, _active_dialect

    if _engine is None:
        db_url = settings.DATABASE_URL
        try:
            # Attempt to initialize configured engine
            _engine = create_async_engine(
                db_url,
                echo=settings.DB_ECHO,
                future=True,
            )
            _active_dialect = _engine.dialect.name
            logger.info("Initialized database engine with URL dialect: %s", _active_dialect)
        except Exception as exc:
            logger.warning(
                "Failed to initialize primary database engine (%s). Falling back to SQLite: %s",
                exc,
                settings.DATABASE_FALLBACK_SQLITE_URL,
            )
            db_url = settings.DATABASE_FALLBACK_SQLITE_URL
            _engine = create_async_engine(
                db_url,
                echo=settings.DB_ECHO,
                future=True,
            )
            _active_dialect = _engine.dialect.name

        _session_factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Returns the initialized async session factory."""
    global _session_factory
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_connectivity() -> dict:
    """Tests the database connection and returns latency and status information."""
    engine = get_engine()
    start_time = asyncio.get_event_loop().time()
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            row = result.scalar()
            elapsed_ms = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
            if row == 1:
                return {
                    "status": "up",
                    "dialect": engine.dialect.name,
                    "latency_ms": elapsed_ms,
                    "error": None,
                }
            return {
                "status": "degraded",
                "dialect": engine.dialect.name,
                "latency_ms": elapsed_ms,
                "error": "Unexpected query return value",
            }
    except Exception as exc:
        elapsed_ms = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)
        logger.error("Database connectivity check failed: %s", exc)
        return {
            "status": "down",
            "dialect": engine.dialect.name if engine else "unknown",
            "latency_ms": elapsed_ms,
            "error": str(exc),
        }
