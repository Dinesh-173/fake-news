"""Application configuration using Pydantic Settings."""

import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core Application Settings
    PROJECT_NAME: str = "TruthLens"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # CORS Configuration
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, tuple)):
            return [str(i) for i in v]
        return ["*"]

    # Database Configuration
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./truthlens.db",
        description="Async database connection string. Supports postgresql+asyncpg or sqlite+aiosqlite",
    )
    DATABASE_FALLBACK_SQLITE_URL: str = "sqlite+aiosqlite:///./truthlens.db"
    DB_ECHO: bool = False

    # Machine Learning Runtimes & Devices
    MODEL_DEVICE: str = "cpu"
    DEFAULT_RANDOM_SEED: int = 42
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    NLI_MODEL_NAME: str = "roberta-large-mnli"

    # Security & Request Limits
    MAX_ARTICLE_CHARS: int = 50000
    MIN_ARTICLE_CHARS: int = 20
    MAX_CLAIMS_PER_ARTICLE: int = 10
    REQUEST_TIMEOUT_SECONDS: int = 30


settings = Settings()
