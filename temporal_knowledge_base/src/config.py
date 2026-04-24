"""CHRONOS configuration — settings, model configs, and environment loading."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class ModelConfig:
    """Training cutoff dates for supported LLMs.

    The training cutoff is the last date the model saw data during training.
    CHRONOS uses this as the lower bound of the retrieval window:
    events between [training_cutoff, simulation_date] are retrieved.
    """

    CUTOFFS: dict[str, date] = {
        "gpt-4o": date(2023, 10, 1),
        "gpt-4.1": date(2024, 6, 1),
        "claude-3.5-sonnet": date(2024, 4, 1),
        "claude-4-sonnet": date(2025, 1, 1),
        "gemini-2.0-flash": date(2024, 6, 1),
        "gemini-2.0-flash-lite": date(2024, 8, 1),
        "gemini-2.5-flash": date(2025, 1, 1),
        "gemini-2.5-pro": date(2025, 1, 1),
        "gemini-3-pro": date(2025, 6, 1),
        "gemini-3-flash": date(2025, 6, 1),
    }

    @classmethod
    def get_cutoff(cls, model_name: str) -> date:
        """Get training cutoff for a model. Raises KeyError if unknown."""
        key = model_name.lower().strip()
        if key not in cls.CUTOFFS:
            raise KeyError(
                f"Unknown model '{model_name}'. "
                f"Known models: {list(cls.CUTOFFS.keys())}. "
                f"Add it to ModelConfig.CUTOFFS with its training cutoff date."
            )
        return cls.CUTOFFS[key]

    @classmethod
    def earliest_cutoff(cls) -> date:
        """The earliest training cutoff across all models — determines how far back data collection must go."""
        return min(cls.CUTOFFS.values())


class DateConfidence(str, Enum):
    """How confident we are in an event's date."""

    VERIFIED = "verified"  # 3+ sources agree
    HIGH = "high"  # 2 sources agree, or authoritative primary source
    MEDIUM = "medium"  # Majority vote but not unanimous
    SINGLE_SOURCE = "single_source"  # 1 source, but date is explicit and logical
    APPROXIMATE = "approximate"  # Known to week/month, not exact day
    UNCERTAIN = "uncertain"  # Sources disagree or date is ambiguous — QUARANTINED


class DatePrecision(str, Enum):
    """Granularity of the event date."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    UNKNOWN = "unknown"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/chronos",
        description="PostgreSQL connection string (asyncpg driver)",
    )

    # Google AI
    google_api_key: str = Field(default="", description="Google AI API key")

    # Web search (Tavily)
    tavily_api_key: str = Field(default="", description="Tavily API key for web search")

    # ACLED (optional structured source)
    acled_api_key: str = Field(default="", description="ACLED API key")

    # Anthropic (optional, for cross-model validation)
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # Collection settings
    collection_subject: str = Field(
        default="Donald J. Trump",
        description="The leader to collect data about",
    )
    collection_start: date = Field(
        default_factory=ModelConfig.earliest_cutoff,
        description="Start of collection window (defaults to earliest model cutoff)",
    )

    # Retrieval settings
    default_top_k: int = Field(default=15, description="Default number of events to retrieve")
    similarity_threshold: float = Field(
        default=0.8, description="Maximum cosine distance for vector search"
    )

    # Embedding
    embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Embedding model for vector search",
    )
    embedding_dimensions: int = Field(default=3072, description="Embedding vector dimensions")

    # Research LLM
    research_model: str = Field(
        default="gemini-2.5-flash",
        description="LLM for research agents (fast, cheap)",
    )

    # Validation LLM
    validation_model: str = Field(
        default="gemini-3-pro",
        description="LLM for temporal validation (higher quality)",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton — gracefully handle environments where .env is unreadable
try:
    settings = Settings()
except PermissionError:
    # Sandbox/CI: .env stat blocked → fall back to env-vars only
    settings = Settings(_env_file=None)
