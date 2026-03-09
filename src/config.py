"""Application configuration via environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    default_model: str = Field(default="gpt-4o-mini", description="Default LiteLLM model identifier")
    fallback_model: str = Field(default="gpt-3.5-turbo", description="Fallback model on primary failure")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=1)
    llm_timeout: int = Field(default=120, description="LLM call timeout in seconds")
    llm_max_retries: int = Field(default=3, ge=0)

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    tavily_api_key: str = Field(default="", description="Tavily search API key")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    use_fakeredis: bool = Field(default=True, description="Use fakeredis for local development")

    # ChromaDB
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_persist_dir: str = Field(default="./data/chroma")
    chroma_use_local: bool = Field(default=True, description="Use local persistent ChromaDB")

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)

    # API Server
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    api_workers: int = Field(default=1, ge=1)
    cors_origins: list[str] = Field(default=["*"])
    rate_limit: str = Field(default="30/minute")

    # Research defaults
    default_depth: str = Field(default="standard")
    max_revision_cycles: int = Field(default=2, ge=0)
    critique_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings singleton."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (useful in tests)."""
    global _settings  # noqa: PLW0603
    _settings = None
