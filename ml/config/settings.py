"""Configuration settings for the TNTU Assistant AI Agent."""

from typing import List
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Tavily settings
    TAVILY_API_KEY: str = Field(..., description="Tavily API key for web search functionality")

    # OpenAI settings
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key for language model access")
    DEFAULT_LLM: str = Field(default="gpt-5.1", description="Default LLM model to use")
    DEFAULT_LLM_TEMPERATURE: float = Field(default=1.0, ge=0.0, le=2.0, description="Default temperature for LLM responses")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="OpenAI embedding model to use")

    # Allowed LLM models
    ALLOWED_LLM_MODELS: List[str] = Field(
        default=["gpt-5.1", "gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"],
        description="List of allowed LLM model names",
    )

    # Agent settings
    AGENT_MAX_TOKENS: int = Field(default=16000, ge=1, description="Maximum tokens for agent responses")
    DOCUMENT_MAX_TOKENS: int = Field(default=1000, ge=1, description="Maximum tokens for document content")

    # Application settings
    APP_NAME: str = Field(default="TNTU Assistant AI", description="Application name")
    APP_VERSION: str = Field(default="0.1.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Enable debug mode")

    # PostgreSQL Database settings
    POSTGRES_HOST: str = Field(..., description="PostgreSQL database host")
    POSTGRES_PORT: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL database port")
    POSTGRES_DATABASE: str = Field(..., description="PostgreSQL database name")
    POSTGRES_USER: str = Field(..., description="PostgreSQL database user")
    POSTGRES_PASSWORD: str = Field(..., description="PostgreSQL database password")
    POSTGRES_SSL_MODE: str = Field(default="prefer", description="PostgreSQL SSL mode")
    POSTGRES_SCHEMA: str = Field(default="public", description="PostgreSQL schema name")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    def get_postgres_connection_string(self, async_driver: bool = False) -> str:
        """
        Generate PostgreSQL connection string.

        Args:
            async_driver: If True, returns asyncpg connection string, otherwise psycopg2

        Returns:
            Database connection string
        """
        driver = "postgresql+asyncpg" if async_driver else "postgresql+psycopg2"

        conn_string = (
            f"{driver}://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DATABASE}"
        )

        if self.POSTGRES_SSL_MODE != "disable":
            conn_string += f"?sslmode={self.POSTGRES_SSL_MODE}"

        return conn_string


# Global settings instance
settings = Settings()
