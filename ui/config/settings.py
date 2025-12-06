"""Configuration settings for the UI."""

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=True)


class Settings(BaseSettings):
    """UI Configuration settings."""

    # API Settings
    API_BASE_URL: str = Field(default="http://tntuassistant-ml:8000/api/v1", description="Base URL of the backend API")

    # Page Configuration
    PAGE_TITLE: str = Field(default="TNTU Assistant", description="Title of the application")
    PAGE_ICON: str = Field(default="🎓", description="Icon of the application")
    LAYOUT: str = Field(default="wide", description="Layout of the application")

    # Chat Settings
    DEFAULT_LLM: str = Field(default="gpt-5.1", description="Default model of the application")
    DEFAULT_LLM_TEMPERATURE: float = Field(default=1.0, ge=0.0, le=2.0, description="Default temperature of the application")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")


settings = Settings()
