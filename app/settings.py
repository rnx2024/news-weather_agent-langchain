# settings.py
from __future__ import annotations
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API keys
    api_key: str
    openrouter_api_key: str
    openweather_api_key: str
    serp_api_key: str
    tavily_api: str
    ors_api: str

    # LLM config
    openrouter_base_url: str
    openrouter_model: str
    openrouter_temperature: float

    # External API base URLs
    openweather_current_url: str
    openmeteo_geocode_url: str
    openmeteo_forecast_url: str
    serpapi_search_url: str
    tavily_search_url: str
    ors_directions_url: str
    frontend_cors_origin: str

    # Database / cache
    redis_url: str
    session_secret: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("*", mode="before")
    @classmethod
    def _strip_required_strings(cls, value):
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("Empty environment values are not allowed")
            return cleaned
        return value


settings = Settings()
