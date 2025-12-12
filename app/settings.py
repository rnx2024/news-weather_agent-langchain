# settings.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API keys
    api_key: str
    openrouter_api_key: str
    openweather_api_key: str
    serp_api_key: str 

    # LLM config
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "gpt-4o-mini"
    openrouter_temperature: float = 0.0

    # External API base URLs
    openweather_current_url: str = "https://api.openweathermap.org/data/2.5/weather"
    openmeteo_geocode_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    openmeteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    serpapi_search_url: str = "https://serpapi.com/search.json"
    frontend_cors_origin: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
