# TripBites Backend

FastAPI backend for the current TripBites app.

This service powers a city-based assistant that combines weather data, recent news, and AI-generated responses into a single backend API. It is currently designed around chat, weather, and news retrieval, with a LangGraph ReAct agent orchestrating tool usage and summary generation. :contentReference[oaicite:0]{index=0}

A separate frontend can call this backend for:
- conversational city insights
- weather summaries
- recent local news

## Current Scope

The backend currently provides:
- a FastAPI API service
- a LangGraph ReAct agent for `/chat`
- weather retrieval and summarization
- news retrieval and filtering
- API key protection on endpoints
- retry and rate-limiting support in the tool layer :contentReference[oaicite:1]{index=1}

This README reflects the backend as it is currently designed. It does not describe future TripBites travel features yet.

## Main Features

### 1. Weather intelligence

The backend uses weather providers to collect forecast and weather-condition data, then converts that into a concise human-readable summary. The current design includes automatic geocoding fallback and hazard-style classification such as heavy rain, thunderstorm, and extreme heat. :contentReference[oaicite:2]{index=2}

### 2. News retrieval

The backend fetches recent news through SerpAPI using Google News, filters results to the last 7 days, and returns the top recent headlines for a city or place. It also supports country-aware localization. :contentReference[oaicite:3]{index=3}

### 3. LangGraph agent orchestration

The `/chat` flow is powered by a LangGraph ReAct agent. The agent decides when to call tools such as weather, news, and city-risk logic, then produces a final natural-language response. Debug mode can expose tool calls, arguments, and observations. :contentReference[oaicite:4]{index=4}

### 4. Backend safeguards

The current implementation includes token-bucket rate limiting, automatic retries with exponential backoff, and structured tools defined with Pydantic schemas. :contentReference[oaicite:5]{index=5}

### 5. Secure API access

All endpoints currently require an API key through the `x-api-key` header. :contentReference[oaicite:6]{index=6}

## API Endpoints

### `GET /health`
Simple health check endpoint.

### `POST /chat`
Main agent endpoint. Accepts a user query and returns an AI-generated response based on available weather and news context.

### `GET /weather`
Returns a concise weather summary for a requested place.

### `GET /news`
Returns filtered recent news for a requested place. Current behavior is limited to recent headlines within the last 7 days, with up to the top 3 recent items. :contentReference[oaicite:7]{index=7}

## Architecture Summary

Current backend flow:

1. The client sends a request to the backend.
2. For chat requests, the LangGraph agent evaluates the query.
3. The agent decides whether to call weather, news, or city-risk tools.
4. Tool outputs are combined into a final concise response.
5. Dedicated weather and news endpoints are also available for direct frontend use. :contentReference[oaicite:8]{index=8}

## Tech Stack

- FastAPI
- LangGraph
- Pydantic
- SerpAPI
- Open-Meteo
- OpenWeather
- Python
- Docker support in the repository :contentReference[oaicite:9]{index=9}

## Project Purpose

This backend is currently best described as a city insight service that combines:
- weather context
- recent news context
- AI-generated natural-language summaries

It is the backend companion for the current TripBites frontend, even though the backend itself is still functionally centered on weather, news, and chat. That matches the present implementation more accurately than describing it as a full travel-planning system. :contentReference[oaicite:10]{index=10}

## Environment Notes

The repository includes a `Dockerfile`, `pyproject.toml`, and `uv.lock`, which indicates support for containerized setup and Python dependency management through the project configuration. :contentReference[oaicite:11]{index=11}

Document the actual environment variables used by your codebase in this section, for example:
- API keys
- model provider keys
- weather provider keys
- SerpAPI key
- allowed frontend origin
- rate-limit settings

## Local Development

Example development flow:

```bash
uv sync
uvicorn app.main:app --reload
