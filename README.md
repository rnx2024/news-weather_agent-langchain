# Weather + News Intelligent Agent (FastAPI + LangGraph)

This project provides a production-ready FastAPI backend powered by a LangGraph ReAct agent.
The agent integrates weather, news, and risk estimation into a single intelligent reasoning pipeline.

A separate React frontend can call the `/chat` endpoint for conversational responses.

---

## Features

### 1. Hybrid Weather Intelligence

Uses Open-Meteo for:
- Daily forecast  
- Hourly forecast  
- Weather codes for thunderstorms, fog, extreme heat, wind, snow  

Uses OpenWeather for:
- One-line human-readable weather description (`weather_line`)

Additional capabilities:
- Automatic geocoding with fallback  
- Hazard classification (heavy_rain, thunderstorm, extreme heat, etc.)

---

### 2. Intelligent News Fetching (SerpAPI)

- Uses Google News engine  
- Filters headlines to the last 7 days only  
- Returns the top 3 most recent headlines  
- Supports global localization via dynamic country-code detection

---

### 3. LangGraph-Powered ReAct Agent

The agent:
- Reasons step-by-step  
- Decides when to call tools (`weather_tool`, `news_tool`, `city_risk_tool`)  
- Produces a concise, final natural-language summary  

Debug mode exposes:
- Tool calls  
- Arguments  
- Observations  

---

### 4. Built-in Tooling Infrastructure

- Token-bucket rate limiting  
- Automatic retries with exponential backoff  
- Structured tools using Pydantic schemas  
- City risk assessment combining weather and real-time news patterns  

---

### 5. Secure FastAPI Service

All endpoints require an API key via the `x-api-key` header.

Available endpoints:
- `GET /health` — service health check  
- `POST /chat` — main LangGraph agent endpoint  
- `GET /weather` — one-line weather summary  
- `GET /news` — filtered news (≤7 days, top 3 headlines)  
---
