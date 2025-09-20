📄 Updated README (README.md)
# 🛰 Weather + News Agent (Modularized)

This Streamlit app fetches weather + news for a city and summarizes them using a LangChain ReAct agent powered by OpenAI.

---

## Features
- **Weather**: Uses OpenWeather API for current conditions.
- **News**: Uses SerpAPI (Google News engine) for top 5 headlines.
- **Agent Reasoning**: LangChain + OpenAI agent decides when to call tools (`get_weather`, `get_news`).
- **Summary**: Produces a single-paragraph natural language brief.
- **UI Options**:
  - Toggle between *direct pipeline* and *agent reasoning*.
  - Optional summary toggle.

---

## Modularized Code Layout


app.py # Streamlit UI
agent_service.py # LangChain ReAct agent with tools
http_utils.py # Shared HTTP request + retry helper
llm_service.py # LLM summarizer for direct pipeline
news_service.py # SerpAPI news fetcher
weather_service.py # OpenWeather fetcher
settings.py # Secret key loader via st.secrets
requirements.txt # Python dependencies
.streamlit/secrets.toml.example # Example config for keys


---

## Requirements

Install dependencies:

```
pip install -r requirements.txt
```

Main libraries:
```
streamlit
requests
langchain
langchain-openai
openai
``` 
## Setup

1. Clone this repo
```
git clone https://github.com/rnx2024/news-weather_agent-langchain.git
cd news-weather_agent-langchain
```

2. Add API keys
Create .streamlit/secrets.toml (not tracked by Git) with:
```
OPENAI_API_KEY = "your-openai-api-key"
OPENWEATHER_API_KEY = "your-openweather-api-key"
SERPAPI_API_KEY = "your-serpapi-api-key"
```

3. (Optional) Change locations
In app.py:
```
LOCATIONS = ["Vigan City", "Laoag City", "Dagupan City", "Manila", "Cebu City", "Davao City"]
```

4. Run the app
```
streamlit run app.py
```
### Usage

Open the UI in your browser (default: http://localhost:8501
).

1. Select a location.
2. Click Get Updates.
3. Toggle between direct summary or agent reasoning.
4. Read weather, news headlines, and AI-generated summary.

### Notes

1. Weather may vary if multiple cities share the same name.
2. Free SerpAPI has limited requests/day.
3. OpenAI API usage may incur costs depending on your plan.