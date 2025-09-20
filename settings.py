from __future__ import annotations
import streamlit as st

def require_secret(name: str) -> str:
    v = st.secrets.get(name)
    if not v:
        st.error(f"Missing secret: {name}")
        st.stop()
    return str(v)

OPENAI_API_KEY      = require_secret("OPENAI_API_KEY")
OPENWEATHER_API_KEY = require_secret("OPENWEATHER_API_KEY")
SERPAPI_API_KEY     = require_secret("SERPAPI_API_KEY")
