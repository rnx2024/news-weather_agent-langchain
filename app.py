import streamlit as st
from agent_service import run_agent

st.set_page_config(page_title="Weather + News", page_icon="🛰")
st.markdown("""
<style>
.block-container { max-width: 820px; margin: auto; padding-top: 1rem; }
.dim { opacity:.8; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("🛰 Weather + News")

LOCATIONS = [
    "Vigan City","Laoag City","Candon City","San Fernando City, La Union",
    "Dagupan City","Lingayen City","Manila","Cebu City","Davao City",
    "Baguio City","Texas","India",
]

place = st.selectbox("Choose location", LOCATIONS, index=0)

if st.button("Get Updates", type="primary"):
    p = (place or "").strip()
    if not p:
        st.warning("Select a location.")
        st.stop()

    with st.spinner("Fetching..."):
        result = run_agent(p)

    # Single clean output. No traces, no toggles.
    st.write(result.get("final", "No output."))
