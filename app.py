import streamlit as st
from weather_service import get_weather_line
from news_service import get_news_items
from llm_service import summarize_plain
from agent_service import run_agent

st.set_page_config(page_title="Weather + News (Agent)", page_icon="🛰")
st.markdown("""
<style>
.block-container { max-width: 820px; margin: auto; padding-top: 1rem; }
.dim { opacity:.8; font-size:0.9rem; }
pre, code { white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

st.title("🛰 Weather + News Briefs")

LOCATIONS = [
    "Vigan City","Laoag City","Candon City","San Fernando City, La Union",
    "Dagupan City","Lingayen City","Manila","Cebu City","Davao City",
    "Baguio City","Texas","India",
]

col1, col2, col3 = st.columns([2,1,1])
with col1:
    loc = st.selectbox("Choose location", LOCATIONS, index=0)
with col2:
    do_summary = st.toggle("Summarize", value=True)
with col3:
    use_agent = st.toggle("Use Agent", value=True)

if st.button("Get Updates", type="primary"):
    loc_s = (loc or "").strip()
    if not loc_s:
        st.warning("Select a location."); st.stop()

    if use_agent:
        with st.spinner("Agent reasoning..."):
            out = run_agent(loc_s)
        st.subheader("Agent Output")
        st.write(out["final"])
        with st.expander("Agent trace"):
            st.write(out["trace"])
    else:
        with st.spinner("Fetching..."):
            w_line, w_err = get_weather_line(loc_s)
            headlines, n_err = get_news_items(loc_s)

        if w_err: st.error(w_err)
        if n_err: st.error(n_err)

        st.subheader("Weather")
        st.write(w_line or "n/a")

        st.subheader("News")
        if headlines:
            for h in headlines:
                st.markdown(
                    f"- [{h['title']}]({h['link']})  \n"
                    f"  <span class='dim'>{h['source']} • {h['date']}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.write("No recent news.")

        if do_summary:
            st.subheader("Summary")
            try:
                st.write(summarize_plain(loc_s, w_line, headlines))
            except Exception:
                st.info("Summary unavailable.")
