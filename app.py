import streamlit as st
import json
from src.agent import analyze_strategic_query

# Configure the page layout
st.set_page_config(page_title="BMW Strategic AI CEO", layout="wide", page_icon="🚙")

st.title("🚙 BMW Group Executive Intelligence Dashboard")

# Safely load the article count for the sidebar
try:
    with open('data/raw_articles.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
        doc_count = len(articles)
except FileNotFoundError:
    doc_count = 0
    articles = []

# --- Section 1: Company Overview ---
st.sidebar.header("Section 1: Company Overview")
st.sidebar.write("**Company:** BMW Group")
st.sidebar.write("**Industry:** Premium Automotive Manufacturing")
st.sidebar.write(f"**Documents Indexed:** {doc_count}")
st.sidebar.write("**Data Sources:** Financial & Tech News RSS")
st.sidebar.caption("Status: Systems Online 🟢")

# Create tabs for the dashboard interface
tab1, tab2, tab3, tab4 = st.tabs([
    "CEO Briefing & Recs", 
    "Opportunity Monitor", 
    "Risk Monitor", 
    "Market Intelligence"
])

with tab1:
    # --- Section 6 & 7: Recommendations and CEO Briefing ---
    st.header("Executive Briefing & Strategic Recommendations")
    st.write("Generate a high-level executive summary based on the latest vector data.")
    
    if st.button("Generate Executive Briefing", type="primary"):
        with st.spinner("Synthesizing corporate strategy..."):
            insight, sources = analyze_strategic_query("corporate strategy")
            st.write(insight)

with tab2:
    # --- Section 3: Opportunity Monitor ---
    st.header("Opportunity Monitor")
    if st.button("Scan Market Opportunities"):
        with st.spinner("Querying ChromaDB for expansion trends..."):
            insights, sources = analyze_strategic_query("opportunities")
            st.write(insights)
            if sources:
                st.subheader("Supporting Evidence Sources:")
                for source in sources:
                    st.caption(f"• {source['title']} ({source['source']})")

with tab3:
    # --- Section 4: Risk Monitor ---
    st.header("Risk Monitor")
    if st.button("Scan Strategic Risks"):
        with st.spinner("Querying ChromaDB for market threats..."):
            insights, sources = analyze_strategic_query("risks")
            st.write(insights)
            if sources:
                st.subheader("Supporting Evidence Sources:")
                for source in sources:
                    st.caption(f"• {source['title']} ({source['source']})")

with tab4:
    # --- Section 2 & 5: Market Intelligence and Sentiment ---
    st.header("Market Intelligence Feed")
    st.write("Recent headlines actively shaping the vector database:")
    
    if articles:
        # Display the 5 most recent articles
        for article in articles[:5]: 
            st.markdown(f"**{article['title']}**")
            st.caption(f"Publisher: {article['source']} | Date: {article['date']}")
            st.divider()
    else:
        st.warning("No news data available. Please run the scraper pipeline.")