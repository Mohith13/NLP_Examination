import streamlit as st
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from src.agent import analyze_strategic_query

st.set_page_config(page_title="BMW Strategic AI CEO", layout="wide", page_icon="🚙")
st.title("🚙 BMW Executive Intelligence Dashboard")

# Load FinBERT from Hugging Face (Runs locally on your RTX A6000)
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    # Move model to the RTX A6000 GPU
    if torch.cuda.is_available():
        model = model.to('cuda')
    return tokenizer, model

tokenizer, model = load_sentiment_model()

def get_finbert_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # FinBERT labels: 0 -> Positive, 1 -> Negative, 2 -> Neutral
    probs = prediction.cpu().numpy()[0]
    score = probs[0] - probs[1] # Net sentiment calculation
    return score

# Load and prepare data
try:
    with open('data/raw_articles.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
        df = pd.DataFrame(articles)
except FileNotFoundError:
    articles, df = [], pd.DataFrame()

# --- Section 1: Company Overview ---
st.sidebar.header("Company Overview")
st.sidebar.write("**Company:** BMW Group")
st.sidebar.write("**Industry:** Premium Automotive")
st.sidebar.write(f"**Documents Indexed:** {len(articles)}")
st.sidebar.write("**Data Sources:** Financial News RSS, Google Feeds")
if not df.empty:
    st.sidebar.write(f"**Last Update:** {df['date'].iloc[0]}")

# Layout tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "CEO Briefing & Recs", 
    "Opportunity Monitor", 
    "Risk Monitor", 
    "Sentiment Analysis",
    "Market Intel"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.header("CEO Briefing")
        if st.button("Generate Briefing", type="primary"):
            with st.spinner("Compiling briefing via Llama3..."):
                brief, sources = analyze_strategic_query("ceo_briefing")
                st.write(brief)
    with col2:
        st.header("Strategic Recommendations")
        if st.button("Generate Recommendations"):
            with st.spinner("Generating strategy..."):
                rec, sources = analyze_strategic_query("recommendations")
                st.write(rec)

with tab2:
    st.header("Opportunity Monitor")
    if st.button("Scan Opportunities"):
        with st.spinner("Querying ChromaDB..."):
            opps, sources = analyze_strategic_query("opportunities")
            st.write(opps)

with tab3:
    st.header("Risk Monitor")
    if st.button("Scan Risks"):
        with st.spinner("Querying ChromaDB..."):
            risks, sources = analyze_strategic_query("risks")
            st.write(risks)

with tab4:
    st.header("Sentiment Analysis")
    if not df.empty:
        if st.button("Run GPU Sentiment Analysis"):
            with st.spinner("Analyzing text polarity via FinBERT Transformer..."):
                df['Sentiment_Score'] = df['title'].apply(get_finbert_sentiment)
                
                df['Type'] = 'Public Sentiment'
                df.loc[df['source'].str.contains("News|Reuters|Bloomberg", case=False, na=False), 'Type'] = 'News Sentiment'
                
                avg_news = df[df['Type'] == 'News Sentiment']['Sentiment_Score'].mean()
                avg_pub = df[df['Type'] == 'Public Sentiment']['Sentiment_Score'].mean()
                
                col1, col2 = st.columns(2)
                col1.metric("Transformer News Sentiment", f"{avg_news:.2f}")
                col2.metric("Transformer Public Sentiment", f"{avg_pub:.2f}")
                
                # --- NEW PIE CHART CODE STARTS HERE ---
                st.subheader("Sentiment Distribution")
                
                # Categorize the raw FinBERT scores for the Pie Chart
                def categorize_sentiment(score):
                    if score > 0.05: return 'Positive'
                    elif score < -0.05: return 'Negative'
                    else: return 'Neutral'
                
                df['Sentiment_Category'] = df['Sentiment_Score'].apply(categorize_sentiment)
                sentiment_counts = df['Sentiment_Category'].value_counts()
                
                # Draw the Pie Chart
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_alpha(0.0) # Transparent background for dark mode
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
                       colors=['#4ade80', '#94a3b8', '#f87171'], textprops={'color':"w"})
                st.pyplot(fig)
                # --- NEW PIE CHART CODE ENDS HERE ---

                st.subheader("Sentiment Trends across Sources")
                trend_data = df.groupby('source')['Sentiment_Score'].mean().head(15)
                st.bar_chart(trend_data)
    else:
        st.warning("No data available.")

with tab5:
    st.header("Market Intelligence")
    if articles:
        st.subheader("Emerging Technologies & Competitor Activities")
        for article in articles[:5]:
            st.markdown(f"**{article['title']}**")
            st.caption(f"Source: {article['source']} | Date: {article['date']}")
            st.divider()