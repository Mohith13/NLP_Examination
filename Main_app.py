"""Streamlit Executive Intelligence Dashboard for BMW AI CEO Agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import run_agent
from config import COMPANY_NAME, INDUSTRY, MIN_DOCUMENTS_REQUIRED
from database import (
    fetch_documents,
    fetch_insights,
    fetch_recommendations,
    get_documents_by_ids,
    get_sentiment_by_category,
    get_sentiment_distribution,
    get_source_type_distribution,
    get_stats,
    store_insights,
)
from intelligence import generate_all_insights
from scraper import run_ingestion

st.set_page_config(
    page_title="BMW AI CEO Strategic Intelligence Agent",
    page_icon="🚘",
    layout="wide",
)


def df_or_empty(records: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def show_status_badge(label: str, value: str) -> None:
    st.markdown(f"**{label}:** `{value}`")


def refresh_insights() -> Dict[str, Any]:
    docs = fetch_documents(limit=1000)
    insights = generate_all_insights(docs)
    store_insights(insights, replace=True)
    return {"documents": len(docs), "insights": len(insights)}


def show_evidence_cards(evidence: List[Dict[str, Any]]) -> None:
    if not evidence:
        st.info("No evidence available yet.")
        return
    for item in evidence:
        with st.expander(item.get("title", "Untitled")):
            st.write(f"**Source:** {item.get('source', 'Unknown')} | **Type:** {item.get('source_type', 'unknown')}")
            st.write(f"**Date:** {item.get('publish_date') or item.get('collected_at', 'unknown')}")
            st.write(f"**Category:** {item.get('category', 'unknown')} | **Sentiment:** {item.get('sentiment_label', 'neutral')}")
            url = item.get("url")
            if url:
                st.markdown(f"[Open source]({url})")
            text = item.get("clean_text") or item.get("text") or ""
            st.write(text[:1200] + ("..." if len(text) > 1200 else ""))


st.title("🚘 BMW AI CEO: Strategic Intelligence Agent")
st.caption("Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate")

with st.sidebar:
    st.header("Control Center")
    st.write("Use these controls during the demo.")

    if st.button("1. Run Live Data Collection", use_container_width=True):
        with st.spinner("Collecting BMW intelligence from live public sources..."):
            result = run_ingestion(reset=False)
        st.success("Ingestion finished")
        st.json(result)

    if st.button("2. Generate Risk/Opportunity/Trend Insights", use_container_width=True):
        with st.spinner("Analyzing collected documents..."):
            result = refresh_insights()
        st.success("Insights generated")
        st.json(result)

    if st.button("3. Refresh Dashboard", use_container_width=True):
        st.rerun()

    # st.divider()
    # st.markdown("**Exam demo sequence**")
    # st.markdown("1. Run collection\n2. Generate insights\n3. Ask CEO question\n4. Show Agent Trace")

stats = get_stats()

tabs = st.tabs(
    [
        "Company Overview",
        "Market Intelligence",
        "Competitor Intelligence",
        "Opportunity Monitor",
        "Risk Monitor",
        "Sentiment Analysis",
        "Strategic Recommendations",
        "CEO Briefing",
        "Agent Trace",
    ]
)

with tabs[0]:
    st.subheader("Company Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Company", COMPANY_NAME)
    col2.metric("Documents", stats["document_count"])
    col3.metric("Sources", stats["source_count"])
    col4.metric("Source Types", stats["source_type_count"])

    st.write(f"**Industry:** {INDUSTRY}")
    st.write(f"**Last update:** {stats['last_update']}")

    if stats["document_count"] < MIN_DOCUMENTS_REQUIRED:
        st.warning(
            f"You currently have {stats['document_count']} documents. "
            f"The project requirement needs at least {MIN_DOCUMENTS_REQUIRED}. Run data collection until this is reached."
        )
    else:
        st.success("Minimum document requirement reached.")

    source_df = df_or_empty(get_source_type_distribution())
    if not source_df.empty:
        fig = px.bar(source_df, x="source_type", y="count", title="Documents by Source Type")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Market Intelligence")
    docs = fetch_documents(limit=30)
    if not docs:
        st.info("No documents yet. Run live data collection from the sidebar.")
    else:
        table = pd.DataFrame(docs)[["title", "source", "source_type", "publish_date", "category", "sentiment_label", "url"]]
        st.dataframe(table, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Competitor Intelligence")
    competitor_docs = fetch_documents(
        limit=30,
        where="competitor != '' OR source_type LIKE ? OR category LIKE ?",
        params=("%competitor%", "%Competitor%"),
    )
    if not competitor_docs:
        st.info("No competitor records yet. Run collection and insights generation.")
    else:
        table = pd.DataFrame(competitor_docs)[["title", "competitor", "source", "publish_date", "category", "sentiment_label", "url"]]
        st.dataframe(table, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Opportunity Monitor")
    opportunities = fetch_insights("opportunity", limit=20)
    if not opportunities:
        st.info("No opportunity insights yet. Click 'Generate Risk/Opportunity/Trend Insights'.")
    else:
        df = pd.DataFrame(opportunities)
        st.dataframe(
            df[["title", "description", "impact_level", "confidence_score", "evidence_doc_ids"]],
            use_container_width=True,
            hide_index=True,
        )

with tabs[4]:
    st.subheader("Risk Monitor")
    risks = fetch_insights("risk", limit=20)
    if not risks:
        st.info("No risk insights yet. Click 'Generate Risk/Opportunity/Trend Insights'.")
    else:
        df = pd.DataFrame(risks)
        st.dataframe(
            df[["title", "description", "severity_level", "confidence_score", "evidence_doc_ids"]],
            use_container_width=True,
            hide_index=True,
        )

with tabs[5]:
    st.subheader("Sentiment Analysis")
    sentiment_df = df_or_empty(get_sentiment_distribution())
    if sentiment_df.empty:
        st.info("No sentiment data yet.")
    else:
        col1, col2 = st.columns(2)
        fig = px.pie(sentiment_df, names="sentiment_label", values="count", title="Overall News/Public Sentiment")
        col1.plotly_chart(fig, use_container_width=True)

        by_cat = df_or_empty(get_sentiment_by_category())
        if not by_cat.empty:
            fig2 = px.bar(
                by_cat,
                x="category",
                y="count",
                color="sentiment_label",
                title="Sentiment by Strategic Category",
            )
            col2.plotly_chart(fig2, use_container_width=True)

with tabs[6]:
    st.subheader("Stored Strategic Recommendations")
    recs = fetch_recommendations(limit=10)
    if not recs:
        st.info("No recommendations stored yet. Run a CEO briefing question.")
    else:
        for rec in recs:
            with st.expander(f"{rec.get('validation_status')} | Confidence {rec.get('confidence_score')}% | {rec.get('strategic_goal')}"):
                st.markdown(rec.get("recommendation", ""))
                st.write("**Evidence IDs:**", rec.get("evidence_doc_ids", []))

with tabs[7]:
    st.subheader("CEO Briefing Generator")
    default_goal = "If you were the CEO of BMW today, what would you do next and why?"
    goal = st.text_area(
        "Strategic question",
        value=default_goal,
        height=100,
        help="Try: What should BMW do about Chinese EV competition?",
    )

    if st.button("Generate CEO Briefing", type="primary"):
        with st.spinner("Agent is planning, retrieving, analyzing, deciding, recommending, and validating..."):
            result = run_agent(goal)
        st.session_state["last_agent_result"] = result
        st.success("CEO briefing generated")

    result = st.session_state.get("last_agent_result")
    if result:
        st.markdown(result["recommendation"])
        st.divider()
        st.subheader("Validation")
        validation = result["validation"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", validation["validation_status"])
        col2.metric("Confidence", f"{validation['confidence_score']}%")
        col3.metric("Evidence Count", validation["evidence_count"])
        st.write(validation["reason"])
        st.json(validation["component_scores"])
        st.subheader("Evidence Used")
        show_evidence_cards(result["evidence"])

with tabs[8]:
    st.subheader("Agent Trace")
    result = st.session_state.get("last_agent_result")
    if not result:
        st.info("Generate a CEO briefing first to see the agent trace.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Goal")
            st.write(result["goal"])

            st.markdown("### Plan")
            for i, step in enumerate(result["plan"], start=1):
                st.write(f"{i}. {step}")

            st.markdown("### Tools Used")
            for tool in result["tools_used"]:
                st.write(f"- `{tool}`")

        with col2:
            st.markdown("### Trace")
            for event in result["agent_trace"]:
                st.write(f"✅ {event}")

            st.markdown("### Validation Checks")
            checks = result["validation"].get("checks", {})
            for name, passed in checks.items():
                icon = "✅" if passed else "⚠️"
                st.write(f"{icon} {name}")

        st.markdown("### Analysis")
        st.write(result["analysis"])
