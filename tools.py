"""Tool layer used by the AI CEO Agent.

The point of this file is to show that the agent is not only Prompt -> LLM -> Answer.
It can call explicit tools for retrieval, competitors, sentiment, risks, opportunities,
and evidence validation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from database import (
    fetch_documents,
    fetch_insights,
    get_documents_by_ids,
    get_recent_documents,
    get_sentiment_distribution,
    get_source_type_distribution,
    search_vector_db,
)
from validator import validate_recommendation


def get_recent_news(limit: int = 8) -> List[Dict[str, Any]]:
    return get_recent_documents(limit=limit)


def search_strategic_documents(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    semantic_results = search_vector_db(query, top_k=top_k)
    # Enrich vector results with full SQLite metadata where possible.
    ids = [item["doc_id"] for item in semantic_results]
    full_docs = {doc["doc_id"]: doc for doc in get_documents_by_ids(ids)}
    output = []
    for item in semantic_results:
        full = full_docs.get(item["doc_id"], {})
        merged = dict(full)
        merged.update(item)
        output.append(merged)
    return output


def get_documents_by_category(category: str, limit: int = 8) -> List[Dict[str, Any]]:
    return fetch_documents(limit=limit, where="category LIKE ?", params=(f"%{category}%",))


def get_risk_signals(limit: int = 8) -> List[Dict[str, Any]]:
    return fetch_insights("risk", limit=limit)


def get_opportunity_signals(limit: int = 8) -> List[Dict[str, Any]]:
    return fetch_insights("opportunity", limit=limit)


def get_trend_signals(limit: int = 8) -> List[Dict[str, Any]]:
    return fetch_insights("trend", limit=limit)


def get_competitor_activity(limit: int = 10) -> List[Dict[str, Any]]:
    return fetch_documents(
        limit=limit,
        where="source_type LIKE ? OR category LIKE ? OR competitor != ''",
        params=("%competitor%", "%Competitor%"),
    )


def get_sentiment_summary() -> Dict[str, Any]:
    return {
        "sentiment_distribution": get_sentiment_distribution(),
        "source_type_distribution": get_source_type_distribution(),
    }


def validate_evidence_sources(
    strategic_goal: str,
    recommendation: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return validate_recommendation(strategic_goal, recommendation, evidence)


TOOLS = {
    "recent_news": get_recent_news,
    "semantic_search": search_strategic_documents,
    "category_search": get_documents_by_category,
    "risk_signals": get_risk_signals,
    "opportunity_signals": get_opportunity_signals,
    "trend_signals": get_trend_signals,
    "competitor_activity": get_competitor_activity,
    "sentiment_summary": get_sentiment_summary,
    "evidence_validation": validate_evidence_sources,
}


def list_tools() -> List[str]:
    return sorted(TOOLS.keys())
