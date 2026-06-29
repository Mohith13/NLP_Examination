"""Agent orchestrator: Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List

import ollama

from config import LLM_MODEL
from database import store_recommendation
from prompts import make_analysis_prompt, make_planner_prompt, make_recommendation_prompt
from tools import (
    get_competitor_activity,
    get_opportunity_signals,
    get_recent_news,
    get_risk_signals,
    get_sentiment_summary,
    get_trend_signals,
    search_strategic_documents,
    validate_evidence_sources,
)


def call_llm(prompt: str, temperature: float = 0.2) -> str:
    """Call local Ollama. If Ollama is unavailable, return a useful fallback message."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        return (
            "[LLM fallback mode] Ollama did not return a response. "
            f"Reason: {exc}\n\n"
            "The pipeline still retrieved evidence and performed validation. "
            "Start Ollama and check `ollama list` if you want full generated briefings."
        )


def default_plan(goal: str) -> List[str]:
    return [
        "Understand the strategic goal and identify BMW themes involved.",
        "Retrieve recent market and company evidence using semantic search.",
        "Check competitor activity if the goal involves Tesla, BYD, Mercedes-Benz, Volkswagen, China, or EV pressure.",
        "Review risk, opportunity, trend, and sentiment signals from the intelligence repository.",
        "Generate an evidence-backed CEO recommendation.",
        "Validate the recommendation using evidence count, source diversity, recency, and confidence score.",
    ]


def parse_plan(text: str, goal: str) -> List[str]:
    if text.startswith("[LLM fallback mode]"):
        return default_plan(goal)
    steps = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[\).:-]\s*", "", line)
        if len(line) > 8:
            steps.append(line)
    return steps[:6] if steps else default_plan(goal)


def select_tools(goal: str) -> List[str]:
    """
    Select tools for the agent.

    For CEO-level strategic questions, we intentionally use multiple tools:
    retrieval, recent news, risk signals, opportunity signals, trend signals,
    competitor activity, sentiment, and validation.

    This makes the system visibly agentic instead of simple RAG.
    """
    lowered = goal.lower()

    tools = [
        "semantic_search",
        "recent_news",
        "sentiment_summary",
    ]

    strategic_question = any(
        term in lowered
        for term in [
            "what should",
            "ceo",
            "strategy",
            "strategic",
            "next 12 months",
            "recommend",
            "competition",
            "market pressure",
            "ev",
            "electric",
            "software",
            "china",
            "byd",
            "tesla",
        ]
    )

    if strategic_question:
        tools.extend([
            "risk_signals",
            "opportunity_signals",
            "trend_signals",
            "competitor_activity",
        ])
    else:
        if any(term in lowered for term in ["risk", "threat", "danger", "regulation", "supply", "china", "tariff"]):
            tools.append("risk_signals")

        if any(term in lowered for term in ["opportunity", "growth", "invest", "market", "product", "technology", "software"]):
            tools.append("opportunity_signals")

        if any(term in lowered for term in ["trend", "future", "emerging", "monitor"]):
            tools.append("trend_signals")

        if any(term in lowered for term in ["competitor", "tesla", "byd", "mercedes", "volkswagen", "vw", "china"]):
            tools.append("competitor_activity")

    tools.append("evidence_validation")

    return list(dict.fromkeys(tools))

def _unique_evidence(items: List[Dict[str, Any]], max_items: int = 9) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for item in items:
        doc_id = item.get("doc_id")
        url = item.get("url")
        key = doc_id or url or item.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= max_items:
            break
    return output

def _goal_keywords(goal: str) -> List[str]:
    """
    Build a focused keyword set from the CEO question.
    This helps the agent keep evidence aligned with the strategic goal.
    """
    lowered = goal.lower()

    keywords = set(
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", lowered)
        if token not in {
            "what", "should", "about", "with", "from", "that", "this",
            "next", "months", "today", "would", "there", "their", "which",
            "because", "company", "group"
        }
    )

    domain_keywords = [
        "bmw", "ev", "electric", "vehicle", "vehicles", "china", "chinese",
        "byd", "tesla", "competition", "competitor", "market", "pressure",
        "software", "defined", "battery", "batteries", "neue", "klasse",
        "autonomous", "digital", "technology"
    ]

    for word in domain_keywords:
        if word in lowered:
            keywords.add(word)

    return list(keywords)


def _evidence_relevance_score(goal: str, item: Dict[str, Any]) -> float:
    """
    Score evidence based on how well it matches the strategic question.
    Higher score means better evidence for the CEO recommendation.
    """
    keywords = _goal_keywords(goal)

    title = str(item.get("title", "")).lower()
    category = str(item.get("category", "")).lower()
    source_type = str(item.get("source_type", "")).lower()
    clean_text = str(item.get("clean_text") or item.get("text") or "").lower()

    blob = f"{title} {category} {source_type} {clean_text[:1200]}"

    score = 0.0

    for keyword in keywords:
        if keyword in title:
            score += 4.0
        if keyword in category:
            score += 2.0
        if keyword in source_type:
            score += 1.5
        if keyword in clean_text[:1200]:
            score += 1.0

    if "competitor" in source_type:
        score += 3.0
    if "technology" in source_type:
        score += 2.0
    if "industry" in source_type:
        score += 2.0

    # Penalize weak evidence for strategic EV/software/China questions.
    weak_terms = [
        "imsa", "racing", "podium", "spa-francorchamps", "weathertech",
        "concours", "eleganza", "classic", "villa d'este",
        "humanoid robot", "figure 03"
    ]

    for term in weak_terms:
        if term in blob:
            score -= 6.0

    return score


def _rank_evidence_for_goal(
    goal: str,
    items: List[Dict[str, Any]],
    max_items: int = 9,
) -> List[Dict[str, Any]]:
    """
    Deduplicate and rank retrieved evidence so the final recommendation uses
    strategically relevant evidence, not merely recent BMW news.
    """
    unique_items = _unique_evidence(items, max_items=40)

    scored = []
    for item in unique_items:
        score = _evidence_relevance_score(goal, item)
        item = dict(item)
        item["agent_relevance_score"] = round(score, 2)
        scored.append(item)

    scored.sort(
        key=lambda x: (
            x.get("agent_relevance_score", 0),
            x.get("retrieval_relevance", 0),
        ),
        reverse=True,
    )

    # Prefer evidence with positive score, but keep fallback if too few.
    strong_items = [item for item in scored if item.get("agent_relevance_score", 0) > 0]

    if len(strong_items) >= 3:
        return strong_items[:max_items]

    return scored[:max_items]


def retrieve_evidence(goal: str, selected_tools: List[str]) -> Dict[str, Any]:
    evidence: List[Dict[str, Any]] = []
    tool_outputs: Dict[str, Any] = {}

    search_queries = [
        goal,
        "BMW China EV competition BYD Tesla market pressure",
        "BMW software-defined vehicle electric vehicle Neue Klasse battery strategy",
    ]

    if "semantic_search" in selected_tools:
        semantic_results = []
        for query in search_queries:
            result = search_strategic_documents(query, top_k=6)
            semantic_results.extend(result)

        tool_outputs["semantic_search"] = semantic_results
        evidence.extend(semantic_results)

    if "competitor_activity" in selected_tools:
        result = get_competitor_activity(limit=10)
        tool_outputs["competitor_activity"] = result
        evidence.extend(result)

    if "recent_news" in selected_tools:
        # Recent news is useful, but we add it after semantic and competitor evidence
        # so it does not dominate the final evidence pack.
        result = get_recent_news(limit=5)
        tool_outputs["recent_news"] = result
        evidence.extend(result)

    if "risk_signals" in selected_tools:
        tool_outputs["risk_signals"] = get_risk_signals(limit=5)

    if "opportunity_signals" in selected_tools:
        tool_outputs["opportunity_signals"] = get_opportunity_signals(limit=5)

    if "trend_signals" in selected_tools:
        tool_outputs["trend_signals"] = get_trend_signals(limit=5)

    if "sentiment_summary" in selected_tools:
        tool_outputs["sentiment_summary"] = get_sentiment_summary()

    ranked_evidence = _rank_evidence_for_goal(goal, evidence, max_items=9)

    return {
        "evidence": ranked_evidence,
        "tool_outputs": tool_outputs,
    }

def fallback_analysis(goal: str, evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "No evidence was found. Run scraper.py first, then generate insights."
    titles = [item.get("title", "Untitled") for item in evidence[:5]]
    return (
        f"The evidence retrieved for '{goal}' indicates that BMW's strategic environment is shaped by "
        "EV transition, competitor pressure, software-defined vehicles, supply-chain factors, and market sentiment. "
        "Key evidence titles include: " + "; ".join(titles) + "."
    )


def fallback_recommendation(goal: str, evidence: List[Dict[str, Any]], validation: Dict[str, Any] | None = None) -> str:
    if not evidence:
        return (
            "CEO BRIEFING\n\n"
            "What happened?\nNo stored evidence is available yet.\n\n"
            "Why does it matter?\nThe agent cannot make a reliable strategic recommendation without collected documents.\n\n"
            "Strategic recommendation:\nRun the ingestion pipeline first and collect at least 100 BMW-related documents.\n\n"
            "Supporting evidence:\nNone available.\n\n"
            "Expected impact:\n- Improves evidence quality\n- Enables risk and opportunity monitoring\n- Makes recommendations defensible\n\n"
            "Risk assessment:\n- Financial risk: Low, this is an implementation step\n- Operational risk: Medium, requires live source availability\n- Strategic risk: High if decisions are made without evidence\n\n"
            "Priority:\nHigh"
        )

    evidence_lines = []
    for i, item in enumerate(evidence[:3], start=1):
        evidence_lines.append(
            f"{i}. [{item.get('source', 'Unknown')}/{item.get('publish_date') or item.get('collected_at', 'unknown date')}/{item.get('title', 'Untitled')}]"
        )

    return (
        "CEO BRIEFING\n\n"
        f"What happened?\nBMW-related evidence was retrieved for the strategic goal: {goal}. The strongest signals point to market pressure, EV transition, competitor movement, and technology shifts.\n\n"
        "Why does it matter?\nThese signals affect BMW's product positioning, capital allocation, technology roadmap, and competitive defense.\n\n"
        "Strategic recommendation:\nBMW should prioritize an evidence-backed response that combines EV product acceleration, software-defined vehicle capability, and competitor monitoring.\n\n"
        "Supporting evidence:\n" + "\n".join(evidence_lines) + "\n\n"
        "Expected impact:\n- Stronger strategic focus\n- Better risk visibility\n- Clearer executive decision priorities\n\n"
        "Risk assessment:\n- Financial risk: Medium, because execution may require investment\n- Operational risk: Medium, because roadmap changes affect teams and suppliers\n- Strategic risk: Medium, because delay could strengthen competitors\n\n"
        "Priority:\nHigh"
    )


def run_agent(strategic_goal: str) -> Dict[str, Any]:
    strategic_goal = strategic_goal.strip()
    if not strategic_goal:
        strategic_goal = "If you were the CEO of BMW today, what would you do next and why?"

    # 1. Plan
    plan_text = call_llm(make_planner_prompt(strategic_goal), temperature=0.1)
    plan = parse_plan(plan_text, strategic_goal)

    # 2. Tool selection
    selected_tools = select_tools(strategic_goal)

    # 3. Retrieval + tool usage
    retrieval = retrieve_evidence(strategic_goal, selected_tools)
    evidence = retrieval["evidence"]
    tool_outputs = retrieval["tool_outputs"]

    # 4. Analyze
    analysis = call_llm(make_analysis_prompt(strategic_goal, evidence), temperature=0.2)
    if analysis.startswith("[LLM fallback mode]"):
        analysis = fallback_analysis(strategic_goal, evidence)

    # 5. Decide + recommend
    recommendation = call_llm(make_recommendation_prompt(strategic_goal, analysis, evidence), temperature=0.2)
    if recommendation.startswith("[LLM fallback mode]"):
        recommendation = fallback_recommendation(strategic_goal, evidence)

    # 6. Validate
    validation = validate_evidence_sources(strategic_goal, recommendation, evidence)

    # 7. Store recommendation
    evidence_doc_ids = [item.get("doc_id") for item in evidence if item.get("doc_id")]
    store_recommendation(
        {
            "strategic_goal": strategic_goal,
            "recommendation": recommendation,
            "priority": "High" if validation["confidence_score"] >= 70 else "Medium",
            "expected_impact": "Executive decision support based on retrieved evidence, risk/opportunity signals, and validation.",
            "risk_level": "Medium",
            "confidence_score": validation["confidence_score"],
            "evidence_doc_ids": evidence_doc_ids,
            "validation_status": validation["validation_status"],
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
    )

    return {
        "goal": strategic_goal,
        "plan": plan,
        "tools_used": selected_tools,
        "tool_outputs": tool_outputs,
        "evidence": evidence,
        "analysis": analysis,
        "recommendation": recommendation,
        "validation": validation,
        "agent_trace": [
            "Goal received",
            "Plan generated",
            "Tools selected",
            f"Evidence retrieved: {len(evidence)} documents",
            "Evidence analyzed",
            "Recommendation generated",
            f"Validation completed: {validation['validation_status']}",
        ],
    }


if __name__ == "__main__":
    result = run_agent("If you were the CEO of BMW today, what would you do next and why?")
    print(result["recommendation"])
    print("\nValidation:", result["validation"])
