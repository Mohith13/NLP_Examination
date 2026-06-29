"""Prompt templates used by the BMW AI CEO Agent."""

from __future__ import annotations

from typing import Any, Dict, List

from config import COMPANY_NAME, COMPETITORS, INDUSTRY, STRATEGIC_THEMES


def make_planner_prompt(goal: str) -> str:
    return f"""
You are the planning module of an AI CEO Strategic Intelligence Agent for {COMPANY_NAME}.
Company industry: {INDUSTRY}.
Competitors to consider: {', '.join(COMPETITORS)}.
Strategic themes: {', '.join(STRATEGIC_THEMES)}.

User strategic goal:
{goal}

Create a concise execution plan with 4 to 6 steps. The plan must include:
1. Which evidence to retrieve.
2. Which risks/opportunities/trends to analyze.
3. Which competitors or market signals to check.
4. How to validate before recommending.

Return only numbered steps.
""".strip()


def format_evidence(evidence: List[Dict[str, Any]], max_chars: int = 900) -> str:
    lines = []
    for i, item in enumerate(evidence, start=1):
        title = item.get("title", "Untitled")
        source = item.get("source", "Unknown")
        source_type = item.get("source_type", "unknown")
        date = item.get("publish_date") or item.get("collected_at") or "unknown date"
        category = item.get("category", "unknown category")
        sentiment = item.get("sentiment_label", "neutral")
        text = item.get("clean_text") or item.get("text") or ""
        snippet = text[:max_chars].replace("\n", " ")
        lines.append(
            f"Evidence {i}:\n"
            f"Title: {title}\n"
            f"Source: {source} ({source_type})\n"
            f"Date: {date}\n"
            f"Category: {category}\n"
            f"Sentiment: {sentiment}\n"
            f"Snippet: {snippet}\n"
        )
    return "\n".join(lines)


def make_analysis_prompt(goal: str, evidence: List[Dict[str, Any]]) -> str:
    return f"""
You are a strategic intelligence analyst advising the CEO of {COMPANY_NAME}.

Strategic goal:
{goal}

Evidence collected:
{format_evidence(evidence)}

Analyze the evidence under these headings:
1. What happened?
2. Why it matters for BMW.
3. Key risks.
4. Key opportunities.
5. Emerging trends.
6. Strategic options.

Be precise. Do not invent facts outside the evidence.
""".strip()


def make_recommendation_prompt(goal: str, analysis: str, evidence: List[Dict[str, Any]]) -> str:
    return f"""
You are the AI CEO Strategic Intelligence Agent for {COMPANY_NAME}.

Strategic goal:
{goal}

Prior analysis:
{analysis}

Evidence:
{format_evidence(evidence, max_chars=500)}

Produce an executive recommendation in this exact format:

CEO BRIEFING

What happened?
...

Why does it matter?
...

Strategic recommendation:
...

Supporting evidence:
1. [source/date/title] ...
2. [source/date/title] ...
3. [source/date/title] ...

Expected impact:
- ...
- ...
- ...

Risk assessment:
- Financial risk: Low/Medium/High + reason
- Operational risk: Low/Medium/High + reason
- Strategic risk: Low/Medium/High + reason

Priority:
High/Medium/Low

Do not include unsupported claims. Every recommendation must clearly connect to the evidence.
""".strip()
