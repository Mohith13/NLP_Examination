"""Lightweight intelligence layer: cleaning, sentiment, categories, risks, opportunities, trends."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import COMPETITORS, STRATEGIC_THEMES

_SENTIMENT = SentimentIntensityAnalyzer()

CATEGORY_KEYWORDS = {
    "EV Transition": ["electric vehicle", "ev", "bev", "battery", "charging", "neue klasse", "electrification"],
    "China Competition": ["china", "chinese", "byd", "price war", "tariff", "asia"],
    "Software & AI": ["software-defined", "software defined", "digital", "autonomous", "ai", "connected car", "adas"],
    "Regulation": ["regulation", "emission", "eu", "tariff", "policy", "compliance", "ban"],
    "Financial Performance": ["profit", "margin", "revenue", "sales", "earnings", "forecast", "guidance"],
    "Supply Chain": ["supply chain", "semiconductor", "battery material", "lithium", "shortage", "supplier"],
    "Competitor Activity": ["tesla", "mercedes", "volkswagen", "vw", "byd", "audi"],
    "Brand & Product": ["premium", "luxury", "model", "launch", "x3", "i4", "i5", "ix", "mini", "rolls-royce"],
}

RISK_KEYWORDS = {
    "Competitive Risk": ["competition", "price war", "market share", "rival", "byd", "tesla", "pressure"],
    "Regulatory Risk": ["regulation", "tariff", "emission", "ban", "compliance", "investigation"],
    "Supply Chain Risk": ["shortage", "supply chain", "battery material", "semiconductor", "dependency"],
    "Financial Risk": ["profit warning", "decline", "loss", "margin pressure", "weak demand", "slump"],
    "Technology Risk": ["software delay", "cybersecurity", "autonomous", "recall", "defect"],
    "Reputation Risk": ["recall", "complaint", "criticism", "lawsuit", "controversy"],
}

OPPORTUNITY_KEYWORDS = {
    "EV Growth": ["electric vehicle", "bev", "charging", "battery", "neue klasse", "electrification"],
    "Technology Leadership": ["ai", "software-defined", "autonomous", "digital", "connected car"],
    "Partnership": ["partnership", "collaboration", "joint venture", "alliance"],
    "Market Expansion": ["growth", "expansion", "new market", "launch", "increase"],
    "Premium Differentiation": ["premium", "luxury", "performance", "brand", "customer experience"],
    "Operational Efficiency": ["cost reduction", "efficiency", "production", "platform", "manufacturing"],
}

TREND_KEYWORDS = {
    "Software-defined vehicles": ["software-defined", "software defined", "connected car", "digital cockpit"],
    "Battery localization": ["battery", "lithium", "supply chain", "gigafactory", "cell"],
    "China EV price pressure": ["china", "byd", "price war", "tariff", "chinese ev"],
    "Autonomous and AI mobility": ["autonomous", "ai", "driver assistance", "adas"],
    "Premium EV positioning": ["premium", "luxury", "neue klasse", "electric sedan", "electric suv"],
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ").strip()
    return text


def make_hash(text: str) -> str:
    return hashlib.sha256(clean_text(text).lower().encode("utf-8")).hexdigest()


def keyword_score(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lowered)


def classify_category(text: str) -> Tuple[str, float]:
    scores = {category: keyword_score(text, kws) for category, kws in CATEGORY_KEYWORDS.items()}
    best_category, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score == 0:
        return "General BMW Intelligence", 0.35
    confidence = min(0.95, 0.45 + best_score * 0.12)
    return best_category, confidence


def analyze_sentiment(text: str) -> Tuple[str, float]:
    scores = _SENTIMENT.polarity_scores(text[:5000])
    compound = float(scores["compound"])
    if compound >= 0.05:
        return "positive", compound
    if compound <= -0.05:
        return "negative", compound
    return "neutral", compound


def detect_competitor(text: str) -> str:
    lowered = text.lower()
    matches = [c for c in COMPETITORS if c.lower() in lowered]
    return ", ".join(matches)


def strategic_relevance(text: str) -> float:
    lowered = text.lower()
    score = 0
    for theme in STRATEGIC_THEMES:
        if theme.lower() in lowered:
            score += 1
    for comp in COMPETITORS:
        if comp.lower() in lowered:
            score += 1
    if "bmw" in lowered:
        score += 2
    return min(1.0, score / 8.0)


def enrich_document(raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and enrich one raw scraped document with category/sentiment/relevance metadata."""
    raw_text = raw_doc.get("raw_text") or raw_doc.get("summary") or ""
    clean = clean_text(raw_text)
    title = clean_text(raw_doc.get("title", "Untitled"))
    combined = f"{title}. {clean}"

    category, category_confidence = classify_category(combined)
    sentiment_label, sentiment_score = analyze_sentiment(combined)
    competitor = detect_competitor(combined)
    relevance = max(strategic_relevance(combined), category_confidence * 0.65)

    enriched = dict(raw_doc)
    enriched.update(
        {
            "title": title or "Untitled",
            "raw_text": raw_text,
            "clean_text": clean,
            "category": category,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "strategic_relevance": round(relevance, 3),
            "competitor": competitor,
            "content_hash": make_hash(f"{title} {clean}"),
        }
    )
    return enriched


def impact_from_count(count: int) -> str:
    if count >= 7:
        return "High"
    if count >= 3:
        return "Medium"
    return "Low"


def confidence_from_count(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    base = min(0.95, 0.45 + (count / max(total, 1)) * 1.2)
    if count >= 5:
        base += 0.08
    return round(min(base, 0.95), 2)


def _matched_doc_ids(docs: List[Dict[str, Any]], keywords: Iterable[str], limit: int = 6) -> List[str]:
    ids: List[str] = []
    for doc in docs:
        text = f"{doc.get('title', '')} {doc.get('clean_text', '')}".lower()
        if any(kw.lower() in text for kw in keywords):
            ids.append(doc["doc_id"])
        if len(ids) >= limit:
            break
    return ids


def detect_risk_signals(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = len(docs)
    insights: List[Dict[str, Any]] = []

    for category, keywords in RISK_KEYWORDS.items():
        evidence_ids = _matched_doc_ids(docs, keywords)
        if not evidence_ids:
            continue
        count = len(evidence_ids)
        severity = impact_from_count(count)
        insights.append(
            {
                "insight_id": str(uuid.uuid4()),
                "insight_type": "risk",
                "title": category,
                "description": f"Detected {count} evidence items suggesting {category.lower()} for BMW's strategic environment.",
                "category": category,
                "impact_level": "Medium",
                "severity_level": severity,
                "confidence_score": confidence_from_count(count, total),
                "evidence_doc_ids": evidence_ids,
                "created_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )
    return sorted(insights, key=lambda x: x["confidence_score"], reverse=True)


def detect_opportunity_signals(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = len(docs)
    insights: List[Dict[str, Any]] = []

    for category, keywords in OPPORTUNITY_KEYWORDS.items():
        evidence_ids = _matched_doc_ids(docs, keywords)
        if not evidence_ids:
            continue
        count = len(evidence_ids)
        impact = impact_from_count(count)
        insights.append(
            {
                "insight_id": str(uuid.uuid4()),
                "insight_type": "opportunity",
                "title": category,
                "description": f"Detected {count} evidence items suggesting an opportunity around {category.lower()}.",
                "category": category,
                "impact_level": impact,
                "severity_level": "Low",
                "confidence_score": confidence_from_count(count, total),
                "evidence_doc_ids": evidence_ids,
                "created_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )
    return sorted(insights, key=lambda x: x["confidence_score"], reverse=True)


def detect_trends(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = len(docs)
    insights: List[Dict[str, Any]] = []

    for trend, keywords in TREND_KEYWORDS.items():
        evidence_ids = _matched_doc_ids(docs, keywords)
        if not evidence_ids:
            continue
        count = len(evidence_ids)
        insights.append(
            {
                "insight_id": str(uuid.uuid4()),
                "insight_type": "trend",
                "title": trend,
                "description": f"The collected intelligence shows repeated mentions of {trend.lower()}, supported by {count} evidence items.",
                "category": trend,
                "impact_level": impact_from_count(count),
                "severity_level": "Medium",
                "confidence_score": confidence_from_count(count, total),
                "evidence_doc_ids": evidence_ids,
                "created_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )
    return sorted(insights, key=lambda x: x["confidence_score"], reverse=True)


def generate_all_insights(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate risk, opportunity, and trend insights from stored documents."""
    return detect_risk_signals(docs) + detect_opportunity_signals(docs) + detect_trends(docs)


def summarize_categories(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(doc.get("category", "Unknown") for doc in docs))


def summarize_sentiment(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(doc.get("sentiment_label", "neutral") for doc in docs))
