"""Intelligence layer: cleaning, FinBERT sentiment, categories, risks, opportunities, trends."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Tuple

from transformers import pipeline

from config import COMPETITORS, STRATEGIC_THEMES


FINBERT_MODEL_NAME = "ProsusAI/finbert"
_FINBERT_PIPELINE = None


CATEGORY_KEYWORDS = {
    "EV Transition": [
        "electric vehicle",
        "ev",
        "bev",
        "battery",
        "charging",
        "neue klasse",
        "electrification",
    ],
    "China Competition": [
        "china",
        "chinese",
        "byd",
        "price war",
        "tariff",
        "asia",
    ],
    "Software & AI": [
        "software-defined",
        "software defined",
        "digital",
        "autonomous",
        "ai",
        "connected car",
        "adas",
    ],
    "Regulation": [
        "regulation",
        "emission",
        "eu",
        "tariff",
        "policy",
        "compliance",
        "ban",
    ],
    "Financial Performance": [
        "profit",
        "margin",
        "revenue",
        "sales",
        "earnings",
        "forecast",
        "guidance",
    ],
    "Supply Chain": [
        "supply chain",
        "semiconductor",
        "battery material",
        "lithium",
        "shortage",
        "supplier",
    ],
    "Competitor Activity": [
        "tesla",
        "mercedes",
        "volkswagen",
        "vw",
        "byd",
        "audi",
    ],
    "Brand & Product": [
        "premium",
        "luxury",
        "model",
        "launch",
        "x3",
        "i4",
        "i5",
        "ix",
        "mini",
        "rolls-royce",
    ],
}


RISK_KEYWORDS = {
    "Competitive Risk": [
        "competition",
        "price war",
        "market share",
        "rival",
        "byd",
        "tesla",
        "pressure",
        "challenged",
        "lost ground",
    ],
    "Regulatory Risk": [
        "regulation",
        "tariff",
        "emission",
        "ban",
        "compliance",
        "investigation",
    ],
    "Supply Chain Risk": [
        "shortage",
        "supply chain",
        "battery material",
        "semiconductor",
        "dependency",
    ],
    "Financial Risk": [
        "profit warning",
        "decline",
        "loss",
        "margin pressure",
        "weak demand",
        "slump",
        "guidance cut",
        "headwinds",
    ],
    "Technology Risk": [
        "software delay",
        "cybersecurity",
        "autonomous",
        "recall",
        "defect",
    ],
    "Reputation Risk": [
        "recall",
        "complaint",
        "criticism",
        "lawsuit",
        "controversy",
    ],
}


OPPORTUNITY_KEYWORDS = {
    "EV Growth": [
        "electric vehicle",
        "bev",
        "charging",
        "battery",
        "neue klasse",
        "electrification",
    ],
    "Technology Leadership": [
        "ai",
        "software-defined",
        "autonomous",
        "digital",
        "connected car",
    ],
    "Partnership": [
        "partnership",
        "collaboration",
        "joint venture",
        "alliance",
        "cooperate",
        "cooperation",
    ],
    "Market Expansion": [
        "growth",
        "expansion",
        "new market",
        "launch",
        "increase",
    ],
    "Premium Differentiation": [
        "premium",
        "luxury",
        "performance",
        "brand",
        "customer experience",
    ],
    "Operational Efficiency": [
        "cost reduction",
        "efficiency",
        "production",
        "platform",
        "manufacturing",
    ],
}


TREND_KEYWORDS = {
    "Software-defined vehicles": [
        "software-defined",
        "software defined",
        "connected car",
        "digital cockpit",
    ],
    "Battery localization": [
        "battery",
        "lithium",
        "supply chain",
        "gigafactory",
        "cell",
    ],
    "China EV price pressure": [
        "china",
        "byd",
        "price war",
        "tariff",
        "chinese ev",
    ],
    "Autonomous and AI mobility": [
        "autonomous",
        "ai",
        "driver assistance",
        "adas",
    ],
    "Premium EV positioning": [
        "premium",
        "luxury",
        "neue klasse",
        "electric sedan",
        "electric suv",
    ],
}


def now_utc_iso() -> str:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def get_finbert_pipeline():
    """
    Load FinBERT only once and reuse it.

    FinBERT is used because the project analyzes business, financial,
    market, and company news. It is more suitable than generic sentiment
    tools for CEO-level strategic intelligence.
    """
    global _FINBERT_PIPELINE

    if _FINBERT_PIPELINE is None:
        device = -1

        try:
            import torch

            if torch.cuda.is_available():
                device = 0
        except Exception:
            device = -1

        _FINBERT_PIPELINE = pipeline(
            "text-classification",
            model=FINBERT_MODEL_NAME,
            tokenizer=FINBERT_MODEL_NAME,
            device=device,
        )

    return _FINBERT_PIPELINE


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def make_hash(text: str) -> str:
    return hashlib.sha256(clean_text(text).lower().encode("utf-8")).hexdigest()


def keyword_score(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def classify_category(text: str) -> Tuple[str, float]:
    scores = {
        category: keyword_score(text, keywords)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }

    best_category, best_score = max(scores.items(), key=lambda item: item[1])

    if best_score == 0:
        return "General BMW Intelligence", 0.35

    confidence = min(0.95, 0.45 + best_score * 0.12)
    return best_category, confidence


def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Financial/business sentiment analysis using FinBERT plus strategic calibration.

    FinBERT gives the base financial sentiment.
    The calibration layer adjusts only when the text clearly contains business
    risk or opportunity signals relevant to CEO-level intelligence.
    """
    if not text or not text.strip():
        return "neutral", 0.0

    classifier = get_finbert_pipeline()
    sample_text = clean_text(text)
    sample_text = sample_text[:3000]
    lowered = sample_text.lower()

    try:
        try:
            result = classifier(
                sample_text,
                truncation=True,
                max_length=512,
                return_all_scores=True,
            )
        except TypeError:
            result = classifier(
                sample_text,
                truncation=True,
                max_length=512,
                top_k=None,
            )

        if isinstance(result, list) and result and isinstance(result[0], list):
            scores = result[0]
        else:
            scores = result

        score_map = {
            item["label"].lower(): float(item["score"])
            for item in scores
        }

        finbert_label = max(score_map, key=score_map.get)
        finbert_confidence = score_map[finbert_label]

        positive_prob = score_map.get("positive", 0.0)
        negative_prob = score_map.get("negative", 0.0)
        neutral_prob = score_map.get("neutral", 0.0)

        negative_terms = [
            "headwind", "headwinds", "guidance cut", "cuts guidance",
            "delay recovery", "delayed recovery", "decline", "declines",
            "loss", "losses", "lost ground", "margin pressure",
            "price pressure", "price war", "weak demand", "slowdown",
            "slump", "tariff", "tariffs", "risk", "threat", "challenged",
            "challenge", "competitive pressure", "pressure",
            "underperform", "recall", "lawsuit", "shortage", "warning",
            "falls", "drops", "cut", "misses", "struggle", "struggling"
        ]

        strong_negative_terms = [
            "guidance cut", "cuts guidance", "headwinds", "price war",
            "margin pressure", "weak demand", "lost ground",
            "delay recovery", "delayed recovery"
        ]

        positive_terms = [
            "growth", "market share gain", "record", "boost", "increase",
            "increases", "jump", "jumps", "jumped", "profit growth",
            "strong demand", "outperforming", "partnership", "cooperation",
            "collaboration", "innovation", "innovative", "launch",
            "expansion", "achieves", "success", "wins", "improves",
            "improvement", "new model", "new platform", "range",
            "real-world range", "800-volt", "investment", "invests",
            "sales rise", "sales growth", "premium", "leadership"
        ]

        strong_positive_terms = [
            "record", "strong demand", "outperforming", "market share gain",
            "profit growth", "sales growth", "sales rise", "boost",
            "partnership", "cooperation", "innovation", "innovative",
            "800-volt", "real-world range", "expansion"
        ]

        negative_hits = sum(1 for term in negative_terms if term in lowered)
        strong_negative_hits = sum(1 for term in strong_negative_terms if term in lowered)

        positive_hits = sum(1 for term in positive_terms if term in lowered)
        strong_positive_hits = sum(1 for term in strong_positive_terms if term in lowered)

        # 1. Strong negative business signals override neutral FinBERT.
        if strong_negative_hits >= 1:
            score = min(0.95, 0.62 + strong_negative_hits * 0.10 + negative_hits * 0.03)
            return "negative", round(-score, 3)

        # 2. Trust clear FinBERT negative result.
        if finbert_label == "negative" and finbert_confidence >= 0.50:
            return "negative", round(-finbert_confidence, 3)

        # 3. Strong positive business signals override neutral FinBERT
        # only when there is no clear risk signal.
        if strong_positive_hits >= 1 and negative_hits == 0:
            score = min(0.95, 0.60 + strong_positive_hits * 0.08 + positive_hits * 0.02)
            return "positive", round(score, 3)

        # 4. Several positive signals with no risk should be positive.
        if positive_hits >= 2 and negative_hits == 0:
            score = min(0.90, 0.55 + positive_hits * 0.05 + positive_prob * 0.15)
            return "positive", round(score, 3)

        # 5. Several risk signals with no opportunity signal should be negative.
        if negative_hits >= 2 and positive_hits == 0:
            score = min(0.90, 0.55 + negative_hits * 0.05 + negative_prob * 0.15)
            return "negative", round(-score, 3)

        # 6. Mixed signals are neutral because strategy news is often ambiguous.
        if negative_hits >= 1 and positive_hits >= 1:
            return "neutral", round(max(neutral_prob, finbert_confidence), 3)

        # 7. Trust clear FinBERT positive result.
        if finbert_label == "positive" and finbert_confidence >= 0.55:
            return "positive", round(finbert_confidence, 3)

        # 8. Fall back to FinBERT.
        if finbert_label == "positive":
            return "positive", round(finbert_confidence, 3)

        if finbert_label == "negative":
            return "negative", round(-finbert_confidence, 3)

        return "neutral", round(finbert_confidence, 3)

    except Exception as error:
        print(f"FinBERT sentiment failed. Falling back to neutral. Error: {error}")
        return "neutral", 0.0
def detect_competitor(text: str) -> str:
    lowered = text.lower()
    matches = [competitor for competitor in COMPETITORS if competitor.lower() in lowered]
    return ", ".join(matches)


def strategic_relevance(text: str) -> float:
    lowered = text.lower()
    score = 0

    for theme in STRATEGIC_THEMES:
        if theme.lower() in lowered:
            score += 1

    for competitor in COMPETITORS:
        if competitor.lower() in lowered:
            score += 1

    if "bmw" in lowered:
        score += 2

    return min(1.0, score / 8.0)


def enrich_document(raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and enrich one raw scraped document with:
    - category
    - FinBERT sentiment
    - competitor mention
    - strategic relevance
    - content hash for deduplication
    """
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


def _matched_doc_ids(
    docs: List[Dict[str, Any]],
    keywords: Iterable[str],
    limit: int = 6,
) -> List[str]:
    ids: List[str] = []

    for doc in docs:
        text = f"{doc.get('title', '')} {doc.get('clean_text', '')}".lower()

        if any(keyword.lower() in text for keyword in keywords):
            doc_id = doc.get("doc_id")
            if doc_id:
                ids.append(doc_id)

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
                "description": (
                    f"Detected {count} evidence items suggesting "
                    f"{category.lower()} for BMW's strategic environment."
                ),
                "category": category,
                "impact_level": "Medium",
                "severity_level": severity,
                "confidence_score": confidence_from_count(count, total),
                "evidence_doc_ids": evidence_ids,
                "created_at": now_utc_iso(),
            }
        )

    return sorted(insights, key=lambda item: item["confidence_score"], reverse=True)


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
                "description": (
                    f"Detected {count} evidence items suggesting "
                    f"an opportunity around {category.lower()}."
                ),
                "category": category,
                "impact_level": impact,
                "severity_level": "Low",
                "confidence_score": confidence_from_count(count, total),
                "evidence_doc_ids": evidence_ids,
                "created_at": now_utc_iso(),
            }
        )

    return sorted(insights, key=lambda item: item["confidence_score"], reverse=True)


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
                "description": (
                    f"The collected intelligence shows repeated mentions of "
                    f"{trend.lower()}, supported by {count} evidence items."
                ),
                "category": trend,
                "impact_level": impact_from_count(count),
                "severity_level": "Medium",
                "confidence_score": confidence_from_count(count, total),
                "evidence_doc_ids": evidence_ids,
                "created_at": now_utc_iso(),
            }
        )

    return sorted(insights, key=lambda item: item["confidence_score"], reverse=True)


def generate_all_insights(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate risk, opportunity, and trend insights from stored documents."""
    return (
        detect_risk_signals(docs)
        + detect_opportunity_signals(docs)
        + detect_trends(docs)
    )


def summarize_categories(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(doc.get("category", "Unknown") for doc in docs))


def summarize_sentiment(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(doc.get("sentiment_label", "neutral") for doc in docs))