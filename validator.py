"""Evidence validation and confidence scoring for CEO recommendations."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from dateutil import parser

from config import VALIDATION_WEIGHTS


def _safe_parse_date(value: str | None):
    if not value:
        return None
    try:
        return parser.parse(value).replace(tzinfo=None)
    except Exception:
        return None


def _recency_score(evidence: List[Dict[str, Any]]) -> float:
    if not evidence:
        return 0.0
    now = datetime.utcnow()
    scores = []
    for item in evidence:
        dt = _safe_parse_date(item.get("publish_date") or item.get("collected_at"))
        if not dt:
            scores.append(0.45)
            continue
        age = now - dt
        if age <= timedelta(days=30):
            scores.append(1.0)
        elif age <= timedelta(days=90):
            scores.append(0.8)
        elif age <= timedelta(days=180):
            scores.append(0.6)
        elif age <= timedelta(days=365):
            scores.append(0.4)
        else:
            scores.append(0.2)
    return sum(scores) / len(scores)


def _source_diversity_score(evidence: List[Dict[str, Any]]) -> float:
    if not evidence:
        return 0.0
    source_types = {item.get("source_type", "unknown") for item in evidence}
    sources = {item.get("source", "unknown") for item in evidence}
    # Full score if evidence spans at least 3 source types or 4 concrete sources.
    return min(1.0, max(len(source_types) / 3.0, len(sources) / 4.0))


def _evidence_count_score(evidence: List[Dict[str, Any]]) -> float:
    return min(1.0, len(evidence) / 5.0)


def _category_match_score(goal: str, recommendation: str, evidence: List[Dict[str, Any]]) -> float:
    text = f"{goal} {recommendation}".lower()
    if not evidence:
        return 0.0
    matches = 0
    for item in evidence:
        category = str(item.get("category", "")).lower()
        title = str(item.get("title", "")).lower()
        if category and category in text:
            matches += 1
        elif any(word in title for word in text.split() if len(word) > 4):
            matches += 1
    return min(1.0, matches / max(1, len(evidence)))


def _sentiment_consistency_score(evidence: List[Dict[str, Any]]) -> float:
    if not evidence:
        return 0.0
    labels = [item.get("sentiment_label", "neutral") for item in evidence]
    if not labels:
        return 0.0
    majority = max(labels.count(label) for label in set(labels))
    return majority / len(labels)


def validate_recommendation(
    strategic_goal: str,
    recommendation: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return validation status and weighted confidence score."""
    weights = VALIDATION_WEIGHTS

    source_div = _source_diversity_score(evidence)
    ev_count = _evidence_count_score(evidence)
    recency = _recency_score(evidence)
    category_match = _category_match_score(strategic_goal, recommendation, evidence)
    sentiment_consistency = _sentiment_consistency_score(evidence)

    confidence = (
        weights.source_diversity * source_div
        + weights.evidence_count * ev_count
        + weights.recency * recency
        + weights.category_match * category_match
        + weights.sentiment_consistency * sentiment_consistency
    )
    confidence_percent = round(confidence * 100, 1)

    source_types = sorted({item.get("source_type", "unknown") for item in evidence})
    evidence_count = len(evidence)

    checks = {
        "minimum_3_evidence_items": evidence_count >= 3,
        "minimum_2_source_types": len(source_types) >= 2,
        "non_empty_recommendation": len(recommendation.strip()) >= 50,
        "confidence_above_60": confidence_percent >= 60,
    }
    passed = all(checks.values())

    if passed:
        reason = "Recommendation passed evidence, source diversity, recency, and confidence checks."
        status = "PASSED"
    else:
        failed = [name for name, ok in checks.items() if not ok]
        reason = "Recommendation needs improvement. Failed checks: " + ", ".join(failed)
        status = "NEEDS_REVIEW"

    return {
        "validation_status": status,
        "confidence_score": confidence_percent,
        "source_types": source_types,
        "evidence_count": evidence_count,
        "checks": checks,
        "reason": reason,
        "component_scores": {
            "source_diversity": round(source_div * 100, 1),
            "evidence_count": round(ev_count * 100, 1),
            "recency": round(recency * 100, 1),
            "category_match": round(category_match * 100, 1),
            "sentiment_consistency": round(sentiment_consistency * 100, 1),
        },
    }
