"""Central configuration for the BMW AI CEO Strategic Intelligence Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "bmw_metadata.db")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "bmw_vector_store")

os.makedirs(DATA_DIR, exist_ok=True)

COMPANY_NAME = "BMW Group"
INDUSTRY = "Automotive / Electric Mobility / Premium Vehicles"

# Use a model that is actually available in your Ollama installation.
# Check with: ollama list
LLM_MODEL = os.getenv("BMW_AI_CEO_LLM", "qwen3:8b")
EMBEDDING_MODEL = os.getenv("BMW_AI_CEO_EMBEDDINGS", "all-MiniLM-L6-v2")

MIN_DOCUMENTS_REQUIRED = 100
TARGET_DOCUMENTS = 150
MAX_ARTICLE_CHARS = 7000
REQUEST_TIMEOUT = 15

COMPETITORS = ["Mercedes-Benz", "Volkswagen", "Tesla", "BYD"]

STRATEGIC_THEMES = [
    "electric vehicles",
    "Neue Klasse",
    "battery supply chain",
    "China EV competition",
    "software-defined vehicles",
    "autonomous driving",
    "EU regulation",
    "premium automotive market",
    "charging infrastructure",
    "profit margins",
]

# RSS-based public sources. Google News RSS resolves to many independent publishers,
# while the source_type tells the dashboard what strategic role each feed plays.
RSS_SOURCES = [
    {
        "name": "Google News - BMW Group",
        "url": "https://news.google.com/rss/search?q=BMW%20Group%20when:90d&hl=en-US&gl=US&ceid=US:en",
        "source_type": "market_news",
    },
    {
        "name": "Google News - BMW Electric Vehicles",
        "url": "https://news.google.com/rss/search?q=BMW%20electric%20vehicles%20OR%20BMW%20Neue%20Klasse%20when:90d&hl=en-US&gl=US&ceid=US:en",
        "source_type": "industry_trends",
    },
    {
        "name": "Google News - BMW China Competition",
        "url": "https://news.google.com/rss/search?q=BMW%20China%20EV%20competition%20BYD%20Tesla%20when:90d&hl=en-US&gl=US&ceid=US:en",
        "source_type": "competitor_news",
    },
    {
        "name": "Google News - BMW Software Defined Vehicle",
        "url": "https://news.google.com/rss/search?q=BMW%20software-defined%20vehicle%20autonomous%20AI%20when:180d&hl=en-US&gl=US&ceid=US:en",
        "source_type": "technology_news",
    },
    {
        "name": "Google News - Automotive Battery Supply Chain",
        "url": "https://news.google.com/rss/search?q=BMW%20battery%20supply%20chain%20lithium%20when:180d&hl=en-US&gl=US&ceid=US:en",
        "source_type": "supply_chain_news",
    },
    {
        "name": "Yahoo Finance - BMWYY",
        "url": "https://finance.yahoo.com/rss/headline?s=BMWYY",
        "source_type": "financial_news",
    },
    {
        "name": "Google News - Mercedes Volkswagen Tesla BYD",
        "url": "https://news.google.com/rss/search?q=Mercedes%20Volkswagen%20Tesla%20BYD%20electric%20vehicles%20when:90d&hl=en-US&gl=US&ceid=US:en",
        "source_type": "competitor_news",
    },
]

@dataclass(frozen=True)
class ValidationWeights:
    source_diversity: float = 0.30
    evidence_count: float = 0.25
    recency: float = 0.20
    category_match: float = 0.15
    sentiment_consistency: float = 0.10

VALIDATION_WEIGHTS = ValidationWeights()
