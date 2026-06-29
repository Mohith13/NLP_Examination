"""Live ingestion pipeline for BMW Strategic Intelligence Agent.

Run:
    python scraper.py
    python scraper.py --reset

This collects RSS items, tries to extract fuller article text, cleans/enriches documents,
deduplicates them, stores metadata in SQLite, and indexes text into ChromaDB.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Dict, List

import feedparser
import requests
import trafilatura
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from config import COMPANY_NAME, MAX_ARTICLE_CHARS, REQUEST_TIMEOUT, RSS_SOURCES, TARGET_DOCUMENTS
from database import get_document_count, init_database, store_document
from intelligence import clean_text, enrich_document

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BMW-AI-CEO-Agent/1.0; academic prototype)"
}


def normalize_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        return date_parser.parse(value).date().isoformat()
    except Exception:
        return ""


def strip_html_summary(summary: str) -> str:
    if not summary:
        return ""
    soup = BeautifulSoup(summary, "html.parser")
    return clean_text(soup.get_text(" "))


def fetch_full_article(url: str) -> str:
    """Try trafilatura first; fallback to requests + BeautifulSoup text."""
    if not url:
        return ""

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and len(extracted.strip()) > 300:
                return clean_text(extracted)[:MAX_ARTICLE_CHARS]
    except Exception:
        pass

    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if response.ok:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            text = soup.get_text(" ")
            text = clean_text(text)
            if len(text) > 300:
                return text[:MAX_ARTICLE_CHARS]
    except Exception:
        pass

    return ""


def parse_feed_source(entry: Any, fallback_name: str) -> str:
    source = getattr(entry, "source", None)
    if isinstance(source, dict):
        return source.get("title") or fallback_name
    return fallback_name


def collect_from_rss(source_config: Dict[str, str], max_items: int = 40) -> List[Dict[str, Any]]:
    feed = feedparser.parse(source_config["url"])
    documents: List[Dict[str, Any]] = []

    for entry in feed.entries[:max_items]:
        title = clean_text(getattr(entry, "title", "Untitled"))
        url = getattr(entry, "link", "")
        published = getattr(entry, "published", "") or getattr(entry, "updated", "")
        summary = strip_html_summary(getattr(entry, "summary", ""))
        source = parse_feed_source(entry, source_config["name"])

        full_text = fetch_full_article(url)
        raw_text = full_text if len(full_text) > len(summary) else summary

        if len(raw_text) < 120:
            continue

        documents.append(
            {
                "title": title,
                "source": source,
                "source_type": source_config["source_type"],
                "url": url,
                "publish_date": normalize_date(published),
                "collected_at": datetime.utcnow().isoformat(timespec="seconds"),
                "company": COMPANY_NAME,
                "raw_text": raw_text,
            }
        )

    return documents


def run_ingestion(reset: bool = False, target_documents: int = TARGET_DOCUMENTS) -> Dict[str, Any]:
    init_database(reset=reset)

    attempted = 0
    inserted = 0
    failed_sources: List[str] = []

    for source_config in RSS_SOURCES:
        try:
            docs = collect_from_rss(source_config, max_items=45)
        except Exception as exc:
            failed_sources.append(f"{source_config['name']}: {exc}")
            continue

        for raw_doc in docs:
            attempted += 1
            try:
                enriched = enrich_document(raw_doc)
                doc_id = store_document(enriched)
                if doc_id:
                    inserted += 1
            except Exception as exc:
                print(f"[WARN] Failed to store document: {exc}")

        if get_document_count() >= target_documents:
            break

    total = get_document_count()
    return {
        "attempted": attempted,
        "inserted": inserted,
        "total_documents": total,
        "target_documents": target_documents,
        "failed_sources": failed_sources,
        "status": "OK" if total >= 100 else "NEEDS_MORE_DATA",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BMW intelligence ingestion pipeline.")
    parser.add_argument("--reset", action="store_true", help="Delete existing database/vector store first.")
    parser.add_argument("--target", type=int, default=TARGET_DOCUMENTS, help="Target document count.")
    args = parser.parse_args()

    result = run_ingestion(reset=args.reset, target_documents=args.target)
    print("\nBMW Strategic Intelligence Ingestion Result")
    print("=" * 52)
    for key, value in result.items():
        print(f"{key}: {value}")
    if result["total_documents"] < 100:
        print("\nTip: Run again later or add more RSS sources in config.py.")


if __name__ == "__main__":
    main()
