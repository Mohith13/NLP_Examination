"""SQLite + ChromaDB storage layer for the BMW Strategic Intelligence Agent."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, EMBEDDING_MODEL, SQLITE_DB_PATH

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_chroma_collection():
    global _chroma_client, _collection
    if _chroma_client is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(name="bmw_strategic_documents")
    return _collection


def init_database(reset: bool = False) -> None:
    """Create required tables. If reset=True, reset SQLite and ChromaDB safely."""
    global _chroma_client, _collection

    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    if reset:
        # Reset SQLite database
        if os.path.exists(SQLITE_DB_PATH):
            try:
                os.remove(SQLITE_DB_PATH)
                print("Removed old SQLite database.")
            except Exception as e:
                print(f"Could not remove SQLite database: {e}")

        # Reset ChromaDB safely by deleting the collection, not the folder.
        try:
            _collection = None
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

            try:
                _chroma_client.delete_collection(name="bmw_strategic_documents")
                print("Removed old ChromaDB collection.")
            except Exception:
                print("No old ChromaDB collection found. Creating fresh one.")

            _collection = _chroma_client.get_or_create_collection(
                name="bmw_strategic_documents"
            )

        except Exception as e:
            print(f"ChromaDB reset warning: {e}")
            print("Trying fallback reset by recreating vector store folder...")

            import time
            old_path = CHROMA_DB_PATH + "_old_" + str(int(time.time()))

            try:
                if os.path.exists(CHROMA_DB_PATH):
                    os.rename(CHROMA_DB_PATH, old_path)
                os.makedirs(CHROMA_DB_PATH, exist_ok=True)
                _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                _collection = _chroma_client.get_or_create_collection(
                    name="bmw_strategic_documents"
                )
                print("Fallback ChromaDB reset completed.")
            except Exception as fallback_error:
                print(f"Fallback reset failed: {fallback_error}")

    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                source TEXT,
                source_type TEXT,
                url TEXT,
                publish_date TEXT,
                collected_at TEXT,
                company TEXT,
                competitor TEXT,
                category TEXT,
                raw_text TEXT,
                clean_text TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                strategic_relevance REAL,
                content_hash TEXT UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS insights (
                insight_id TEXT PRIMARY KEY,
                insight_type TEXT,
                title TEXT,
                description TEXT,
                category TEXT,
                impact_level TEXT,
                severity_level TEXT,
                confidence_score REAL,
                evidence_doc_ids TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                rec_id TEXT PRIMARY KEY,
                strategic_goal TEXT,
                recommendation TEXT,
                priority TEXT,
                expected_impact TEXT,
                risk_level TEXT,
                confidence_score REAL,
                evidence_doc_ids TEXT,
                validation_status TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_publish_date ON documents(publish_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(insight_type)")

    # Ensure Chroma collection exists.
    get_chroma_collection()


def row_to_dict(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def document_exists(content_hash: str, url: str | None = None) -> bool:
    with get_connection() as conn:
        if url:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE content_hash = ? OR url = ? LIMIT 1",
                (content_hash, url),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
    return row is not None


def store_document(doc: Dict[str, Any]) -> Optional[str]:
    """Insert one document into SQLite and ChromaDB. Returns doc_id if inserted."""
    init_database(reset=False)

    content_hash = doc.get("content_hash")
    url = doc.get("url")
    if not content_hash:
        raise ValueError("Document must include content_hash")

    if document_exists(content_hash, url):
        return None

    doc_id = doc.get("doc_id") or str(uuid.uuid4())
    collected_at = doc.get("collected_at") or datetime.utcnow().isoformat(timespec="seconds")
    clean_text = (doc.get("clean_text") or "").strip()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (
                doc_id, title, source, source_type, url, publish_date, collected_at,
                company, competitor, category, raw_text, clean_text, sentiment_label,
                sentiment_score, strategic_relevance, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                doc.get("title", "Untitled"),
                doc.get("source", "Unknown"),
                doc.get("source_type", "unknown"),
                doc.get("url", ""),
                doc.get("publish_date", ""),
                collected_at,
                doc.get("company", "BMW Group"),
                doc.get("competitor", ""),
                doc.get("category", "Uncategorized"),
                doc.get("raw_text", ""),
                clean_text,
                doc.get("sentiment_label", "neutral"),
                float(doc.get("sentiment_score", 0.0)),
                float(doc.get("strategic_relevance", 0.0)),
                content_hash,
            ),
        )

    # Add meaningful text to vector store. Chroma metadata values must be simple types.
    vector_text = clean_text[:6000]
    if vector_text:
        embedding = get_embedding_model().encode(vector_text).tolist()
        metadata = {
            "title": str(doc.get("title", "Untitled"))[:500],
            "source": str(doc.get("source", "Unknown"))[:200],
            "source_type": str(doc.get("source_type", "unknown")),
            "url": str(doc.get("url", ""))[:1000],
            "publish_date": str(doc.get("publish_date", "")),
            "category": str(doc.get("category", "Uncategorized")),
            "sentiment_label": str(doc.get("sentiment_label", "neutral")),
            "strategic_relevance": float(doc.get("strategic_relevance", 0.0)),
        }
        get_chroma_collection().add(
            ids=[doc_id],
            documents=[vector_text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    return doc_id


def fetch_documents(limit: int = 500, where: str = "", params: tuple = ()) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM documents"
    if where:
        sql += f" WHERE {where}"
    sql += " ORDER BY COALESCE(publish_date, collected_at) DESC LIMIT ?"
    with get_connection() as conn:
        rows = conn.execute(sql, (*params, limit)).fetchall()
    return rows_to_dicts(rows)


def get_document_count() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM documents").fetchone()
    return int(row["cnt"])


def get_source_count() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(DISTINCT source) AS cnt FROM documents").fetchone()
    return int(row["cnt"])


def get_source_type_count() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(DISTINCT source_type) AS cnt FROM documents").fetchone()
    return int(row["cnt"])


def get_last_update() -> str:
    with get_connection() as conn:
        row = conn.execute("SELECT MAX(collected_at) AS ts FROM documents").fetchone()
    return row["ts"] or "No data yet"


def get_stats() -> Dict[str, Any]:
    return {
        "document_count": get_document_count(),
        "source_count": get_source_count(),
        "source_type_count": get_source_type_count(),
        "last_update": get_last_update(),
    }


def get_recent_documents(limit: int = 10) -> List[Dict[str, Any]]:
    return fetch_documents(limit=limit)


def get_documents_by_ids(doc_ids: List[str]) -> List[Dict[str, Any]]:
    if not doc_ids:
        return []
    placeholders = ",".join("?" for _ in doc_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        ).fetchall()
    return rows_to_dicts(rows)


def get_sentiment_distribution() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT sentiment_label, COUNT(*) AS count
            FROM documents
            GROUP BY sentiment_label
            ORDER BY count DESC
            """
        ).fetchall()
    return rows_to_dicts(rows)


def get_sentiment_by_category() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT category, sentiment_label, COUNT(*) AS count
            FROM documents
            GROUP BY category, sentiment_label
            ORDER BY category, count DESC
            """
        ).fetchall()
    return rows_to_dicts(rows)


def get_source_type_distribution() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT source_type, COUNT(*) AS count
            FROM documents
            GROUP BY source_type
            ORDER BY count DESC
            """
        ).fetchall()
    return rows_to_dicts(rows)


def search_vector_db(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    if get_document_count() == 0:
        return []

    embedding = get_embedding_model().encode(query).tolist()
    results = get_chroma_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output: List[Dict[str, Any]] = []
    for doc_id, text, meta, distance in zip(ids, docs, metadatas, distances):
        relevance = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0
        item = {
            "doc_id": doc_id,
            "text": text,
            "distance": float(distance) if distance is not None else None,
            "retrieval_relevance": relevance,
        }
        item.update(meta or {})
        output.append(item)
    return output


def clear_insights() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM insights")


def store_insights(insights: List[Dict[str, Any]], replace: bool = True) -> None:
    init_database(reset=False)
    if replace:
        clear_insights()

    with get_connection() as conn:
        for insight in insights:
            conn.execute(
                """
                INSERT OR REPLACE INTO insights (
                    insight_id, insight_type, title, description, category,
                    impact_level, severity_level, confidence_score,
                    evidence_doc_ids, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    insight.get("insight_id") or str(uuid.uuid4()),
                    insight.get("insight_type"),
                    insight.get("title"),
                    insight.get("description"),
                    insight.get("category"),
                    insight.get("impact_level", "Medium"),
                    insight.get("severity_level", "Medium"),
                    float(insight.get("confidence_score", 0.0)),
                    json.dumps(insight.get("evidence_doc_ids", [])),
                    insight.get("created_at") or datetime.utcnow().isoformat(timespec="seconds"),
                ),
            )


def fetch_insights(insight_type: str | None = None, limit: int = 20) -> List[Dict[str, Any]]:
    if insight_type:
        where = "WHERE insight_type = ?"
        params = (insight_type,)
    else:
        where = ""
        params = ()
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM insights
            {where}
            ORDER BY confidence_score DESC, created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    items = rows_to_dicts(rows)
    for item in items:
        try:
            item["evidence_doc_ids"] = json.loads(item.get("evidence_doc_ids") or "[]")
        except json.JSONDecodeError:
            item["evidence_doc_ids"] = []
    return items


def store_recommendation(record: Dict[str, Any]) -> str:
    init_database(reset=False)
    rec_id = record.get("rec_id") or str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO recommendations (
                rec_id, strategic_goal, recommendation, priority, expected_impact,
                risk_level, confidence_score, evidence_doc_ids, validation_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec_id,
                record.get("strategic_goal", ""),
                record.get("recommendation", ""),
                record.get("priority", "Medium"),
                record.get("expected_impact", ""),
                record.get("risk_level", "Medium"),
                float(record.get("confidence_score", 0.0)),
                json.dumps(record.get("evidence_doc_ids", [])),
                record.get("validation_status", "Unchecked"),
                record.get("created_at") or datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )
    return rec_id


def fetch_recommendations(limit: int = 10) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM recommendations
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    items = rows_to_dicts(rows)
    for item in items:
        try:
            item["evidence_doc_ids"] = json.loads(item.get("evidence_doc_ids") or "[]")
        except json.JSONDecodeError:
            item["evidence_doc_ids"] = []
    return items


# Create tables when imported so Streamlit does not fail on first launch.
init_database(reset=False)
