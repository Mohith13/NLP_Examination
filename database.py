import os
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from config import SQLITE_DB_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL

def init_databases():
    """
    Initializes both the structured SQLite database and the semantic ChromaDB vector store.
    This creates the hybrid storage architecture.
    """
    # 1. Initialize SQLite (For exact metadata: dates, URLs, sources)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bmw_documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            url TEXT,
            publish_date TEXT,
            raw_text TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    
    # 2. Initialize ChromaDB (For meaning-based semantic search)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    chroma_client.get_or_create_collection(
        name="bmw_strategic_intelligence",
        embedding_function=ef
    )
    print("✔ Hybrid databases (SQLite + ChromaDB) initialized successfully.")

def store_document(doc_id, title, source, url, publish_date, raw_text):
    """
    Saves a single document into BOTH databases simultaneously using a shared doc_id.
    """
    # Save to SQLite
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO bmw_documents (doc_id, title, source, url, publish_date, raw_text)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, title, source, url, publish_date, raw_text))
        conn.commit()
    except Exception as e:
        print(f"SQLite Error: {e}")
    finally:
        conn.close()

    # Save to ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        collection = chroma_client.get_or_create_collection(
            name="bmw_strategic_intelligence",
            embedding_function=ef
        )
        
        # We embed a combination of the title and text for better search accuracy
        combined_text = f"Title: {title}\nContent: {raw_text}"
        
        collection.upsert(
            documents=[combined_text],
            metadatas=[{"source": source, "date": publish_date, "url": url}],
            ids=[doc_id]
        )
    except Exception as e:
        print(f"ChromaDB Error: {e}")