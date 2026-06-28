import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from config import SQLITE_DB_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL

def get_chroma_collection():
    """Connects to the ChromaDB vector store."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return chroma_client.get_or_create_collection(
        name="bmw_strategic_intelligence", 
        embedding_function=ef
    )

def search_strategic_documents(query: str, n_results: int = 5) -> str:
    """
    TOOL: Searches the vector database for documents related to the query's meaning.
    The AI Agent will use this to find specific risks, opportunities, or evidence.
    """
    try:
        collection = get_chroma_collection()
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant documents found for this query."
            
        formatted_results = []
        for i in range(len(results['documents'][0])):
            doc_text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            formatted_results.append(
                f"Source: {metadata.get('source', 'Unknown')} | "
                f"Date: {metadata.get('date', 'Unknown')}\n"
                f"Content: {doc_text}\n---"
            )
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Database error during search: {str(e)}"

def get_recent_news(limit: int = 3) -> str:
    """
    TOOL: Retrieves the absolute most recent news articles from the SQLite database.
    The AI Agent will use this to understand the current immediate timeline.
    """
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Pull the most recent entries based on publication date
        cursor.execute('''
            SELECT title, source, publish_date, raw_text 
            FROM bmw_documents 
            ORDER BY publish_date DESC 
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "No recent news found. The database might be empty."
            
        formatted_results = []
        for row in rows:
            title, source, date, text = row
            # Provide a truncated summary to save AI token space
            formatted_results.append(
                f"Title: {title}\nSource: {source} | Date: {date}\n"
                f"Summary: {text[:250]}...\n---"
            )
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"SQLite error retrieving news: {str(e)}"