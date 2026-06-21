import json
import chromadb
from chromadb.utils import embedding_functions

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into smaller chunks with overlap.
    This prevents the LLM from losing context if a sentence is cut off.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_vector_db():
    print("Connecting to local ChromaDB...")
    # This will create a local folder inside your data directory
    chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
    
    # Using the required open-source embedding model from the exam rubric
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    collection = chroma_client.get_or_create_collection(
        name="bmw_intelligence", 
        embedding_function=sentence_transformer_ef
    )
    
    print("Loading raw BMW articles...")
    with open('data/raw_articles.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
        
    documents = []
    metadatas = []
    ids = []
    
    print("Chunking text and generating embeddings... This may take a minute.")
    doc_id_counter = 0
    
    for article in articles:
        # Skip empty content
        if not article['content'] or article['content'] == "No Content":
            continue
            
        # Break the article down into digestible overlapping chunks
        chunks = chunk_text(article['content'])
        
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({"title": article['title'], "source": article['source']})
            ids.append(f"bmw_chunk_{doc_id_counter}")
            doc_id_counter += 1
            
    # Add to the vector database in batches to protect your local PC's memory
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"Indexed {min(i+batch_size, len(documents))}/{len(documents)} vector chunks...")
        
    print("\n✅ BMW Vector Knowledge Base built successfully!")

def retrieve_context(query, n_results=5):
    """Searches the database for the most relevant text chunks."""
    chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = chroma_client.get_collection("bmw_intelligence")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results['documents'] and len(results['documents'][0]) > 0:
        return results['documents'][0], results['metadatas'][0]
    return [], []

if __name__ == "__main__":
    build_vector_db()