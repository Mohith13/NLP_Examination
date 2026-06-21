# 🚙 BMW Group Strategic Intelligence Agent

## Project Overview
This project is an AI-powered Strategic Intelligence Agent designed to continuously collect live public data, analyze developments, and generate evidence-based, executive-level recommendations for the BMW Group. The system acts as a CEO Advisor, transforming unstructured market information into structured strategic insights.

---

## 1. System Architecture Diagram
The architecture is designed entirely for local, private execution to ensure data security and eliminate reliance on commercial APIs.

```mermaid
graph TD
    subgraph Frontend
        A[Streamlit Dashboard] -->|User Input| B(App Logic)
    end
    
    subgraph Core AI Pipeline
        B -->|Queries| C{Ollama LLM Agent}
        B -->|Retrieval| D[(ChromaDB Local Vector Store)]
        D -->|Context Chunks| C
    end
    
    subgraph Data Ingestion
        E[RSS News Feeds] -->|Raw XML| F[Python Scraper]
        F -->|JSON| G[Text Chunker & Overlap]
        G -->|Embeddings| D
    end
    
    C -->|JSON/Markdown Response| A