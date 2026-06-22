🚙 BMW Strategic Intelligence AI (CEO Advisor)

📌 Project Overview

Organizations today operate in an environment of continuous information flow. This project implements an AI-powered Strategic Intelligence Agent that continuously collects, analyzes, and reasons over live information related to the BMW Group.

It acts as a digital advisor to the CEO, moving beyond standard information retrieval (search) to generate actionable, evidence-based business insights.

🏗️ 1. System Architecture Diagram

The system is built on a highly modular Retrieval-Augmented Generation (RAG) architecture, running on a GPU-accelerated environment.

[ External World ]       [ Data Pipeline ]           [ AI Engine ]             [ User Interface ]
                                                                             
  +-------------+       +------------------+      +-------------------+      +-------------------+
  | RSS Feeds   | ----> | Text Preprocessor|      | Llama 3 (8B) LLM  |      | Streamlit Web App |
  | (News/PR)   |       | & Text Chunker   |      | (Reasoning Agent) |      | (CEO Dashboard)   |
  +-------------+       +--------+---------+      +---------+---------+      +---------+---------+
                                 |                          ^                          ^
                                 v                          |                          |
                        +------------------+      +---------+---------+                |
                        | BAAI/bge-small   | ---> | ChromaDB          | <--------------+
                        | Embedding Model  |      | (Vector Database) |   (User Query / Sentiment)
                        +------------------+      +-------------------+


🌊 2. Data Flow Diagram

The data lifecycle is divided into two distinct phases: Ingestion and Retrieval/Reasoning.

Phase A: Data Ingestion & Indexing

Fetch: Live XML data is scraped from independent public sources (Company Press, Financial News, Market Feeds).

Clean & Chunk: HTML tags are stripped. Text is sliced into 500-character chunks with a 50-character overlap to preserve semantic context across boundaries.

Embed: The BAAI/bge-small-en-v1.5 transformer converts chunks into 384-dimensional dense vectors.

Store: Vectors and metadata (URL, Date) are indexed into ChromaDB.

Phase B: Strategic Intelligence Generation

Query: The CEO Dashboard triggers a specific strategic query (e.g., "Scan Risks").

Vector Search: ChromaDB uses Cosine Similarity to find the Top-K most relevant text chunks.

Sentiment Analysis: FinBERT processes the raw text to map the polarity of the market.

Reasoning: Meta's Llama 3 receives the retrieved context alongside a strict system prompt, synthesizing the data into structured strategic recommendations (Expected Impact, Risk Assessment).

💻 3. Technology Stack

Frontend Dashboard: Streamlit (Python)

Vector Database: ChromaDB

Reasoning LLM: Meta Llama 3 (8B parameters) via Ollama

Embedding Model: BAAI/bge-small-en-v1.5 (HuggingFace)

Sentiment Transformer: ProsusAI/finbert (PyTorch / CUDA)

Data Processing: Pandas, BeautifulSoup, Feedparser

🧠 4. Design Decisions & AI Pipeline Justifications

Why RSS Feeds instead of HTML Web Scraping?

Corporate websites and financial news outlets aggressively block standard HTML scrapers via CAPTCHAs and IP bans. RSS is a structured syndication standard designed for machine-reading, ensuring a robust, enterprise-grade data pipeline that operates reliably in real-time.

Why BAAI/bge-small for Embeddings?

The small variant provides an optimal balance between accuracy and computational efficiency. It achieves 95% of the semantic accuracy of the base model but executes Cosine Similarity searches twice as fast, which is critical for a responsive live dashboard.

Why FinBERT for Sentiment Analysis?

Standard lexical models (like VADER or TextBlob) fail on corporate data. For example, a generic model classifies the phrase "BMW cuts operating costs" as negative due to the word "cuts." FinBERT is fine-tuned specifically on financial data and correctly identifies cost-cutting as a positive strategic action.

Why Llama 3 (8B) and Prompt Engineering?

The project requires the AI to output specific formats (Recommendation, Evidence, Expected Impact). Llama 3 has superior instruction-following capabilities compared to smaller models. By setting the temperature parameter to 0.0, the system is forced into deterministic behavior, strictly relying on the retrieved ChromaDB context to eliminate AI hallucinations.