# BMW AI CEO: Strategic Intelligence Agent

## Project Overview

This project implements a **BMW Strategic Intelligence Agent** for CEO-level decision support.

The system collects live public information about BMW, the automotive market, EV competition, software-defined vehicles, technology trends and competitor activity. It then cleans, enriches, stores, retrieves and analyzes this information to generate evidence-backed strategic recommendations.

The main idea of this project is:

> The LLM is not used as a standalone chatbot. It is part of a controlled strategic intelligence pipeline with live data collection, retrieval, analysis, tool usage and validation.

---

## Core Objective

The goal is to support strategic decision-making for BMW by answering questions such as:

- What are the major risks BMW faces in the EV market?
- What opportunities should BMW focus on?
- How strong is Chinese EV competition?
- What trends are emerging around software-defined vehicles?
- What should BMW do in the next 12 months?

The system follows an agentic workflow:

```text
Goal → Plan → Tool Selection → Retrieve → Analyze → Recommend → Validate
```

---

## Architecture

```text
Live RSS / Public Sources
        ↓
scraper.py
Collects BMW-related live documents
        ↓
intelligence.py
Cleaning + deduplication + enrichment
Sentiment + categories + risks + opportunities + trends
        ↓
database.py
Stores enriched data
        ↓
SQLite metadata database + ChromaDB vector store
        ↓
tools.py
Exposes retrieval and intelligence tools
        ↓
agent.py
Custom Strategic Intelligence Agent
Goal → Plan → Tool Selection → Retrieve → Analyze → Recommend
        ↓
validator.py
Evidence validation + confidence score
        ↓
Main_app.py
Executive Streamlit dashboard
```

---

## Project Structure

```text
bmw_ai_ceo/
├── Main_app.py
├── README.md
├── agent.py
├── config.py
├── database.py
├── intelligence.py
├── prompts.py
├── requirements.txt
├── scraper.py
├── tools.py
├── validator.py
├── .gitignore
└── data/
    ├── bmw_metadata.db
    └── bmw_vector_store/
```

---

## Main Components

### `scraper.py`

Collects live BMW-related documents from public sources such as market news, competitor updates, technology news and automotive industry sources.

It is responsible for:

- Live data collection
- Extracting titles, links, source names and article text
- Sending documents into the processing pipeline
- Supporting reset and re-ingestion

Without this file, the system would become a static chatbot instead of a live intelligence system.

---

### `intelligence.py`

This is the analysis and enrichment layer.

It performs:

- Text cleaning
- Deduplication hash generation
- Category classification
- FinBERT sentiment analysis
- Competitor detection
- Strategic relevance scoring
- Risk signal detection
- Opportunity signal detection
- Trend detection

This file turns raw documents into structured business intelligence.

---

### `database.py`

This file manages the storage layer.

It uses:

- **SQLite** for structured metadata
- **ChromaDB** for vector embeddings and semantic retrieval

SQLite stores information such as title, source, date, category, sentiment and document text.

ChromaDB stores embeddings so the system can perform semantic search.

---

### `tools.py`

This file exposes callable tools to the agent.

Main tools include:

- `semantic_search`
- `recent_news`
- `sentiment_summary`
- `risk_signals`
- `opportunity_signals`
- `trend_signals`
- `competitor_activity`
- `evidence_validation`

These tools allow the agent to retrieve and analyze evidence before generating recommendations.

---

### `agent.py`

This is the main Strategic Intelligence Agent.

The most important function is:

```python
run_agent()
```

It performs:

- Goal understanding
- Planning
- Tool selection
- Evidence retrieval
- Evidence analysis
- Recommendation generation
- Validation request
- Agent Trace generation

This file is the main orchestrator of the system.

---

### `validator.py`

This file validates recommendations before they are shown to the user.

It checks:

- Evidence count
- Source diversity
- Recency
- Category relevance
- Sentiment consistency
- Confidence threshold
- Non-empty recommendation

The confidence score is not the LLM's self-confidence. It is an evidence-quality score.

---

### `Main_app.py`

This is the Streamlit executive dashboard.

Dashboard sections include:

- Company Overview
- Market Intelligence
- Competitor Intelligence
- Opportunity Monitor
- Risk Monitor
- Sentiment Analysis
- Strategic Recommendations
- CEO Briefing
- Agent Trace

---

## LLM Usage

The system uses a local open-source LLM through Ollama.

The model is configured in `config.py`:

```python
LLM_MODEL = "qwen3:8b"
```

or another installed Ollama model such as:

```python
LLM_MODEL = "llama3"
```

The LLM is used for:

- Planning
- Strategic analysis
- Recommendation generation
- CEO briefing writing

The LLM is not responsible for:

- Data collection
- Database storage
- Embedding generation
- FinBERT sentiment analysis
- Rule-based risk detection
- Confidence scoring
- Dashboard rendering

---

## Custom Agent vs LangChain or LangGraph

This project uses a custom Python orchestrator instead of LangChain or LangGraph.

Reason:

> The goal was to make the workflow transparent, lightweight and easy to explain during examination and live coding.

The custom workflow is:

```text
Goal → Plan → Tool Selection → Retrieve → Analyze → Recommend → Validate
```

This is compatible with LangGraph. Each current function could become a LangGraph node:

```text
Planner Node
Tool Selector Node
Retriever Node
Analyzer Node
Recommendation Node
Validator Node
Output Node
```

In a production extension, LangGraph could be used for:

- Conditional branching
- Retry loops
- Human-in-the-loop validation
- State persistence
- Multi-agent workflows

---

## Difference From Basic RAG

A basic RAG system usually follows:

```text
User question → Retrieve documents → LLM answer
```

This project follows:

```text
CEO goal → Plan → Select tools → Retrieve evidence → Analyze risks/opportunities/trends → Generate recommendation → Validate → Dashboard output
```

Therefore, RAG is only one part of the system.

The full system includes:

- Retrieval
- Tool usage
- Business intelligence analysis
- Sentiment monitoring
- Risk detection
- Opportunity detection
- Trend detection
- Evidence validation
- Executive dashboard

---

## Sentiment Analysis

The system uses FinBERT for business sentiment analysis.

Sentiment labels:

- Positive
- Neutral
- Negative

Why FinBERT:

> BMW-related articles are business and market texts. FinBERT is more suitable for financial and corporate language than general sentiment tools.

Most business news articles are factual, so neutral sentiment is expected.

---

## Confidence Score

The confidence score is calculated by `validator.py`.

It is based on evidence quality, not LLM self-confidence.

Factors include:

- Number of evidence items
- Diversity of source types
- Recency of evidence
- Category match with the strategic question
- Sentiment consistency
- Recommendation completeness

High confidence means the recommendation is well supported by retrieved evidence. It does not mean the recommendation is guaranteed to be correct.

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
```

If Ollama is already running, this may show an address already in use message. That is fine.

### 3. Pull the recommended model

```bash
ollama pull qwen3:8b
```

Alternative backup model:

```bash
ollama pull llama3
```

### 4. Configure the model

In `config.py`, set:

```python
LLM_MODEL = "qwen3:8b"
```

or:

```python
LLM_MODEL = "llama3"
```

### 5. Run data ingestion

```bash
python scraper.py --reset
```

Expected result:

```text
BMW Strategic Intelligence Ingestion Result
attempted: 180
inserted: 177
status: OK
```

The exact number may change depending on available live sources.

### 6. Run the dashboard

```bash
streamlit run Main_app.py --server.address 0.0.0.0 --server.port 8501
```

---

## Suggested Demo Question

Use this question in the CEO Briefing tab:

```text
What should BMW do about Chinese EV competition, software-defined vehicles, and EV market pressure in the next 12 months?
```

This question activates:

- Semantic search
- Competitor intelligence
- Risk signals
- Opportunity signals
- Trend signals
- Sentiment summary
- Evidence validation

---

## Agent Trace

Agent Trace shows the high-level workflow followed by the agent:

```text
Goal received
Plan generated
Tools selected
Evidence retrieved
Analysis completed
Recommendation generated
Validation completed
```

Agent Trace is a structured audit log. It is not private chain-of-thought.

---

## Limitations

This is an academic prototype, not a complete enterprise deployment.

Current limitations:

- Uses public web and RSS sources
- Risk and opportunity extraction partly uses rule-based logic
- Source quality depends on available public feeds
- No scheduled background monitoring yet
- No human approval layer yet
- No LangGraph-based retry loop yet

Future improvements:

- Add financial APIs
- Add official BMW investor documents
- Add scheduled daily ingestion
- Add LangGraph workflow with conditional loops
- Add human-in-the-loop validation
- Add more advanced classifiers
- Add stronger source reliability scoring
- Deploy as a full web application

---

## Final Summary

This project demonstrates an evidence-based Strategic Intelligence Agent for BMW.

It combines:

- Live data collection
- Data cleaning and enrichment
- SQLite metadata storage
- ChromaDB semantic retrieval
- FinBERT sentiment analysis
- Risk, opportunity and trend detection
- Custom agent orchestration
- Local LLM generation through Ollama
- Recommendation validation
- Streamlit executive dashboard

The main contribution is:

> The system controls how the answer is created, using evidence, tools, analysis and validation before presenting a CEO-level recommendation.