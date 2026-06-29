# BMW AI CEO: Strategic Intelligence Agent

This project is an AI-powered Strategic Intelligence Agent for **BMW Group**. It collects live public information, stores and indexes evidence, analyzes risks/opportunities/trends/sentiment, and generates CEO-level strategic recommendations with validation.

The system is intentionally designed as more than a simple RAG chatbot. RAG is one tool inside a wider agent workflow:

```text
Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate
```

## Company

- Company: BMW Group
- Industry: Automotive / Electric Mobility / Premium Vehicles
- Competitors monitored: Mercedes-Benz, Volkswagen, Tesla, BYD
- Strategic themes: EV transition, Neue Klasse, battery supply chain, China EV competition, software-defined vehicles, autonomous driving, regulation, premium market, charging infrastructure, profit margins

## Architecture

```text
Live RSS/Public Sources
        ↓
scraper.py
        ↓
Cleaning + deduplication + enrichment
        ↓
SQLite metadata database + ChromaDB vector store
        ↓
intelligence.py generates sentiment, categories, risks, opportunities, trends
        ↓
tools.py exposes retrieval and intelligence tools
        ↓
agent.py orchestrates planning, retrieval, analysis, recommendation, validation
        ↓
Main_app.py executive Streamlit dashboard
```

## Data Flow

```text
RSS source → article URL → article text extraction → clean text
→ content hash duplicate check → sentiment/category/relevance enrichment
→ SQLite structured metadata → ChromaDB semantic index
→ agent tools retrieve evidence → LLM analyzes evidence
→ validator checks recommendation → CEO dashboard displays result
```

## Files

| File | Purpose |
|---|---|
| `scraper.py` | Collects live BMW intelligence from public RSS/news sources |
| `database.py` | Manages SQLite database and ChromaDB vector store |
| `intelligence.py` | Cleans text, classifies category, calculates sentiment, detects risks/opportunities/trends |
| `tools.py` | Provides explicit tools for the agent |
| `agent.py` | Main agent workflow and orchestration |
| `validator.py` | Evidence-based validation and confidence scoring |
| `prompts.py` | Prompt templates for planning, analysis, and CEO briefing |
| `Main_app.py` | Executive Intelligence Dashboard |
| `config.py` | Company, model, source, and path configuration |

## Agent Tools

The agent can use the following tools:

- `semantic_search`
- `recent_news`
- `competitor_activity`
- `risk_signals`
- `opportunity_signals`
- `trend_signals`
- `sentiment_summary`
- `evidence_validation`

This proves that the LLM is not the whole system. The LLM helps with planning and executive language, while tools handle retrieval, monitoring, analysis, and validation.

## Validation Strategy

Each recommendation is validated using:

1. Evidence count
2. Independent source diversity
3. Evidence recency
4. Category/goal match
5. Sentiment consistency

The final output includes:

- Validation status
- Confidence score
- Evidence count
- Source types
- Component scores

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Install and start Ollama if not already running. Check available models:

```bash
ollama list
```

If `qwen3:8b` is not available, change `LLM_MODEL` in `config.py` to a model available on your server.

## Run the Pipeline

Collect live data:

```bash
python scraper.py --reset
```

Run dashboard:

```bash
streamlit run Main_app.py
```

Inside the dashboard:

1. Click **Run Live Data Collection**
2. Click **Generate Risk/Opportunity/Trend Insights**
3. Open **CEO Briefing**
4. Ask: `If you were the CEO of BMW today, what would you do next and why?`
5. Open **Agent Trace** to show the complete agent workflow

## Dashboard Sections

- Company Overview
- Market Intelligence
- Competitor Intelligence
- Opportunity Monitor
- Risk Monitor
- Sentiment Analysis
- Strategic Recommendations
- CEO Briefing
- Agent Trace



## Limitations

- Some publishers may block article extraction, so the system falls back to RSS summaries.
- RSS source availability can change.
- Lightweight keyword-based intelligence rules are used for explainability.
- The LLM depends on local Ollama availability.

## Future Improvements

- Add scheduled collection
- Add PostgreSQL/pgvector
- Add knowledge graph for company/competitor relationships
- Add stronger financial metrics
- Add more official company and regulatory sources
- Add contradiction detection between evidence sources
