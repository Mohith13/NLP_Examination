import ollama
from src.embeddings import retrieve_context

def analyze_strategic_query(query_type):
    print(f"Agent is scanning the database for {query_type}...")
    
    # 1. Semantic Search: Fetch relevant text chunks from your local ChromaDB
    if query_type == "opportunities":
        search_term = "BMW new markets, electric vehicle expansion, solid state batteries, partnerships"
    elif query_type == "risks":
        search_term = "BMW supply chain issues, Chinese EV competition, tariffs, emission regulations"
    elif query_type == "trends":
        search_term = "BMW autonomous driving, AI integration, consumer shift to EV"
    else:
        search_term = "BMW corporate strategy"
        
    # Retrieve top 5 most relevant chunks
    context_docs, meta = retrieve_context(search_term, n_results=5)
    
    if not context_docs:
        return "No data available in the knowledge base.", []

    # Join the chunks into a single string to feed to the LLM
    context_parts = []
    for doc, m in zip(context_docs, meta):
        source_name = m.get('source', 'Unknown Source')
        context_parts.append(f"[Source: {source_name}]\n{doc}")
    context_string = "\n\n".join(context_parts)
    
    # 2. Strict Agentic Prompting tailored to the exact Rubric Requirements
    if query_type == "opportunities":
        format_rules = """For EACH opportunity, you MUST provide:
- **Opportunity Title:** [Name of the opportunity]
- **Impact Level:** [High / Medium / Low]
- **Evidence:** [Quote the fact AND provide the [Source: ...]]
- **Confidence Score:** [0-100%]"""
    
    elif query_type == "risks":
        format_rules = """For EACH risk, you MUST provide:
- **Risk Title:** [Name of the risk]
- **Risk Category:** [Strategic / Operational / Financial / Tech]
- **Severity Level:** [High / Medium / Low]
- **Evidence:** [Quote the fact AND provide the [Source: ...]]
- **Confidence Score:** [0-100%]"""
    
    elif query_type == "recommendations":
        format_rules = """For EACH recommendation, you MUST provide:
- **Recommendation:** [State the strategic action clearly]
- **Priority:** [High / Medium / Low]
- **Supporting Evidence:** [Quote the fact AND provide the [Source: ...]]
- **Expected Impact:** [What this will achieve for the business]
- **Risk Level:** [High / Medium / Low]"""
    
    elif query_type == "ceo_briefing":
        format_rules = """You MUST structure your response into exactly three sections:
1. **What happened?** [Summarize the critical events]
2. **Why does it matter?** [Explain the business impact]
3. **What should management do next?** [Executive action plan]"""
    
    else:
        format_rules = "Provide a clear executive summary with evidence."

    prompt = f"""
You are the AI Chief Executive Officer (CEO) Advisor for the BMW Group. 
Analyze the following recent intelligence reports strictly to identify strategic {query_type}:

{context_string}

Provide your output in a clear, executive-level format.
{format_rules}

Do not invent information. Rely ONLY on the provided text. Keep it concise.
"""
    
    # 3. Call your local 4GB-optimized model
    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'], meta
    except Exception as e:
        return f"Error communicating with local LLM: {e}. Is the Ollama app running?", []

# Quick test execution block
if __name__ == "__main__":
    print("\n--- Testing Agent: Strategic Risks ---")
    insight, sources = analyze_strategic_query("risks")
    print("\n" + insight)
    print("\n--- Sources Cited ---")
    for s in sources:
        print(f"- {s['title']} ({s['source']})")