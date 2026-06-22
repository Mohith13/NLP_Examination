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
        
    # Retrieve top 5 most relevant chunks [cite: 81]
    context_docs, meta = retrieve_context(search_term, n_results=5)
    
    if not context_docs:
        return "No data available in the knowledge base.", []

    # Join the chunks into a single string to feed to the LLM
    context_string = "\n\n".join(context_docs)
    
    # 2. Strict Agentic Prompting [cite: 118-133]
    prompt = f"""
    You are the AI Chief Executive Officer (CEO) Advisor for the BMW Group. 
    Analyze the following recent intelligence reports strictly to identify strategic {query_type}:
    
    {context_string}
    
    Provide your output in a clear, executive-level format. You MUST include:
    1. The primary {query_type} identified.
    2. The supporting evidence (quote or reference the text provided).
    3. The Expected Impact.
    4. Strategic Recommendation (What management should do next).
    
    Do not invent information. Rely ONLY on the provided text. Keep it concise.
    """
    
    # 3. Call your local 4GB-optimized model [cite: 184]
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