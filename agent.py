import ollama
import re
from config import LLM_MODEL
from tools import search_strategic_documents, get_recent_news

def query_llm(prompt: str) -> str:
    """Helper function to send prompts to our local Llama 3 model."""
    response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def agent_workflow(strategic_goal: str):
    """
    Executes the strict autonomous workflow required by the examination:
    Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate
    """
    print(f"\n🎯 GOAL RECEIVED: {strategic_goal}")
    
    # ---------------------------------------------------------
    # STEP 1: PLAN
    # ---------------------------------------------------------
    print("\n🧠 STEP 1: PLANNING...")
    plan_prompt = f"""
    You are the AI Strategic Advisor to the CEO of BMW.
    The CEO's goal is: "{strategic_goal}"
    What 3 specific search queries should I use to search our internal intelligence database to gather evidence for this goal?
    Return ONLY the 3 search queries separated by commas, nothing else. No introductory text.
    """
    plan_output = query_llm(plan_prompt)
    
    # Clean up the output to ensure we just have a list of strings
    queries = [q.strip().replace('"', '') for q in plan_output.split(',')]
    print(f"Plan formulated. Target searches: {queries[:3]}")

    # ---------------------------------------------------------
    # STEP 2: RETRIEVE (Tool Usage)
    # ---------------------------------------------------------
    print("\n🛠️ STEP 2: RETRIEVING EVIDENCE...")
    gathered_context = ""
    
    # Tool 1: Get general recent news via SQLite
    gathered_context += "--- RECENT TIMELINE (SQLite) ---\n"
    gathered_context += get_recent_news(limit=3) + "\n\n"
    
    # Tool 2: Semantic search via ChromaDB based on the AI's plan
    gathered_context += "--- SEMANTIC EVIDENCE (ChromaDB) ---\n"
    for query in queries[:3]:
        gathered_context += f"Query Results for '{query}':\n"
        gathered_context += search_strategic_documents(query, n_results=2) + "\n"
        
    print("Evidence successfully retrieved from both databases.")
    # ---------------------------------------------------------
    # NEW: CALCULATE SOURCE DENSITY CONFIDENCE SCORE
    # ---------------------------------------------------------
    # Scan the retrieved text for the word "Source:" and grab the name next to it
    sources_found = re.findall(r"Source:\s*(.*?)\s*\|", gathered_context)
    
    # Convert the list to a "set" to automatically remove any duplicates
    unique_sources = set(sources_found)
    num_sources = len(unique_sources)
    
    # Assign a mathematical confidence score based on corroboration
    if num_sources >= 3:
        confidence_score = 95
    elif num_sources == 2:
        confidence_score = 75
    elif num_sources == 1:
        confidence_score = 50
    else:
        confidence_score = 0
        
    print(f"📊 Source Density Calculated: {num_sources} unique sources found. Confidence = {confidence_score}%")

    # ---------------------------------------------------------
    # STEP 3 & 4: ANALYZE, DECIDE & RECOMMEND
    # ---------------------------------------------------------
    print("\n⚙️ STEP 3 & 4: ANALYZING & GENERATING RECOMMENDATION...")
    recommend_prompt = f"""
    You are advising the CEO of BMW. Based ONLY on the evidence below, formulate a strategic recommendation.
    
    Evidence:
    {gathered_context}
    
    You must format your response exactly like this:
    RECOMMENDATION: [Your specific strategic action]
    EXPECTED IMPACT: [Revenue growth, market differentiation, etc.]
    RISK ASSESSMENT: [Financial, operational, or strategic risks]
    SUPPORTING EVIDENCE: [Cite specific sources from the text]
    """
    draft_recommendation = query_llm(recommend_prompt)

    # ---------------------------------------------------------
    # STEP 5: VALIDATE
    # ---------------------------------------------------------
    print("\n🛡️ STEP 5: VALIDATING RECOMMENDATION...")
    validate_prompt = f"""
    Review this draft recommendation:
    {draft_recommendation}
    
    Does this recommendation logically align with the provided evidence? 
    If yes, output the recommendation as finalized. 
    If it makes assumptions not in the evidence, correct it to be strictly factual.
    Make sure the output is professional and ready for the CEO.
    CRITICAL INSTRUCTION: At the very bottom of your output, you MUST print this exact line:
    CONFIDENCE SCORE: {confidence_score}% (Based on {num_sources} independent sources)
    """
    final_output = query_llm(validate_prompt)
    
    return {
        "context_used": gathered_context,
        "final_recommendation": final_output
    }

if __name__ == "__main__":
    # Test the agent with a real CEO question
    test_goal = "Identify major supply chain risks for our electric vehicle production and propose a mitigation strategy."
    result = agent_workflow(test_goal)
    
    print("\n==================================================")
    print("🏆 FINAL VALIDATED CEO BRIEFING")
    print("==================================================")
    print(result["final_recommendation"])