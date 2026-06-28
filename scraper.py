import urllib.parse
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
import uuid

# Import our custom database functions
from database import init_databases, store_document

def clean_html_text(raw_html):
    """Cleans up raw HTML string into plain, readable text for the LLM."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ").strip()

def fetch_rss_feed(url, source_name):
    """Fetches articles from a given RSS feed URL and returns structured data."""
    print(f"📡 Fetching live data from {source_name}...")
    # We use a User-Agent header so websites don't block us thinking we are a malicious bot
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"⚠ Failed to pull from {source_name}. Status: {response.status_code}")
            return []
            
        # Parse the XML content
        root = ET.fromstring(response.content)
        articles = []
        
        # Loop through every <item> tag in the RSS feed
        for item in root.findall(".//item"):
            title = item.find("title").text if item.find("title") is not None else "No Title"
            link = item.find("link").text if item.find("link") is not None else ""
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            
            # Extract content text and clean it
            description_tag = item.find("description")
            description = description_tag.text if description_tag is not None else ""
            clean_text = clean_html_text(description)
            
            # If the description is too short, we fall back to the title
            if not clean_text or len(clean_text) < 10:
                clean_text = f"Market briefing regarding: {title}"

            articles.append({
                "title": title,
                "link": link,
                "publish_date": pub_date,
                "raw_text": clean_text,
                "source": source_name
            })
        return articles
    except Exception as e:
        print(f"❌ Error parsing {source_name}: {e}")
        return []

def run_pipeline():
    """Main execution loop to harvest data and populate our hybrid database."""
    print("🔄 Initializing Strategic Intelligence Pipeline...")
    
    # 1. Ensure databases exist before we try to save to them
    init_databases()
    
    # 2. Define 3 independent public data streams for BMW
    # Using urllib.parse.quote ensures complex search terms don't break the URL
    feeds = {
        "Yahoo Finance (BMW)": "https://finance.yahoo.com/rss/headline?s=BMW.DE",
        "Google News (BMW Group)": f"https://news.google.com/rss/search?q={urllib.parse.quote('BMW Group corporate')}&hl=en-US&gl=US&ceid=US:en",
        "EV Market Trends": f"https://news.google.com/rss/search?q={urllib.parse.quote('automotive electric vehicle industry trends')}&hl=en-US&gl=US&ceid=US:en"
    }
    
    total_processed = 0
    
    # 3. Loop through each source, download, and save
    for source_name, url in feeds.items():
        extracted_docs = fetch_rss_feed(url, source_name)
        print(f"📥 Found {len(extracted_docs)} documents from {source_name}")
        
        for doc in extracted_docs:
            # Generate a unique hash identifier for every document
            unique_id = f"bmw_doc_{uuid.uuid4().hex[:12]}"
            
            # Send immediately to our dual-database setup
            store_document(
                doc_id=unique_id,
                title=doc["title"],
                source=doc["source"],
                url=doc["link"],
                publish_date=doc["publish_date"],
                raw_text=doc["raw_text"]
            )
            total_processed += 1
            
    print(f"\n✨ Ingestion complete. Total entries indexed: {total_processed}")
    if total_processed < 100:
        print("💡 Note: If you have less than 100 documents, run the script again tomorrow, or add more RSS feeds to hit the minimum requirement.")

if __name__ == "__main__":
    run_pipeline()