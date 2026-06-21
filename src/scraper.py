import requests
from bs4 import BeautifulSoup
import json

def scrape_bmw_news():
    articles = []
    
    # We are using aggregated financial and tech news feeds.
    # This guarantees hundreds of results from dozens of independent publishers.
    rss_urls = [
        "https://news.google.com/rss/search?q=BMW+Group+corporate+strategy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=BMW+electric+vehicles+technology&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=BMW+stock+market+finance&hl=en-US&gl=US&ceid=US:en"
    ]
    
    print("Initializing BMW data collection pipeline...")
    
    for url in rss_urls:
        try:
            # We add a User-Agent header to mimic a real browser and prevent being blocked
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'xml')
            
            items = soup.find_all('item')
            print(f"Found {len(items)} articles in feed.")
            
            for item in items:
                # Extracting the actual publisher (e.g., Forbes, MotorTrend) as the independent source
                publisher = item.source.text if item.source else "Financial News"
                
                articles.append({
                    "title": item.title.text if item.title else "No Title",
                    "content": item.description.text if item.description else "No Content",
                    "source": publisher,
                    "date": item.pubDate.text if item.pubDate else "No Date"
                })
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            
    # Task 3 Requirement: Remove duplicates based on the article title
    unique_articles = {article['title']: article for article in articles}.values()
    final_list = list(unique_articles)
            
    # Save the cleaned, deduplicated data
    with open('data/raw_articles.json', 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=4)
    
    return len(final_list)

if __name__ == "__main__":
    count = scrape_bmw_news()
    print(f"\nSuccessfully collected, cleaned, and saved {count} unique BMW articles.")