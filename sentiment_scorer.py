import os
import feedparser
import requests
import pandas as pd
from google import genai
from google.genai import types
import time
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- CONFIGURATION ---
RSS_FEEDS = {
    "Investing.com (Macro)": "https://www.investing.com/rss/news_286.rss",
    "CNBC (World)": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
    "Yahoo Finance": "https://finance.yahoo.com/news/rss"
}

# --- FUNCTIONS ---
def fetch_top_news(limit_per_feed=5):
    """Fetches the latest headlines from our Macro Terminal feeds."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching live news feeds...")
    articles = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    for source, url in RSS_FEEDS.items():
        try:
            response = requests.get(url, headers=headers, timeout=10)
            feed = feedparser.parse(response.content)
            for entry in feed.entries[:limit_per_feed]:
                articles.append({
                    "title": entry.title,
                    "source": source,
                    "published": entry.get('published', 'Just now')
                })
        except Exception as e:
            print(f"Error loading {source}: {e}")
    return pd.DataFrame(articles)

def analyze_sentiment(df):
    """Uses Gemini to score the sentiment of each headline."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please export it first.")
        
    client = genai.Client(api_key=api_key)
    
    # Define the system prompt for financial sentiment analysis
    system_instruction = """
    You are an expert financial quantitative analyst. Your task is to analyze news headlines and determine their immediate sentiment impact on broad global equity markets.
    
    Respond STRICTLY in the following format:
    Score: [A number between -1.0 (highly bearish) to 1.0 (highly bullish), 0.0 is neutral]
    Rationale: [One short sentence explaining why]
    """
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scoring sentiment with Gemini 2.5 Flash...")
    scores = []
    rationales = []
    
    for idx, row in df.iterrows():
        prompt = f"Analyze this headline: '{row['title']}' from source: {row['source']}"
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1, # Keep it deterministic
                )
            )
            
            # Parse the response
            text = response.text.strip()
            score_line = [line for line in text.split('\n') if line.startswith('Score:')][0]
            rationale_line = [line for line in text.split('\n') if line.startswith('Rationale:')][0]
            
            score = float(score_line.replace('Score:', '').strip())
            rationale = rationale_line.replace('Rationale:', '').strip()
            
            scores.append(score)
            rationales.append(rationale)
            
            print(f"[{row['source']}] {row['title'][:50]}... -> Score: {score}")
            time.sleep(1) # Small delay to avoid aggressive rate limits
            
        except Exception as e:
            print(f"Error analyzing headline '{row['title'][:30]}': {e}")
            scores.append(0.0)
            rationales.append("Error analyzing sentiment.")
            
    df['sentiment_score'] = scores
    df['rationale'] = rationales
    return df

def add_vader_sentiment(df):
    """Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to score sentiment."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scoring sentiment with VADER (Lexicon-based)...")
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = []
    
    for idx, row in df.iterrows():
        # VADER returns a dictionary with pos, neg, neu, and compound. We use compound (-1 to 1).
        score = analyzer.polarity_scores(row['title'])['compound']
        vader_scores.append(score)
        
    df['vader_score'] = vader_scores
    return df

if __name__ == "__main__":
    # 1. Fetch News
    news_df = fetch_top_news(limit_per_feed=5) # Top 15 articles total
    
    if news_df.empty:
        print("No news fetched. Exiting.")
        exit()
        
    # 2. Score Sentiment
    try:
        # Add Lexicon-based sentiment first (VADER)
        scored_df = add_vader_sentiment(news_df)
        
        # Add AI-based sentiment
        scored_df = analyze_sentiment(scored_df)
        
        # 3. Calculate Aggregate Market Sentiment
        avg_sentiment_ai = scored_df['sentiment_score'].mean()
        avg_sentiment_vader = scored_df['vader_score'].mean()
        
        print("\n" + "="*50)
        print(f"📈 AGGREGATE MARKET SENTIMENT (AI): {avg_sentiment_ai:.2f}")
        print(f"📈 AGGREGATE MARKET SENTIMENT (VADER): {avg_sentiment_vader:.2f}")
        
        # Combine or compare sentiments (Using AI for the final verdict as before, or average them)
        avg_combined = (avg_sentiment_ai + avg_sentiment_vader) / 2
        print(f"📈 COMBINED AVERAGE SENTIMENT: {avg_combined:.2f}")
        
        if avg_combined > 0.15:
            print("Verdict: Bullish 🟢")
        elif avg_sentiment < -0.2:
            print("Verdict: Bearish 🔴")
        else:
            print("Verdict: Neutral ⚪")
        print("="*50 + "\n")
        
        # Save to CSV for later backtesting
        output_file = f"market_sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        scored_df.to_csv(output_file, index=False)
        print(f"Full analysis saved to {output_file}")
        
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("Run this command in your terminal before running the script:")
        print("export GEMINI_API_KEY='your_api_key_here'")
