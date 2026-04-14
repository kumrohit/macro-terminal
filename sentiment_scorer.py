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

def analyze_sentiment(df, market_context="Market focus is on 'Bad News is Good News'—weak data is bullish as it brings forward Fed rate cuts."):
    """Uses Gemini to score the sentiment of each headline."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please export it first.")
        
    client = genai.Client(api_key=api_key)
    
    # Define the system prompt for financial sentiment analysis
    system_instruction = """
    You are a Lead Equity Quantitative Strategist. Your task is to analyze headlines through the lens of Equity Risk Premiums and Discounted Cash Flow (DCF) models.

    For each headline, determine:
    1. REGIME: [Goldilocks, Reflation, Stagflation, or Recession].
    2. VECTOR: Does this news impact the 'Numerator' (Earnings/Growth) or the 'Denominator' (Interest Rates/Valuation)?
    3. SCORE: A value from -1.0 (Highly Bearish) to 1.0 (Highly Bullish).

    FORMAT:
    Score: [score]
    Rationale: [1 sentence explaining the equity transmission mechanism]
    Regime: [regime]
    Vector: [Numerator/Denominator/Both]
    """
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scoring sentiment with Gemini 2.5 Flash...")
    scores = []
    rationales = []
    regimes = []
    vectors = []
    
    for idx, row in df.iterrows():
        prompt = f"""
        Current Market Context: {market_context}
        Analyze this headline: '{row['title']}' from source: {row['source']}
        """
        
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
            lines = text.split('\n')
            score_line = next((line for line in lines if line.startswith('Score:')), None)
            rationale_line = next((line for line in lines if line.startswith('Rationale:')), None)
            regime_line = next((line for line in lines if line.startswith('Regime:')), None)
            vector_line = next((line for line in lines if line.startswith('Vector:')), None)
            
            if score_line:
                try:
                    score = float(score_line.replace('Score:', '').strip())
                except:
                    score = 0.0
            else:
                score = 0.0
                
            rationale = rationale_line.replace('Rationale:', '').strip() if rationale_line else "No rationale provided"
            regime = regime_line.replace('Regime:', '').strip() if regime_line else "Unknown"
            vector = vector_line.replace('Vector:', '').strip() if vector_line else "Unknown"
            
            scores.append(score)
            rationales.append(rationale)
            regimes.append(regime)
            vectors.append(vector)
            
            print(f"[{row['source']}] {row['title'][:50]}... -> Score: {score}, Regime: {regime}, Vector: {vector}")
            time.sleep(1) # Small delay to avoid aggressive rate limits
            
        except Exception as e:
            print(f"Error analyzing headline '{row['title'][:30]}': {e}")
            scores.append(0.0)
            rationales.append("Error analyzing sentiment.")
            regimes.append("Unknown")
            vectors.append("Unknown")
            
    df['sentiment_score'] = scores
    df['rationale'] = rationales
    df['regime'] = regimes
    df['vector'] = vectors
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
        
        # Determine dominant regime
        dominant_regime = scored_df['regime'].mode()[0] if not scored_df['regime'].empty else "Unknown"
        dominant_vector = scored_df['vector'].mode()[0] if not scored_df['vector'].empty else "Unknown"
        
        print("\n" + "="*50)
        print(f"📈 AGGREGATE MARKET SENTIMENT (AI): {avg_sentiment_ai:.2f}")
        print(f"📈 AGGREGATE MARKET SENTIMENT (VADER): {avg_sentiment_vader:.2f}")
        print(f"🏛️ DOMINANT MACRO REGIME: {dominant_regime}")
        print(f"🔄 PRIMARY VECTOR: {dominant_vector}")
        
        # Combine or compare sentiments (Using AI for the final verdict as before, or average them)
        avg_combined = (avg_sentiment_ai + avg_sentiment_vader) / 2
        print(f"📈 COMBINED AVERAGE SENTIMENT: {avg_combined:.2f}")
        
        if avg_combined > 0.15:
            print("Verdict: Bullish 🟢")
        elif avg_combined < -0.2:
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
