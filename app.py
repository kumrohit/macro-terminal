import streamlit as st
import feedparser
import pandas as pd
import requests
import datetime
import time
import os
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

import plotly.express as px
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Macro Terminal", layout="wide", initial_sidebar_state="collapsed")

INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Nifty 50": "^NSEI",
    "Nikkei 225": "^N225",
    "10Y Treasury": "^TNX"
}

VOLATILITY_INDICES = {
    "CBOE VIX (US)": "^VIX",
    "Nasdaq VIX": "^VXN",
    "India VIX": "^INDIAVIX"
}

# Inject custom CSS for the Bloomberg Terminal aesthetic
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    /* Custom Card Class */
    .macro-card {
        background-color: #1c2128;
        border: 1px solid #30363D;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    h1, h2, h3 {
        color: #ffffff;
        border-bottom: 1px solid #333333;
    }
    .news-title {
        color: #00ff00;
        font-weight: bold;
        font-size: 1.1em;
    }
    .news-source {
        color: #888888;
        font-size: 0.8em;
    }
    a {
        color: #00ffff !important;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA SOURCES ---
RSS_FEEDS = {
    "Investing.com (Macro)": "https://www.investing.com/rss/news_286.rss",
    "CNBC (World)": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
    "Yahoo Finance": "https://finance.yahoo.com/news/rss"
}

@st.cache_data(ttl=300) # Cache for 5 mins to avoid spamming feeds
def fetch_news():
    articles = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    for source, url in RSS_FEEDS.items():
        try:
            # Fetch using requests to bypass basic anti-bot protections
            response = requests.get(url, headers=headers, timeout=10)
            feed = feedparser.parse(response.content)
            for entry in feed.entries[:10]:  # Top 10 from each
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get('published', 'Just now'),
                    "source": source
                })
        except Exception as e:
            st.sidebar.error(f"Error loading {source}: {e}")
    return pd.DataFrame(articles)

import xml.etree.ElementTree as ET

@st.cache_data(ttl=3600)
def fetch_calendar():
    # ForexFactory daily XML feed for economic events
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check if we got rate limited (returns HTML instead of XML)
        if b"Rate Limited" in response.content or b"<!DOCTYPE html>" in response.content:
            st.sidebar.warning("ForexFactory is temporarily rate-limiting our calendar requests. It will automatically recover in a few minutes.")
            return pd.DataFrame({"Info": ["Calendar rate-limited. Please wait 5 mins."]})

        # Parse XML manually to avoid pandas lxml dependency issues
        root = ET.fromstring(response.content)
        events = []
        for event in root.findall('event'):
            impact = event.find('impact').text if event.find('impact') is not None else ''
            if impact == 'High':
                events.append({
                    'Date': event.find('date').text if event.find('date') is not None else '',
                    'Time': event.find('time').text if event.find('time') is not None else '',
                    'Curr': event.find('currency').text if event.find('currency') is not None else '',
                    'Event': event.find('title').text if event.find('title') is not None else '',
                    'Actual': event.find('actual').text if event.find('actual') is not None else '',
                    'Forecast': event.find('forecast').text if event.find('forecast') is not None else '',
                    'Previous': event.find('previous').text if event.find('previous') is not None else ''
                })
                
        if not events:
            return pd.DataFrame({"Info": ["No high-impact events found this week."]})
            
        return pd.DataFrame(events)
        
    except Exception as e:
        st.sidebar.error(f"Error loading calendar: {e}")
        return pd.DataFrame({"Error": ["Could not load calendar data."]})

@st.cache_data(ttl=60)
def fetch_ticker_data(ticker_dict):
    data = {}
    for name, ticker in ticker_dict.items():
        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period="2d")
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[0]
                current = hist['Close'].iloc[-1]
                change = current - prev_close
                pct_change = (change / prev_close) * 100
                data[name] = {"price": current, "change": change, "pct_change": pct_change}
            elif len(hist) == 1:
                current = hist['Close'].iloc[0]
                data[name] = {"price": current, "change": 0.0, "pct_change": 0.0}
        except Exception as e:
            pass
    return data

@st.cache_data(ttl=3600)
def fetch_5y_chart_data(ticker):
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period="5y")
        return hist['Close']
    except Exception as e:
        return pd.Series()

def fetch_vader_sentiment(news_df):
    """Calculates aggregate market sentiment using VADER lexicon."""
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for title in news_df['title'].head(15).tolist():
        score = analyzer.polarity_scores(title)['compound']
        scores.append(score)
    
    if scores:
        avg_score = sum(scores) / len(scores)
        return round(avg_score, 2)
    return 0.0

def fetch_vader_ai_commentary(news_df, vader_score, api_key):
    """Uses AI to explain the VADER lexicon score based on the headlines."""
    if not HAS_GENAI or not api_key:
        return None
    client = genai.Client(api_key=api_key)
    
    system_instruction = """
    You are a financial quantitative analyst. You have run a VADER lexicon sentiment analysis on recent news headlines, and the resulting aggregate score is provided.
    Your task is to write a single, concise paragraph (2-3 sentences) explaining *why* the VADER score is at that level based on the specific words and themes in the headlines.
    Do not perform a new sentiment analysis yourself; instead, explain what strong positive or negative words VADER (a lexicon-based tool) is likely reacting to in these headlines to arrive at the given score.
    """
    headlines = news_df['title'].head(10).tolist()
    prompt = f"VADER Aggregate Score: {vader_score}\nHeadlines:\n" + "\n".join([f"- {h}" for h in headlines])
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )
        return response.text.strip()
    except Exception as e:
        return None

def get_sector_scores(regime):
    """Returns expected sector performance scores based on macro regime."""
    base_scores = {
        "Technology": 0,
        "Consumer Discretionary": 0,
        "Energy": 0,
        "Materials": 0,
        "Industrials": 0,
        "Utilities": 0,
        "Healthcare": 0,
        "Consumer Staples": 0,
        "Financials": 0,
        "Real Estate": 0
    }
    
    regime = regime.upper()
    if regime == "GOLDILOCKS":
        # High beta sectors perform best
        scores = {
            "Technology": 0.8,
            "Consumer Discretionary": 0.7,
            "Financials": 0.6,
            "Real Estate": 0.5,
            "Industrials": 0.4,
            "Materials": 0.3,
            "Energy": 0.2,
            "Healthcare": 0.1,
            "Consumer Staples": 0.0,
            "Utilities": -0.1
        }
    elif regime == "REFLATION":
        # Cyclical sectors benefit from rising growth/inflation
        scores = {
            "Energy": 0.8,
            "Materials": 0.7,
            "Industrials": 0.6,
            "Financials": 0.5,
            "Consumer Discretionary": 0.4,
            "Technology": 0.3,
            "Real Estate": 0.2,
            "Healthcare": 0.0,
            "Consumer Staples": -0.1,
            "Utilities": -0.2
        }
    elif regime == "STAGFLATION":
        # All sectors suffer, but some less
        scores = {
            "Healthcare": 0.2,
            "Consumer Staples": 0.1,
            "Utilities": 0.0,
            "Technology": -0.1,
            "Consumer Discretionary": -0.2,
            "Financials": -0.3,
            "Industrials": -0.4,
            "Materials": -0.5,
            "Energy": -0.6,
            "Real Estate": -0.7
        }
    elif regime == "RECESSION":
        # Defensive sectors outperform
        scores = {
            "Utilities": 0.6,
            "Healthcare": 0.5,
            "Consumer Staples": 0.4,
            "Real Estate": 0.2,
            "Technology": 0.1,
            "Financials": 0.0,
            "Consumer Discretionary": -0.1,
            "Industrials": -0.2,
            "Materials": -0.3,
            "Energy": -0.4
        }
    else:
        # Neutral/default
        scores = base_scores
    
    return pd.Series(scores)

def get_sector_status(regime):
    """Returns sector rotation recommendations based on macro regime."""
    regimes = {
        "GOLDILOCKS": {"Bullish": ["Technology", "Consumer Discretionary"], "Bearish": ["Utilities", "Consumer Staples"]},
        "REFLATION": {"Bullish": ["Energy", "Materials", "Financials"], "Bearish": ["Technology", "Utilities"]},
        "STAGFLATION": {"Bullish": ["Utilities", "Healthcare", "Consumer Staples"], "Bearish": ["Technology", "Consumer Discretionary"]},
        "RECESSION": {"Bullish": ["Utilities", "Healthcare", "Consumer Staples"], "Bearish": ["Energy", "Materials", "Financials"]}
    }
    return regimes.get(regime.upper(), {"Bullish": [], "Bearish": []})

def fetch_sentiment_score(news_df, api_key, market_context="Equities are currently in a 'Denominator-driven' phase. High inflation news is bearish; weak growth news might be bullish if it suggests rate cuts."):
    if not HAS_GENAI:
        return {"error": "google-genai not installed"}
        
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}
        
    client = genai.Client(api_key=api_key)
    
    system_instruction = """
    You are a Lead Equity Quantitative Strategist. Your task is to analyze a batch of news headlines through the lens of Equity Risk Premiums and Discounted Cash Flow (DCF) models.

    Determine the dominant macro regime from these headlines:

    REGIMES:
    * Goldilocks: Growth stable/rising, Inflation falling (Highest Equity returns).
    * Reflation: Growth rising, Inflation rising (Bullish for Value/Cyclicals).
    * Stagflation: Growth falling, Inflation rising (Highly Bearish).
    * Recession: Growth falling, Inflation falling (Bearish, but potential for rate cuts).

    Also determine the primary VECTOR: Does the news primarily impact the 'Numerator' (Earnings/Growth) or the 'Denominator' (Interest Rates/Valuation)?

    Respond with:
    Regime: [dominant regime]
    Score: [aggregate score -1.0 to 1.0]
    Rationale: [1 sentence explaining the equity transmission mechanism]
    Commentary: [brief 2-3 sentence summary]
    Vector: [primary vector: Numerator/Denominator/Both]
    """
    
    # Just take the top 10 headlines to avoid massive prompts
    headlines = news_df['title'].head(10).tolist()
    prompt = f"Current Market Context: {market_context}\nAnalyze the aggregate sentiment of these current headlines:\n" + "\n".join([f"- {h}" for h in headlines])
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            )
        )
        text = response.text.strip()
        print("RAW GEMINI RESPONSE:")
        print(text)
        print("-" * 40)
        
        # New parsing logic for the multi-line response
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        data = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                # Clean markdown formatting from values
                clean_val = val.strip().replace('*', '').replace('+', '').strip()
                data[key.strip().lower()] = clean_val
        
        return data
    except Exception as e:
        return {"error": str(e)}
        
    client = genai.Client(api_key=api_key)
    
    system_instruction = """
    You are a Lead Equity Quantitative Strategist. Your task is to analyze a batch of news headlines through the lens of Equity Risk Premiums and Discounted Cash Flow (DCF) models.

    Determine the dominant macro regime from these headlines:

    REGIMES:
    * Goldilocks: Growth stable/rising, Inflation falling (Highest Equity returns).
    * Reflation: Growth rising, Inflation rising (Bullish for Value/Cyclicals).
    * Stagflation: Growth falling, Inflation rising (Highly Bearish).
    * Recession: Growth falling, Inflation falling (Bearish, but potential for rate cuts).

    Also determine the primary VECTOR: Does the news primarily impact the 'Numerator' (Earnings/Growth) or the 'Denominator' (Interest Rates/Valuation)?

    Respond with:
    Regime: [dominant regime]
    Score: [aggregate score -1.0 to 1.0]
    Rationale: [1 sentence explaining the equity transmission mechanism]
    Commentary: [brief 2-3 sentence summary]
    Vector: [primary vector: Numerator/Denominator/Both]
    """
    
    # Just take the top 10 headlines to avoid massive prompts
    headlines = news_df['title'].head(10).tolist()
    prompt = f"Current Market Context: {market_context}\nAnalyze the aggregate sentiment of these current headlines:\n" + "\n".join([f"- {h}" for h in headlines])
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            )
        )
        text = response.text.strip()
        print("RAW GEMINI RESPONSE:")
        print(text)
        print("-" * 40)
        
        score = 0.0
        rationale = "No rationale provided"
        commentary = "No commentary provided"
        regime = "Unknown"
        vector = "Unknown"
        
        # New parsing logic for the multi-line response
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        data = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip().lower()] = val.strip()
        
        return data
    except Exception as e:
        return {"error": str(e)}

# --- DASHBOARD UI ---
st.title("🌐 GLOBAL MACRO TERMINAL")

# Market Status Indicator
current_time = datetime.datetime.now()
is_market_open = (current_time.weekday() < 5 and 9 <= current_time.hour < 16)  # Simplified US market hours
market_status = "🟢 NYSE Open" if is_market_open else "🔴 NYSE Closed"
st.markdown(f"<div style='text-align: right; font-size: 0.8em; color: #888;'>{market_status}</div>", unsafe_allow_html=True)

api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not api_key and "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]

# Quick Links Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Quick Links")

if st.sidebar.button("📊 Run Sentiment Analysis"):
    st.sidebar.info("Running sentiment analysis...")
    # This would trigger the sentiment_scorer.py logic

if st.sidebar.button("💾 Export Sentiment CSV"):
    if 'news_df' in locals() and not news_df.empty:
        # Create comprehensive export data
        export_df = news_df.copy()
        export_df['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if api_key:
            res = fetch_sentiment_score(news_df, api_key)
            if "regime" in res:
                score_str = res.get('score', '0.0').replace('*', '').replace('+', '').strip()
                export_df['ai_regime'] = res.get('regime', 'Unknown').replace('*', '').strip()
                export_df['ai_sentiment_score'] = float(score_str) if score_str else 0.0
                export_df['ai_vector'] = res.get('vector', 'Unknown').replace('*', '').strip()
                export_df['ai_rationale'] = res.get('rationale', '')
        
        # Add VADER scores
        vader_analyzer = SentimentIntensityAnalyzer()
        export_df['vader_compound'] = export_df['title'].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])
        
        csv_data = export_df.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"macro_sentiment_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.error("No data available for export")

if st.sidebar.button("📈 Backtest Sentiment vs S&P"):
    st.sidebar.markdown("### Backtesting Results")
    
    # Backtest using recent S&P data
    try:
        sp500 = yf.Ticker("^GSPC")
        sp_hist = sp500.history(period="3mo")  # Last 3 months
        
        if not sp_hist.empty and len(sp_hist) > 10:
            # Calculate daily returns
            sp_returns = sp_hist['Close'].pct_change().dropna()
            
            # Get current sentiment score
            current_sentiment = 0.0
            if api_key and 'news_df' in locals():
                res = fetch_sentiment_score(news_df, api_key)
                score_str = res.get('score', '0.0').replace('*', '').replace('+', '').strip()
                current_sentiment = float(score_str)
            
            # Simulate sentiment-based strategy
            # Assume sentiment predicts next day's return
            predicted_returns = sp_returns.shift(-1).dropna()
            sentiment_signals = np.full(len(predicted_returns), current_sentiment)
            
            # Calculate correlation
            if len(sentiment_signals) == len(predicted_returns):
                correlation = np.corrcoef(sentiment_signals, predicted_returns.values)[0, 1]
                st.sidebar.metric("Sentiment vs S&P Correlation", f"{correlation:.3f}")
            
            # Strategy performance metrics
            avg_daily_return = sp_returns.mean()
            volatility = sp_returns.std()
            
            st.sidebar.metric("S&P Avg Daily Return", f"{avg_daily_return:.2%}")
            st.sidebar.metric("S&P Volatility", f"{volatility:.2%}")
            
            # Hypothetical strategy: Go long when sentiment > 0.1, short when < -0.1
            strategy_returns = sp_returns.copy()
            if current_sentiment > 0.1:
                # Go long - keep positive returns
                pass  # strategy_returns already equals sp_returns
            elif current_sentiment < -0.1:
                # Go short - flip the sign
                strategy_returns *= -1
            else:
                # Neutral - set returns to 0
                strategy_returns = pd.Series(0, index=sp_returns.index)
            
            if len(strategy_returns) > 0:
                strategy_avg_return = strategy_returns.mean()
                st.sidebar.metric("Strategy Avg Daily Return", f"{strategy_avg_return:.2%}")
                
                # Sharpe ratio approximation
                if volatility > 0:
                    sharpe = strategy_avg_return / volatility
                    st.sidebar.metric("Strategy Sharpe Ratio", f"{sharpe:.2f}")
        else:
            st.sidebar.error("Insufficient S&P data for backtesting")
    except Exception as e:
        st.sidebar.error(f"Backtest error: {str(e)}")

# Yield Watch Sparkline
st.sidebar.markdown("---")
st.sidebar.subheader("📈 10Y Yield Watch")
try:
    tnx = yf.Ticker("^TNX")
    yield_hist = tnx.history(period="1d", interval="1h")
    if not yield_hist.empty:
        st.sidebar.line_chart(yield_hist['Close'])
    else:
        st.sidebar.write("Yield data unavailable")
except:
    st.sidebar.write("Yield data unavailable")

st.text(f"LIVE FEED | LAST REFRESH: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

main_tab1, main_tab2, main_tab3 = st.tabs(["🌍 Markets & Sentiment", "📰 Macro News & Calendar", "🏢 Company Financials"])

with main_tab1:
    # --- GLOBAL MARKETS TICKER ---
    st.markdown("### 📊 Global Equities")
    market_data = fetch_ticker_data(INDICES)
    if market_data:
        cols = st.columns(len(market_data))
        for col, (name, stats) in zip(cols, market_data.items()):
            col.metric(
                label=name, 
                value=f"{stats['price']:,.2f}", 
                delta=f"{stats['change']:+.2f} ({stats['pct_change']:+.2f}%)"
            )

    st.markdown("### ⚡ Global Volatility (VIX)")
    vix_data = fetch_ticker_data(VOLATILITY_INDICES)
    if vix_data:
        vix_cols = st.columns(len(vix_data))
        for col, (name, stats) in zip(vix_cols, vix_data.items()):
            col.metric(
                label=name, 
                value=f"{stats['price']:,.2f}", 
                delta=f"{stats['change']:+.2f} ({stats['pct_change']:+.2f}%)",
                delta_color="inverse"
            )

    with st.expander("📈 View 5Y Historical Charts"):
        # Combine both dictionaries for the chart selector
        all_indices = {**INDICES, **VOLATILITY_INDICES}
        selected_index = st.selectbox("Select Index / VIX", list(all_indices.keys()), label_visibility="collapsed")
        chart_data = fetch_5y_chart_data(all_indices[selected_index])
        if not chart_data.empty:
            st.line_chart(chart_data)
        else:
            st.write("Chart data unavailable.")

    st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)
    news_df = fetch_news()

    # --- AI & VADER SENTIMENT BANNER ---
    if not news_df.empty:
        vader_score = fetch_vader_sentiment(news_df)
        vader_color = "#00ff00" if vader_score > 0.1 else "#ff0000" if vader_score < -0.1 else "#ffffff"
        vader_verdict = "BULLISH 🟢" if vader_score > 0.1 else "BEARISH 🔴" if vader_score < -0.1 else "NEUTRAL ⚪"

        vader_commentary_ai = None
        if api_key:
            vader_commentary_ai = fetch_vader_ai_commentary(news_df, vader_score, api_key)

        if vader_commentary_ai:
            vader_commentary = vader_commentary_ai
        else:
            if vader_score > 0.5:
                vader_commentary = "Lexicon analysis indicates strong positive conviction in the headlines, suggesting a highly bullish macro environment."
            elif vader_score > 0.1:
                vader_commentary = "Lexicon analysis shows mild positive sentiment across recent news."
            elif vader_score < -0.5:
                vader_commentary = "Lexicon analysis indicates strong negative language in the headlines, pointing to significant macro fear or bearishness."
            elif vader_score < -0.1:
                vader_commentary = "Lexicon analysis shows mild negative sentiment across recent news."
            else:
                vader_commentary = "Lexicon analysis detects balanced or neutral language in current headlines, lacking strong directional bias."

        # Always show VADER banner
        st.markdown(f"""
        <div style="border: 1px solid #333; padding: 15px; margin-bottom: 20px; background-color: #111;">
            <h3 style="margin-top: 0; color: {vader_color};">📚 VADER SENTIMENT: {vader_score} ({vader_verdict})</h3>
            <p style="color: #ccc; margin-bottom: 0;"><i>{vader_commentary}</i></p>
        </div>
        """, unsafe_allow_html=True)

        if api_key:
            res = fetch_sentiment_score(news_df, api_key)
            if "score" in res:
                # Clean markdown formatting from AI response
                score_str = res['score'].replace('*', '').replace('+', '').strip()
                score = float(score_str)
                regime = res.get('regime', 'N/A').upper().replace('*', '').strip()
                vector = res.get('vector', 'N/A').replace('*', '').strip()
                
                # Custom CSS for the Regime Badge
                regime_colors = {
                    "GOLDILOCKS": "#00ff00", # Bright Green
                    "REFLATION": "#00ffff",  # Cyan
                    "STAGFLATION": "#ff0000",# Red
                    "RECESSION": "#ff9900"   # Orange
                }
                badge_color = regime_colors.get(regime, "#ffffff")

                # Create a 2-column layout for the top "Pulse" section
                col_regime, col_sectors = st.columns([3, 2])

                with col_regime:
                    st.markdown(f"""
                    <div class="macro-card">
                        <h4 style="color: #8B949E; margin:0;">CURRENT REGIME</h4>
                        <h1 style="color: {badge_color}; margin:0;">{regime}</h1>
                        <p style="color: #7D8590;">Transmission Vector: <b>{vector}</b></p>
                        <p style="font-size: 0.9em;">{res.get('rationale', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_sectors:
                    sector_info = get_sector_status(regime)
                    st.markdown('<div class="macro-card">', unsafe_allow_html=True)
                    st.write("🎯 **Strategic Rotations**")
                    
                    for sector in sector_info["Bullish"]:
                        st.success(f"Overweight: {sector}")
                    for sector in sector_info["Bearish"]:
                        st.error(f"Underweight: {sector}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Sector Sentiment Heatmap
                st.subheader("📊 Sector Performance Expectations")
                sector_scores = get_sector_scores(regime)
                # Create horizontal bar chart
                fig = px.bar(sector_scores.sort_values(), orientation='h', 
                           title=f"Sector Performance in {regime} Regime",
                           labels={'value': 'Expected Performance', 'index': 'Sector'},
                           color=sector_scores.sort_values(),
                           color_continuous_scale=['red', 'yellow', 'green'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig)
            else:
                st.warning(f"AI Sentiment Unavailable: {res.get('error', 'Unknown error')}")
        else:
            st.info("💡 Enter your Gemini API Key in the sidebar to enable AI Sentiment Analysis!")

with main_tab2:
    if 'news_df' not in locals():
        news_df = fetch_news()
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📰 Breaking Macro News")
        
        if not news_df.empty:
            for _, row in news_df.iterrows():
                st.markdown(f"<div class='news-title'><a href='{row['link']}' target='_blank'>{row['title']}</a></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='news-source'>[{row['source']}] - {row['published']}</div><br>", unsafe_allow_html=True)

    with col2:
        st.subheader("📅 High-Impact Calendar")
        cal_df = fetch_calendar()
        if not cal_df.empty:
            st.dataframe(cal_df, hide_index=True, use_container_width=True)
        else:
            st.write("No high impact events found.")

with main_tab3:
    st.subheader("🏢 Company Financials Terminal")
    company_search = st.text_input("🔍 Enter Company Name or Ticker (e.g., Apple, Reliance, TSLA):")

    def get_ticker_from_search(query):
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            res = requests.get(url, headers=headers, timeout=5)
            quotes = res.json().get('quotes', [])
            if quotes:
                return quotes[0]['symbol']
        except Exception:
            pass
        return query.upper()

    if company_search:
        resolved_ticker = get_ticker_from_search(company_search)
        st.info(f"Searching data for ticker: **{resolved_ticker}**")
        try:
            tkr = yf.Ticker(resolved_ticker)
            info = tkr.info
            if info and 'symbol' in info:
                st.markdown(f"### {info.get('shortName', company_search.upper())} ({info.get('symbol', company_search.upper())})")
                
                # Key Metrics
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                
                price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                col_f1.metric("Current Price", f"${price}" if price != 'N/A' else 'N/A')
                
                mcap = info.get('marketCap', 0)
                mcap_str = f"${mcap:,.0f}" if mcap else 'N/A'
                col_f2.metric("Market Cap", mcap_str)
                
                pe = info.get('trailingPE', 'N/A')
                col_f3.metric("P/E Ratio", f"{pe:.2f}" if isinstance(pe, (int, float)) else pe)
                
                high_52 = info.get('fiftyTwoWeekHigh', 'N/A')
                col_f4.metric("52W High", f"${high_52}" if high_52 != 'N/A' else 'N/A')
                
                # Financial Data Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Company Profile"])
                
                with tab1:
                    fin = tkr.financials
                    if not fin.empty:
                        # Convert column headers to strings for better Streamlit rendering
                        fin.columns = [str(c).split(' ')[0] for c in fin.columns]
                        st.dataframe(fin, use_container_width=True)
                    else:
                        st.write("No Income Statement data available.")
                
                with tab2:
                    bs = tkr.balance_sheet
                    if not bs.empty:
                        bs.columns = [str(c).split(' ')[0] for c in bs.columns]
                        st.dataframe(bs, use_container_width=True)
                    else:
                        st.write("No Balance Sheet data available.")
                        
                with tab3:
                    cf = tkr.cashflow
                    if not cf.empty:
                        cf.columns = [str(c).split(' ')[0] for c in cf.columns]
                        st.dataframe(cf, use_container_width=True)
                    else:
                        st.write("No Cash Flow data available.")
                        
                with tab4:
                    st.write(info.get('longBusinessSummary', 'No description available.'))
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Website:** {info.get('website', 'N/A')}")
            else:
                st.warning("Could not retrieve company info. Please check if the ticker symbol is correct.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

st.markdown("<br>", unsafe_allow_html=True)
# Auto-refresh button
if st.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.rerun()

# Auto refresh every 5 mins
time.sleep(300)
st.rerun()
