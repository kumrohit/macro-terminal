# Macro Terminal — Codebase Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & File Structure](#2-architecture--file-structure)
3. [Dependencies](#3-dependencies)
4. [app.py — Streamlit Dashboard](#4-apppy--streamlit-dashboard)
   - [Configuration & Constants](#41-configuration--constants)
   - [CSS Theming](#42-css-theming)
   - [Data Fetching Functions](#43-data-fetching-functions)
   - [Sentiment Analysis Functions](#44-sentiment-analysis-functions)
   - [Sector Logic Functions](#45-sector-logic-functions)
   - [Dashboard UI Layout](#46-dashboard-ui-layout)
   - [Sidebar Features](#47-sidebar-features)
5. [sentiment_scorer.py — CLI Batch Scorer](#5-sentiment_scorerpy--cli-batch-scorer)
   - [RSS Feed Fetching](#51-rss-feed-fetching)
   - [VADER Scoring](#52-vader-scoring)
   - [Gemini AI Scoring](#53-gemini-ai-scoring)
   - [Main Execution Flow](#54-main-execution-flow)
6. [Data Flow Diagram](#6-data-flow-diagram)
7. [Macro Regime Framework](#7-macro-regime-framework)
8. [Sentiment Scoring Pipeline](#8-sentiment-scoring-pipeline)
9. [Caching Strategy](#9-caching-strategy)
10. [Known Issues & Code Notes](#10-known-issues--code-notes)
11. [Environment Setup](#11-environment-setup)

---

## 1. Project Overview

**Macro Terminal** is a real-time, Bloomberg Terminal-inspired financial intelligence dashboard. It aggregates macro news from multiple RSS feeds, scores market sentiment using two independent methods (VADER lexicon and Google Gemini AI), and overlays live global market data to help users identify the current macroeconomic regime and take actionable investment decisions.

### Core Capabilities

| Feature | Description |
|---|---|
| Live Market Data | Prices and % changes for major global indices and VIX measures |
| Dual Sentiment Engine | VADER (lexicon-based) + Gemini 2.5 Flash (AI-based) |
| Macro Regime Classifier | Classifies market environment as Goldilocks, Reflation, Stagflation, or Recession |
| Sector Rotation Advisor | Recommends overweight/underweight sectors per regime |
| Economic Calendar | High-impact events from ForexFactory (weekly XML feed) |
| Company Financials | Full income statement, balance sheet, cash flow from Yahoo Finance |
| CSV Export | Download timestamped sentiment analysis for backtesting |
| Backtesting Stub | Compares current sentiment signal against S&P 500 3-month history |

---

## 2. Architecture & File Structure

```
macro-terminal/
│
├── app.py                  # Main Streamlit web application (UI + logic)
├── sentiment_scorer.py     # Standalone CLI script for batch sentiment scoring
├── requirements.txt        # Python package dependencies
└── README.md               # Project title placeholder
```

The project is split into two entry points:

- **`app.py`** — Run with `streamlit run app.py`. This is the full interactive dashboard. All functions here are optimized for Streamlit's caching and reactive rendering model.
- **`sentiment_scorer.py`** — Run directly with `python sentiment_scorer.py`. This is a headless CLI tool intended for batch processing, saving CSVs for later backtesting, and scheduled runs outside of the UI.

---

## 3. Dependencies

From `requirements.txt`:

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `feedparser` | Parsing RSS/Atom feeds |
| `requests` | HTTP fetching of RSS feeds and Yahoo Finance search API |
| `pandas` | DataFrame manipulation for all tabular data |
| `vaderSentiment` | Lexicon-based sentiment scoring (VADER) |
| `yfinance` | Fetching stock/index price data from Yahoo Finance |
| `google-genai` | Google Gemini AI API client |
| `plotly` | Interactive charts (sector performance bar chart) |
| `numpy` | Numerical operations used in backtesting |

---

## 4. app.py — Streamlit Dashboard

### 4.1 Configuration & Constants

```python
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
```

Two dictionaries define the ticker symbols monitored by the terminal:

- **`INDICES`**: The six primary global market indices displayed in the equities banner.
- **`VOLATILITY_INDICES`**: The three volatility gauges (VIX instruments) shown separately. Their `delta_color` is set to `"inverse"` so that rising volatility is shown as red (bad) rather than green.

```python
RSS_FEEDS = {
    "Investing.com (Macro)": "...",
    "CNBC (World)": "...",
    "Yahoo Finance": "..."
}
```

Three RSS sources are polled for live headlines. The feeds cover global macro news (Investing.com), world business news (CNBC), and general finance (Yahoo Finance).

---

### 4.2 CSS Theming

The dashboard injects a custom `<style>` block via `st.markdown(..., unsafe_allow_html=True)` to achieve a Bloomberg Terminal dark aesthetic:

| CSS Rule | Effect |
|---|---|
| `.stApp { background-color: #0E1117 }` | Near-black page background |
| `[data-testid="stMetricValue"]` | IBM Plex Mono monospace font at 1.8rem for price metrics |
| `section[data-testid="stSidebar"]` | Dark sidebar with a subtle border |
| `.macro-card` | Reusable dark card component with border and padding |
| `.news-title { color: #00ff00 }` | Green headlines (terminal green) |
| `a { color: #00ffff }` | Cyan hyperlinks |

---

### 4.3 Data Fetching Functions

#### `fetch_news()` — `@st.cache_data(ttl=300)`

Fetches the top 10 headlines from each of the three RSS feeds. Cache TTL is 5 minutes to avoid aggressive polling.

**Flow:**
1. Loop over `RSS_FEEDS`.
2. Use `requests.get()` with a browser-like `User-Agent` header to bypass basic bot detection.
3. Parse the raw bytes with `feedparser.parse()`.
4. Extract `title`, `link`, `published`, and `source` for each entry.
5. Return a combined `pd.DataFrame`.

**Error handling:** Each feed is in a `try/except` block, so one failing feed does not block others. Errors are shown in the sidebar.

---

#### `fetch_calendar()` — `@st.cache_data(ttl=3600)`

Pulls the weekly high-impact economic events from the ForexFactory XML feed.

**Flow:**
1. Fetch the ForexFactory XML URL.
2. Check response bytes for signs of rate-limiting (`Rate Limited` or `<!DOCTYPE html>`). If detected, shows a sidebar warning and returns a placeholder DataFrame.
3. Parse XML manually using `xml.etree.ElementTree` to avoid `lxml` dependency issues.
4. Filter only events where `impact == 'High'`.
5. Extract Date, Time, Currency, Event title, Actual, Forecast, and Previous values.
6. Return as a `pd.DataFrame`.

Cache TTL is 1 hour since the economic calendar is stable within a day.

---

#### `fetch_ticker_data(ticker_dict)` — `@st.cache_data(ttl=60)`

Fetches last 2 days of OHLCV data for every ticker in the provided dictionary.

**Flow:**
1. For each `(name, ticker)` pair, call `yf.Ticker(ticker).history(period="2d")`.
2. If 2 rows are available, compute `change = current - prev_close` and `pct_change`.
3. If only 1 row exists (e.g., a market that didn't trade yesterday), show price with 0 change.
4. Store results in a `data` dict keyed by display name.

Cache TTL is 60 seconds for near-live pricing.

---

#### `fetch_5y_chart_data(ticker)` — `@st.cache_data(ttl=3600)`

Returns a `pd.Series` of closing prices over 5 years for the selected index. Used for the historical chart expander. Cached for 1 hour since historical data is stable.

---

### 4.4 Sentiment Analysis Functions

#### `fetch_vader_sentiment(news_df)`

Computes a simple average VADER compound score across the top 15 headlines.

**How VADER works:** VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment tool specifically tuned for financial and social media text. It returns a `compound` score between -1.0 (most negative) and +1.0 (most positive).

**Verdict thresholds:**
- Score > 0.1 → **BULLISH 🟢**
- Score < -0.1 → **BEARISH 🔴**
- Otherwise → **NEUTRAL ⚪**

---

#### `fetch_vader_ai_commentary(news_df, vader_score, api_key)`

Uses Gemini 2.5 Flash to generate a 2–3 sentence natural language explanation of *why* the VADER score landed where it did. The AI does not re-score the sentiment; it explains the lexical signals VADER reacted to.

**System prompt role:** Quantitative financial analyst explaining VADER's mechanical reactions to words and themes.

**Temperature:** 0.2 (slightly creative but mostly factual).

---

#### `fetch_sentiment_score(news_df, api_key, market_context=...)`

The primary AI sentiment function. Sends the top 10 headlines to Gemini 2.5 Flash and requests a structured response identifying the macro regime, sentiment score, rationale, commentary, and transmission vector.

**System prompt role:** Lead Equity Quantitative Strategist analysing headlines through a DCF/Equity Risk Premium framework.

**Structured output format:**
```
Regime: [Goldilocks/Reflation/Stagflation/Recession]
Score: [-1.0 to 1.0]
Rationale: [1 sentence]
Commentary: [2–3 sentences]
Vector: [Numerator/Denominator/Both]
```

**Response parsing:** The function splits the response on newlines, then splits each line on the first `:` to build a `data` dict. Markdown artifacts like `*` and `+` are stripped from values before use.

**Temperature:** 0.1 (near-deterministic for consistency).

> **Note:** There is a dead code block below the first `return data` statement in this function (lines 377–434 in `app.py`). It is unreachable because the function returns before reaching it. It appears to be a legacy version of the same parsing logic.

---

### 4.5 Sector Logic Functions

#### `get_sector_scores(regime)` → `pd.Series`

Returns a `pd.Series` of 10 GICS sectors with expected relative performance scores (-1.0 to +1.0) based on the active macro regime. Used to render a Plotly horizontal bar chart.

| Regime | Leaders | Laggards |
|---|---|---|
| **Goldilocks** | Technology, Consumer Discretionary | Utilities, Consumer Staples |
| **Reflation** | Energy, Materials, Industrials | Technology, Utilities |
| **Stagflation** | Healthcare, Consumer Staples | Energy, Real Estate |
| **Recession** | Utilities, Healthcare, Consumer Staples | Energy, Materials, Financials |

---

#### `get_sector_status(regime)` → `dict`

Returns a simpler `{"Bullish": [...], "Bearish": [...]}` dictionary used to render the overweight/underweight recommendation cards (`st.success` / `st.error` badges).

---

### 4.6 Dashboard UI Layout

The UI is organised into a top header, a sidebar, and three main tabs.

```
┌─────────────────────────────────────────────────────┐
│  🌐 GLOBAL MACRO TERMINAL          [NYSE Status]    │
├────────────┬────────────────────────────────────────┤
│  SIDEBAR   │  Tab 1: Markets & Sentiment             │
│            │  Tab 2: Macro News & Calendar           │
│  API Key   │  Tab 3: Company Financials              │
│  Links     │                                         │
│  10Y Chart │  [Force Refresh]                        │
└────────────┴────────────────────────────────────────┘
```

**Market Status Indicator:** Checks `datetime.datetime.now()` against weekday and hour (9–16) to display a 🟢/🔴 NYSE open/closed badge.

---

#### Tab 1 — Markets & Sentiment

1. **Global Equities row:** `st.columns(n)` with `st.metric()` cards for each index. `delta` shows absolute and percentage change.
2. **Volatility row:** Same layout for VIX instruments, with `delta_color="inverse"`.
3. **5Y Chart expander:** A `st.selectbox` lets users pick any tracked index/VIX. Chart rendered with `st.line_chart`.
4. **VADER Sentiment Banner:** Always shown. Colour-coded heading (green/red/white) + AI-generated commentary paragraph.
5. **AI Regime Block** (requires Gemini API key):
   - Left column: `macro-card` showing Regime name, badge colour, vector, and rationale.
   - Right column: `macro-card` showing overweight/underweight sector badges.
6. **Sector Performance Chart:** Plotly `px.bar` horizontal bar chart, coloured on a red→yellow→green scale.

---

#### Tab 2 — Macro News & Calendar

Split into two columns:

- **Left (2/3 width):** Full news feed rendered as styled HTML links (`<div class='news-title'>`).
- **Right (1/3 width):** ForexFactory high-impact economic calendar as a `st.dataframe`.

---

#### Tab 3 — Company Financials

A search bar accepts any company name or ticker symbol. Ticker resolution is handled by querying the Yahoo Finance search endpoint:

```
https://query2.finance.yahoo.com/v1/finance/search?q={query}
```

The first result's `symbol` field is used as the resolved ticker.

Once resolved, four sub-tabs are shown:
- **Income Statement** (`tkr.financials`)
- **Balance Sheet** (`tkr.balance_sheet`)
- **Cash Flow** (`tkr.cashflow`)
- **Company Profile** (business summary, sector, industry, website)

Key metrics displayed: Current Price, Market Cap, P/E Ratio, 52-Week High.

---

### 4.7 Sidebar Features

| Button | Behaviour |
|---|---|
| `Run Sentiment Analysis` | Shows an info message (stub — does not call `sentiment_scorer.py`) |
| `Export Sentiment CSV` | Runs AI + VADER scoring and offers a timestamped CSV download |
| `Backtest Sentiment vs S&P` | Fetches 3-month S&P history and computes correlation, avg daily return, volatility, and a simulated Sharpe ratio |
| `10Y Yield Watch` | Displays an intraday sparkline of the 10Y Treasury yield |
| `Force Refresh` | Clears all Streamlit cache and reruns the app |

**Auto-refresh:** The app calls `time.sleep(300)` followed by `st.rerun()` at the bottom of the script, creating an automatic 5-minute refresh cycle.

---

## 5. sentiment_scorer.py — CLI Batch Scorer

This is a standalone Python script designed to run outside Streamlit. It outputs results to the terminal and saves a CSV file for backtesting.

### 5.1 RSS Feed Fetching

`fetch_top_news(limit_per_feed=5)` pulls the top 5 headlines from each of the three feeds (15 total). The same browser-like `User-Agent` header is used to bypass bot filters. Results are returned as a `pd.DataFrame`.

---

### 5.2 VADER Scoring

`add_vader_sentiment(df)` runs `SentimentIntensityAnalyzer().polarity_scores()` on each headline's `title` field and appends the `compound` score as a new `vader_score` column.

---

### 5.3 Gemini AI Scoring

`analyze_sentiment(df, market_context=...)` sends each headline **individually** to Gemini 2.5 Flash (one API call per headline), unlike `app.py`'s `fetch_sentiment_score()` which sends all headlines in a single batch.

**Per-headline output fields:**

| Field | Description |
|---|---|
| `Score` | Float from -1.0 to +1.0 |
| `Rationale` | 1-sentence equity transmission explanation |
| `Regime` | Goldilocks / Reflation / Stagflation / Recession |
| `Vector` | Numerator / Denominator / Both |

A `time.sleep(1)` delay is inserted between calls to avoid rate-limiting.

Results are appended to the DataFrame as: `sentiment_score`, `rationale`, `regime`, `vector`.

---

### 5.4 Main Execution Flow

```
1. fetch_top_news()          → news_df (15 headlines)
2. add_vader_sentiment()     → adds vader_score column
3. analyze_sentiment()       → adds AI score, rationale, regime, vector columns
4. Compute averages          → avg AI score, avg VADER score, combined average
5. Determine dominant regime → mode() of regime column
6. Print verdict             → Bullish / Bearish / Neutral
7. Save CSV                  → market_sentiment_YYYYMMDD_HHMM.csv
```

**Verdict thresholds (combined average):**
- > 0.15 → Bullish 🟢
- < -0.20 → Bearish 🔴
- Otherwise → Neutral ⚪

**API Key requirement:** The script reads `GEMINI_API_KEY` from the environment. If not set, it raises a `ValueError` with instructions for setting the variable.

---

## 6. Data Flow Diagram

```
                    ┌─────────────────────┐
                    │   RSS Feeds (3x)    │
                    │ Investing.com       │
                    │ CNBC World          │
                    │ Yahoo Finance       │
                    └────────┬────────────┘
                             │ feedparser
                             ▼
                    ┌─────────────────────┐
                    │   news_df           │
                    │ title, link,        │
                    │ source, published   │
                    └────┬────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
   ┌──────────────────┐   ┌──────────────────────┐
   │  VADER Scorer    │   │  Gemini 2.5 Flash    │
   │ (lexicon-based)  │   │  (AI-based, batch)   │
   │ compound: float  │   │  Regime, Score,      │
   └────────┬─────────┘   │  Vector, Rationale   │
            │             └──────────┬───────────┘
            └──────────┬─────────────┘
                       ▼
           ┌───────────────────────┐
           │  Combined Verdict     │
           │  + Sector Rotation    │
           │  + Chart Rendering    │
           └───────────────────────┘
```

---

## 7. Macro Regime Framework

The terminal classifies the market into four regimes using a 2×2 Growth/Inflation matrix:

```
                  INFLATION
                  Rising      Falling
              ┌───────────┬────────────┐
  G  Rising   │ REFLATION │ GOLDILOCKS │
  R           │           │            │
  O  Falling  │STAGFLATION│  RECESSION │
  W           │           │            │
  T           └───────────┴────────────┘
  H
```

| Regime | Growth | Inflation | Equity Outlook |
|---|---|---|---|
| **Goldilocks** | Rising/Stable | Falling | Highest returns; long high-beta (Tech, Consumer) |
| **Reflation** | Rising | Rising | Bullish for Value/Cyclicals (Energy, Materials) |
| **Stagflation** | Falling | Rising | Highly bearish; defensive sectors survive |
| **Recession** | Falling | Falling | Bearish, but rate cut hopes; Utilities, Healthcare |

---

## 8. Sentiment Scoring Pipeline

Two independent methods run in parallel and are both surfaced to the user:

### VADER (Always runs, no API key needed)

- **Type:** Lexicon-based (dictionary lookup + grammatical rules)
- **Speed:** Instantaneous
- **Strength:** No API dependency; consistent and reproducible
- **Weakness:** Cannot understand context, irony, or financial jargon nuance
- **Output:** Single compound float (-1.0 to 1.0), averaged across 15 headlines

### Gemini 2.5 Flash (Requires Gemini API key)

- **Type:** Large language model
- **Speed:** 1–3 seconds per batch (app.py) or ~1s per headline (sentiment_scorer.py)
- **Strength:** Understands financial context, DCF frameworks, and regime dynamics
- **Weakness:** Requires API key; non-deterministic even at low temperature; can be rate-limited
- **Output:** Regime label, score, rationale, commentary, vector

Both scores are displayed side by side. The AI score also drives the regime banner and sector rotation recommendations.

---

## 9. Caching Strategy

Streamlit's `@st.cache_data` decorator is used throughout `app.py` to prevent redundant network calls during reactive rerenders.

| Function | TTL | Rationale |
|---|---|---|
| `fetch_news()` | 300s (5 min) | News refreshes frequently; but not every render |
| `fetch_calendar()` | 3600s (1 hr) | Economic calendar is stable within a day |
| `fetch_ticker_data()` | 60s (1 min) | Near-live prices; short TTL justified |
| `fetch_5y_chart_data()` | 3600s (1 hr) | Historical data is stable |

`sentiment_scorer.py` has no caching since it is designed as a one-shot CLI batch run.

---

## 10. Known Issues & Code Notes

| Issue | Location | Detail |
|---|---|---|
| Dead code block | `app.py`, lines 377–434 | Unreachable code after the first `return data` statement in `fetch_sentiment_score()`. A full duplicate of the prior function body exists below the return. Safe to delete. |
| Backtest stub | `app.py`, sidebar button | The "Run Sentiment Analysis" button only shows an info message. It does not invoke `sentiment_scorer.py` or any scoring logic. |
| NYSE hours approximation | `app.py`, line 441 | Market status uses a simplified `9 <= hour < 16` check with no timezone conversion. A user in IST will see incorrect status. Should use `pytz` or `zoneinfo` with `America/New_York`. |
| `time.sleep(300)` at end of app | `app.py`, line 800 | This blocks the entire Streamlit server thread for 5 minutes. Should be replaced with `st.rerun()` combined with a session state timer, or Streamlit's `st_autorefresh` component. |
| Batch vs per-headline scoring | Both files | `app.py` sends all headlines in one Gemini call (efficient). `sentiment_scorer.py` sends one call per headline (expensive, 15x API calls). Inconsistency in approach. |

---

## 11. Environment Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set Gemini API key

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

### Run the CLI batch scorer

```bash
python sentiment_scorer.py
```

The CLI scorer saves a CSV file named `market_sentiment_YYYYMMDD_HHMM.csv` in the working directory for use in backtesting or further analysis.
