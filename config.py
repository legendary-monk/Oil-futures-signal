"""
config.py — Oil Signal System Configuration
=============================================
All tunable parameters in one place.

Key differences from the Nifty system:
- Three data sources instead of two (news + market + Polymarket)
- Polymarket adds forward-looking probability signals
- Oil has regime-specific behavior (backwardation vs contango matters)
- Multiple correlated instruments (WTI, Brent, USD index, nat gas)
"""

import os
import logging

# ─────────────────────────────────────────────
# TELEGRAM (REQUIRED — EDIT THESE)
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

# ─────────────────────────────────────────────
# OIL MARKET INSTRUMENTS
# ─────────────────────────────────────────────
# Primary instrument
WTI_TICKER = "CL=F"          # WTI Crude Oil Futures (NYMEX front-month)
BRENT_TICKER = "BZ=F"        # Brent Crude Futures (ICE front-month)

# Correlated instruments — used to detect macro context
USD_INDEX_TICKER = "DX-Y.NYB"    # US Dollar Index (oil priced in USD — inverse correlation)
NAT_GAS_TICKER = "NG=F"          # Natural Gas (energy sector sentiment)
XLE_TICKER = "XLE"               # Energy sector ETF (equity market's view on energy)
GOLD_TICKER = "GC=F"             # Gold (risk-off / geopolitical stress proxy)

# Primary trading instrument for signal (WTI is most liquid)
PRIMARY_TICKER = WTI_TICKER

# How many calendar days of price history to fetch
MARKET_LOOKBACK_DAYS = 60    # WHY 60: Oil needs longer window for regime detection

# ─────────────────────────────────────────────
# POLYMARKET CONFIGURATION
# ─────────────────────────────────────────────
# Polymarket's public CLOB (Central Limit Order Book) API
# No API key needed for reading market data — fully public
POLYMARKET_API_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"   # For market search

# Search terms to find oil-relevant prediction markets
# WHY broad terms: Polymarket market titles vary. Cast wide net, filter by relevance.
POLYMARKET_SEARCH_TERMS = [
    "oil",
    "crude",
    "OPEC",
    "petroleum",
    "brent",
    "WTI",
    "energy prices",
    "oil production",
    "Saudi",
    "Russia oil",
]

# Maximum number of Polymarket markets to analyze
MAX_POLYMARKET_MARKETS = 20

# Minimum liquidity (total volume in USD) to trust a market
# WHY $5000: Illiquid markets have wide spreads and unreliable prices
MIN_MARKET_VOLUME_USD = 5000

# ─────────────────────────────────────────────
# NEWS RSS FEEDS (OIL-FOCUSED)
# ─────────────────────────────────────────────
RSS_FEEDS = [
    # Global energy news
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bloomberg.com/markets/news.rss",          # May redirect — fallback gracefully
    "https://oilprice.com/rss/main",                         # Oil-specific news site
    "https://www.eia.gov/rss/news.xml",                      # US Energy Info Admin (authoritative)
    "https://www.opec.org/opec_web/en/press_room/rss.htm",  # OPEC official
    # Backup feeds
    "https://feeds.feedburner.com/OilPrice-Breaking-News",
    "https://www.rigzone.com/news/rss/rigzone_latest.aspx",
]

MAX_ARTICLES = 60          # More than Nifty system — oil news is global
MAX_ARTICLE_AGE_HOURS = 36 # WHY 36: OPEC decisions affect oil for 2+ days

# ─────────────────────────────────────────────
# SIGNAL ENGINE WEIGHTS
# WHY these allocations:
# Polymarket gets highest weight because it's forward-looking
# and aggregates informed market participants with real money.
# News sentiment is noisier for oil than equities.
# Technical trend matters but oil is more macro-driven than equity indices.
# ─────────────────────────────────────────────
WEIGHT_POLYMARKET = 0.30    # Prediction market probabilities
WEIGHT_SENTIMENT = 0.20     # News sentiment
WEIGHT_TREND = 0.20         # Technical price trend
WEIGHT_MACRO = 0.10         # USD / correlated instruments
WEIGHT_QUANT = 0.20         # Comprehensive quant stack

# Adaptive consensus overlay (still fully free/local).
# The engine boosts conviction only when multiple factors agree and
# dampens in disagreement regimes to reduce false positives.
CONSENSUS_BOOST_STRONG = 1.15   # 3+ aligned factors
CONSENSUS_DAMPEN_MIXED = 0.85   # 2v2 split / noisy regime
MIN_DIRECTIONAL_SCORE = 0.18    # below this, force NEUTRAL
MIN_FACTOR_PARTICIPATION = 2    # require at least 2 active factors

# Signal classification thresholds
BULLISH_THRESHOLD = 0.60
BEARISH_THRESHOLD = 0.40

# ─────────────────────────────────────────────
# FEATURE PARAMETERS
# ─────────────────────────────────────────────
TREND_WINDOW = 10          # WHY 10: Oil trends run longer than equity indices
VOLATILITY_WINDOW = 15
HIGH_VOLATILITY_MULTIPLIER = 1.4

# Oil-specific: OPEC meeting schedule detection
# If today is within N days of a scheduled OPEC meeting, flag elevated uncertainty
OPEC_MEETING_UNCERTAINTY_DAYS = 7

# ─────────────────────────────────────────────
# BACKTEST CONFIGURATION
# ─────────────────────────────────────────────
BACKTEST_START_DATE = "2022-01-01"
BACKTEST_END_DATE = None  # None -> today
BACKTEST_WARMUP_BARS = 80
BACKTEST_TX_COST_BPS = 3.0

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "oil_signal.log")
PREDICTIONS_FILE = os.path.join(BASE_DIR, "oil_predictions.csv")
POLYMARKET_CACHE_FILE = os.path.join(BASE_DIR, "polymarket_cache.json")
OPEC_CALENDAR_FILE = os.path.join(BASE_DIR, "data", "opec_calendar.json")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ─────────────────────────────────────────────
# NETWORK
# ─────────────────────────────────────────────
REQUEST_TIMEOUT = 20       # WHY 20: Polymarket API can be slower than RSS
REQUEST_RETRIES = 3
RETRY_DELAY_BASE = 2

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def validate_config():
    errors = []

    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        errors.append("TELEGRAM_TOKEN not set")

    if TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        errors.append("TELEGRAM_CHAT_ID not set")

    total_weight = WEIGHT_POLYMARKET + WEIGHT_SENTIMENT + WEIGHT_TREND + WEIGHT_MACRO + WEIGHT_QUANT
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Weights sum to {total_weight:.2f}, must sum to 1.0")

    return errors
