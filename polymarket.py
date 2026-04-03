"""
polymarket.py — Polymarket Prediction Market Ingestion
========================================================
Responsibility: Query Polymarket's public API for oil-related
prediction markets and extract a directional probability signal.

WHY Polymarket beats news sentiment for oil:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
News sentiment is BACKWARD-looking (reporting what happened).
Polymarket is FORWARD-looking (betting on what WILL happen).

Crucially, Polymarket participants stake real money. This means:
1. Uninformed traders lose money to informed ones over time
2. The market price converges toward the TRUE probability
3. This is empirically demonstrated — prediction markets beat
   expert forecasters on geopolitical events (Good Judgment Project)

For oil specifically:
- OPEC meeting outcome markets tell you what traders EXPECT
- "Will Brent exceed $X by [date]?" markets embed the market
  consensus on price direction, adjusted for geopolitical risk
- This is information that news sentiment cannot provide

How we extract a signal:
━━━━━━━━━━━━━━━━━━━━━━━
For each oil-relevant market:
  1. Determine if it's BULLISH or BEARISH for oil price
  2. Extract the YES probability (0.0–1.0)
  3. Convert to a directional oil signal:
     - BULLISH market with high YES prob → bullish signal
     - BEARISH market with high YES prob → bearish signal
  4. Weight by market liquidity (volume-weighted average)

API used: Polymarket Gamma API (market search) + CLOB API (prices)
Both are public — no API key required.

Limitations:
━━━━━━━━━━━━
- Polymarket focuses on US/global events; OPEC-specific markets
  are seasonal (appear before meetings, expire after)
- Some months have no relevant oil markets → returns neutral (0.0)
- API can be slow or return empty results on weekends
"""

import requests
import json
import time
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta

import config
from logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# MARKET CLASSIFICATION RULES
# ─────────────────────────────────────────────
# Maps keywords in market titles/descriptions to directional bias.
# BULLISH_FOR_OIL = market resolving YES means oil price likely goes UP
# BEARISH_FOR_OIL = market resolving YES means oil price likely goes DOWN

BULLISH_FOR_OIL_KEYWORDS = [
    # Supply cuts → less supply → higher price
    "opec cut", "production cut", "supply cut", "output cut",
    "saudi cut", "russia cut", "quota reduction",
    # Price ceiling breaches (YES = price exceeded threshold = bullish momentum)
    "oil above", "crude above", "brent above", "wti above",
    "oil exceed", "oil reach", "oil hit",
    "oil price high", "energy crisis",
    # Demand drivers
    "china demand", "demand surge", "demand recovery",
    # Geopolitical supply risk
    "iran sanction", "venezuela sanction", "strait of hormuz",
    "middle east escalat", "oil embargo",
    # OPEC solidarity
    "opec maintain", "opec extend", "opec agreement",
]

BEARISH_FOR_OIL_KEYWORDS = [
    # Supply increases → more supply → lower price
    "opec increase", "production increase", "output increase",
    "quota increase", "supply increase", "flood market",
    "us production record", "shale boom",
    # Price floor breaches (YES = price fell below threshold = bearish)
    "oil below", "crude below", "brent below", "wti below",
    "oil fall", "oil drop", "oil crash",
    "oil price low", "oil price decline",
    # Demand destruction
    "recession", "demand destruction", "demand fall", "china slowdown",
    "ev adoption", "peak oil demand",
    # OPEC breakdown
    "opec split", "opec disagreement", "opec collapse", "price war",
    # Dollar strength (oil priced in USD — strong dollar = cheaper oil)
    "dollar surge", "usd surge",
]


def _classify_market(title: str, description: str = "") -> Optional[str]:
    """
    Classifies a Polymarket market as BULLISH_OIL, BEARISH_OIL, or None.

    Args:
        title: Market title from Polymarket API
        description: Optional market description

    Returns:
        "BULLISH", "BEARISH", or None (irrelevant/ambiguous)

    WHY return None for ambiguous: Better to discard uncertain classification
    than to inject noise. We only use high-confidence directional markets.
    """
    text = (title + " " + description).lower()

    bullish_hits = sum(1 for kw in BULLISH_FOR_OIL_KEYWORDS if kw in text)
    bearish_hits = sum(1 for kw in BEARISH_FOR_OIL_KEYWORDS if kw in text)

    if bullish_hits == 0 and bearish_hits == 0:
        return None  # Not oil-relevant

    if bullish_hits > bearish_hits:
        return "BULLISH"
    elif bearish_hits > bullish_hits:
        return "BEARISH"
    else:
        return None  # Ambiguous — discard


def _is_market_active(market: Dict) -> bool:
    """
    Checks if a market is still open for trading (not resolved/closed).

    WHY filter resolved markets: A resolved market's price reflects
    the ACTUAL outcome, not future expectation. We only want markets
    still pricing in future uncertainty.
    """
    # Various Polymarket API fields for market status
    if market.get("closed", False):
        return False
    if market.get("archived", False):
        return False
    if market.get("resolved", False):
        return False

    # Check end date if available
    end_date_str = market.get("end_date_iso") or market.get("end_date")
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            if end_date < datetime.now(timezone.utc):
                return False
        except (ValueError, AttributeError):
            pass

    return True


def _extract_yes_probability(market: Dict) -> Optional[float]:
    """
    Extracts the current YES probability from a binary market.

    Polymarket stores prices in different fields depending on API version.
    We try multiple field paths for robustness.

    Returns:
        float in [0.0, 1.0] or None if unavailable
    """
    # Method 1: Direct probability field
    prob = market.get("outcomePrices") or market.get("outcome_prices")
    if prob:
        try:
            if isinstance(prob, list) and len(prob) >= 1:
                # First outcome is typically YES
                return float(prob[0])
            if isinstance(prob, str):
                prices = json.loads(prob)
                if isinstance(prices, list) and prices:
                    return float(prices[0])
        except (ValueError, json.JSONDecodeError, IndexError):
            pass

    # Method 2: tokens array
    tokens = market.get("tokens", [])
    for token in tokens:
        if token.get("outcome", "").upper() == "YES":
            price = token.get("price")
            if price is not None:
                return float(price)

    # Method 3: bestAsk / bestBid midpoint from CLOB
    best_ask = market.get("bestAsk")
    best_bid = market.get("bestBid")
    if best_ask is not None and best_bid is not None:
        try:
            return (float(best_ask) + float(best_bid)) / 2
        except (ValueError, TypeError):
            pass

    # Method 4: lastTradePrice
    ltp = market.get("lastTradePrice") or market.get("last_trade_price")
    if ltp is not None:
        try:
            return float(ltp)
        except (ValueError, TypeError):
            pass

    return None


def _get_market_volume(market: Dict) -> float:
    """
    Extracts total trading volume in USD for liquidity filtering.
    """
    for field in ["volume", "volumeNum", "volume_num", "liquidity"]:
        val = market.get(field)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 0.0


def _fetch_markets_from_gamma(search_term: str) -> List[Dict]:
    """
    Queries Polymarket's Gamma API to search for markets by keyword.

    The Gamma API is Polymarket's public market discovery endpoint.
    It supports text search and returns market metadata.

    Args:
        search_term: Keyword to search for

    Returns:
        List of market dicts from the API, possibly empty.
    """
    url = f"{config.POLYMARKET_GAMMA_API}/markets"
    params = {
        "search": search_term,
        "limit": 20,
        "active": "true",    # Only active (unresolved) markets
        "closed": "false",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=config.REQUEST_TIMEOUT,
            headers={"User-Agent": "OilSignalBot/1.0"}
        )

        if response.status_code == 200:
            data = response.json()
            # Gamma API returns either a list directly or {"markets": [...]}
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("markets", []) or data.get("data", [])
        else:
            logger.debug("Gamma API returned %d for search '%s'",
                         response.status_code, search_term)

    except requests.exceptions.Timeout:
        logger.warning("Polymarket Gamma API timeout for search '%s'", search_term)
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Polymarket (check internet)")
    except Exception as e:
        logger.debug("Unexpected error fetching Polymarket markets: %s", e)

    return []


def _load_cache() -> Dict:
    """Loads cached Polymarket data to reduce API calls."""
    try:
        if os.path.exists(config.POLYMARKET_CACHE_FILE):
            with open(config.POLYMARKET_CACHE_FILE, "r") as f:
                cache = json.load(f)
            # Cache expires after 4 hours
            cached_at = cache.get("cached_at", 0)
            age_hours = (time.time() - cached_at) / 3600
            if age_hours < 4:
                logger.debug("Using Polymarket cache (%.1f hours old)", age_hours)
                return cache
    except Exception:
        pass
    return {}


def _save_cache(data: Dict):
    """Saves Polymarket data to cache file."""
    try:
        data["cached_at"] = time.time()
        with open(config.POLYMARKET_CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.debug("Failed to save Polymarket cache: %s", e)


def fetch_polymarket_signal() -> Tuple[float, List[Dict]]:
    """
    Main entry point. Fetches all oil-relevant Polymarket markets and
    computes a directional signal.

    Algorithm:
    1. Search Polymarket for each configured search term
    2. Deduplicate markets by ID
    3. Filter: active, oil-relevant, sufficient liquidity
    4. Classify each as BULLISH or BEARISH for oil
    5. Extract YES probability for each
    6. Volume-weighted average of directional probabilities

    Signal interpretation:
    - Each market contributes a score:
      BULLISH market: score = YES_prob (high YES → bullish oil)
      BEARISH market: score = 1 - YES_prob (low YES → bearish oil)
    - Aggregate = volume-weighted mean of scores → [0, 1]
    - Convert to [-1, +1] range: signal = (aggregate - 0.5) × 2

    Returns:
        Tuple of:
        - signal_score (float): [-1.0, +1.0] where +1 = maximally bullish
        - markets_used (list): Dicts describing each market that contributed
    """
    # Try cache first
    cache = _load_cache()
    if cache.get("signal_score") is not None:
        logger.info("Polymarket: using cached signal (%.4f) from %d markets",
                    cache["signal_score"], len(cache.get("markets_used", [])))
        return cache["signal_score"], cache.get("markets_used", [])

    logger.info("Fetching Polymarket oil markets...")

    # Collect all markets across search terms
    all_markets: Dict[str, Dict] = {}  # keyed by market ID for deduplication

    for term in config.POLYMARKET_SEARCH_TERMS:
        markets = _fetch_markets_from_gamma(term)
        for m in markets:
            market_id = m.get("id") or m.get("condition_id") or m.get("slug")
            if market_id:
                all_markets[str(market_id)] = m
        time.sleep(0.5)  # WHY: Be polite to Polymarket's API

    logger.info("Found %d unique markets across all searches", len(all_markets))

    if not all_markets:
        logger.warning("No Polymarket data available — using neutral signal")
        return 0.0, []

    # Filter and classify
    usable_markets = []

    for market_id, market in all_markets.items():
        title = market.get("question") or market.get("title") or ""
        description = market.get("description") or ""

        # Must be active
        if not _is_market_active(market):
            continue

        # Must be oil-relevant
        direction = _classify_market(title, description)
        if direction is None:
            continue

        # Must have sufficient liquidity
        volume = _get_market_volume(market)
        if volume < config.MIN_MARKET_VOLUME_USD:
            logger.debug("Market below liquidity threshold ($%.0f): %s", volume, title[:50])
            continue

        # Must have a readable YES probability
        yes_prob = _extract_yes_probability(market)
        if yes_prob is None:
            logger.debug("Cannot extract probability for: %s", title[:50])
            continue

        # Clamp probability to valid range (API can return slightly out of bounds)
        yes_prob = max(0.01, min(0.99, yes_prob))

        usable_markets.append({
            "id": market_id,
            "title": title[:80],
            "direction": direction,
            "yes_prob": round(yes_prob, 4),
            "volume_usd": round(volume, 0),
            "oil_signal_score": round(yes_prob if direction == "BULLISH" else (1 - yes_prob), 4),
        })

        logger.debug(
            "[%s] %s | YES=%.2f | Vol=$%.0f | OilScore=%.4f",
            direction, title[:50], yes_prob, volume,
            usable_markets[-1]["oil_signal_score"]
        )

    if not usable_markets:
        logger.warning("No usable oil Polymarket markets found — returning neutral")
        return 0.0, []

    # Volume-weighted average of oil signal scores
    total_volume = sum(m["volume_usd"] for m in usable_markets)
    if total_volume == 0:
        # Fallback: simple average
        avg_score = sum(m["oil_signal_score"] for m in usable_markets) / len(usable_markets)
    else:
        avg_score = sum(
            m["oil_signal_score"] * m["volume_usd"] for m in usable_markets
        ) / total_volume

    # Convert from [0, 1] probability space to [-1, +1] signal space
    # WHY: 0.5 = neutral (equal bullish/bearish). Above = bullish. Below = bearish.
    signal_score = (avg_score - 0.5) * 2
    signal_score = round(max(-1.0, min(1.0, signal_score)), 4)

    logger.info(
        "Polymarket signal: %.4f | %d markets used | avg_score=%.4f | total_vol=$%.0f",
        signal_score, len(usable_markets), avg_score, total_volume
    )

    # Cache the result
    _save_cache({"signal_score": signal_score, "markets_used": usable_markets})

    return signal_score, usable_markets


def get_polymarket_summary(signal_score: float, markets: List[Dict]) -> Dict:
    """
    Produces a human-readable summary for the Telegram message.
    """
    if not markets:
        return {
            "signal_score": 0.0,
            "market_count": 0,
            "label": "No data",
            "top_markets": [],
        }

    bullish_markets = [m for m in markets if m["direction"] == "BULLISH"]
    bearish_markets = [m for m in markets if m["direction"] == "BEARISH"]

    # Sort by volume for "top markets" display
    top = sorted(markets, key=lambda m: m["volume_usd"], reverse=True)[:3]

    if signal_score > 0.15:
        label = "Bullish bias 🛢️"
    elif signal_score < -0.15:
        label = "Bearish bias 🛢️"
    else:
        label = "Neutral/mixed 🛢️"

    return {
        "signal_score": signal_score,
        "market_count": len(markets),
        "bullish_count": len(bullish_markets),
        "bearish_count": len(bearish_markets),
        "label": label,
        "top_markets": top,
    }


# ── Standalone test ──
if __name__ == "__main__":
    score, markets = fetch_polymarket_signal()
    print(f"\nPolymarket Oil Signal: {score:+.4f}")
    print(f"Markets used: {len(markets)}")
    if markets:
        print("\nTop markets by volume:")
        for m in sorted(markets, key=lambda x: x["volume_usd"], reverse=True)[:5]:
            print(f"  [{m['direction']:7s}] YES={m['yes_prob']:.2f} | "
                  f"Vol=${m['volume_usd']:,.0f} | {m['title'][:60]}")
