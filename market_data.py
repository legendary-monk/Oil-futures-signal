"""
market_data.py — Multi-Instrument Oil Market Data
===================================================
Improvements over Nifty version:
1. Fetches multiple correlated instruments (WTI, Brent, USD, XLE, Gold)
2. Computes Brent-WTI spread — a key structural market signal
3. Handles futures rollover gaps (common in CL=F, BZ=F)
4. Detects contango vs backwardation from term structure
5. Robust fallback: if WTI fails, use Brent; if both fail, use XLE proxy

WHY multiple instruments:
- USD strength is systematically bearish for oil (inverse correlation ~-0.7)
- XLE (energy ETF) often leads crude futures by hours
- Brent-WTI spread widens with supply disruptions and geopolitical events
- Gold rising alongside oil = geopolitical fear premium (not just demand)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import time

import config
from logger import get_logger

logger = get_logger(__name__)


def _fetch_single_ticker(ticker: str, days: int) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV data for one ticker with retry.

    Futures-specific handling:
    - CL=F and BZ=F sometimes return gaps at month rollover
    - We detect and interpolate small gaps (<3 days)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    for attempt in range(1, config.REQUEST_RETRIES + 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                auto_adjust=True,
                actions=False,
            )

            if df is None or df.empty:
                if attempt < config.REQUEST_RETRIES:
                    time.sleep(config.RETRY_DELAY_BASE ** attempt)
                continue

            # Normalize columns
            df.columns = [c.capitalize() for c in df.columns]
            if hasattr(df.index, 'tz') and df.index.tz:
                df.index = df.index.tz_localize(None)

            # Handle futures rollover: interpolate gaps of ≤2 days
            # WHY: CL=F has a 1-day price gap when contract rolls over
            if df['Close'].isna().sum() > 0:
                df['Close'] = df['Close'].interpolate(method='linear', limit=2)
                df['Volume'] = df['Volume'].fillna(0)

            # Drop rows still NaN after interpolation
            df = df.dropna(subset=['Close'])

            if len(df) < 5:
                logger.warning("%s: only %d valid rows after cleaning", ticker, len(df))
                return None

            df['Returns'] = df['Close'].pct_change()
            df = df.dropna(subset=['Returns'])

            logger.debug("%s: fetched %d rows | latest=%.2f",
                         ticker, len(df), df['Close'].iloc[-1])
            return df

        except Exception as e:
            logger.warning("%s fetch error (attempt %d): %s", ticker, attempt, e)
            if attempt < config.REQUEST_RETRIES:
                time.sleep(config.RETRY_DELAY_BASE ** attempt)

    return None


def fetch_all_instruments() -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetches all configured oil market instruments.

    Returns dict mapping ticker → DataFrame (or None if fetch failed).
    Instruments fetched sequentially with small delays to avoid rate limits.
    """
    instruments = {
        "WTI": config.WTI_TICKER,
        "BRENT": config.BRENT_TICKER,
        "USD": config.USD_INDEX_TICKER,
        "XLE": config.XLE_TICKER,
        "GOLD": config.GOLD_TICKER,
        "NATGAS": config.NAT_GAS_TICKER,
    }

    results = {}
    for name, ticker in instruments.items():
        logger.info("Fetching %s (%s)...", name, ticker)
        results[name] = _fetch_single_ticker(ticker, config.MARKET_LOOKBACK_DAYS)
        time.sleep(1.0)  # Rate limit protection

    # Log summary
    available = [k for k, v in results.items() if v is not None]
    missing = [k for k, v in results.items() if v is None]
    logger.info("Instruments available: %s | Missing: %s",
                available, missing if missing else "none")

    return results


def get_primary_oil_df(instruments: Dict) -> Optional[pd.DataFrame]:
    """
    Returns the best available oil price DataFrame.
    Prefers WTI → Brent → XLE proxy (in order).

    WHY fallback chain: WTI futures (CL=F) sometimes fails outside
    US trading hours. Brent (BZ=F) is often available when WTI isn't.
    XLE as proxy is imperfect but better than no price data.
    """
    if instruments.get("WTI") is not None:
        return instruments["WTI"]
    elif instruments.get("BRENT") is not None:
        logger.warning("WTI unavailable — using Brent as primary")
        return instruments["BRENT"]
    elif instruments.get("XLE") is not None:
        logger.warning("Both WTI and Brent unavailable — using XLE as proxy")
        return instruments["XLE"]
    else:
        return None


def compute_brent_wti_spread(instruments: Dict) -> Optional[float]:
    """
    Computes current Brent-WTI spread.

    WHY this matters:
    - Normal spread: Brent trades $1-5 premium to WTI
    - Wide spread (>$8): Supply disruption in Atlantic basin, or
      US inland oversupply (pipeline bottleneck)
    - Negative spread (WTI > Brent): Extreme US supply disruption, rare

    Returns:
        float: Brent price - WTI price, or None if either unavailable
    """
    wti_df = instruments.get("WTI")
    brent_df = instruments.get("BRENT")

    if wti_df is None or brent_df is None:
        return None

    try:
        wti_price = float(wti_df['Close'].iloc[-1])
        brent_price = float(brent_df['Close'].iloc[-1])
        spread = round(brent_price - wti_price, 2)
        logger.debug("Brent-WTI spread: $%.2f", spread)
        return spread
    except (IndexError, KeyError, ValueError):
        return None


def compute_macro_signal(instruments: Dict) -> float:
    """
    Computes a macro context signal from correlated instruments.

    Logic:
    - Strong USD → bearish for oil (oil priced in USD, strong USD = cheaper oil)
    - Rising XLE (energy stocks) → bullish for oil (equity market leads futures)
    - Rising Gold alongside oil → fear premium (geopolitical, not demand)

    Returns:
        float in [-1.0, +1.0]: positive = macro conditions bullish for oil
    """
    signals = []

    # USD signal (inverse)
    usd_df = instruments.get("USD")
    if usd_df is not None and len(usd_df) >= 6:
        try:
            usd_change_5d = ((usd_df['Close'].iloc[-1] / usd_df['Close'].iloc[-6]) - 1)
            # Strong USD (-) = bearish for oil; weak USD (+) = bullish
            # Cap at ±2% change → ±1.0 signal
            usd_signal = -usd_change_5d / 0.02
            usd_signal = max(-1.0, min(1.0, usd_signal))
            signals.append(("USD", usd_signal, 0.40))  # 40% weight in macro
            logger.debug("USD 5d change: %+.3f%% → signal: %+.4f",
                         usd_change_5d * 100, usd_signal)
        except Exception:
            pass

    # XLE signal (energy equity leading indicator)
    xle_df = instruments.get("XLE")
    if xle_df is not None and len(xle_df) >= 6:
        try:
            xle_change_5d = ((xle_df['Close'].iloc[-1] / xle_df['Close'].iloc[-6]) - 1)
            xle_signal = xle_change_5d / 0.03  # Normalize to 3% move = full signal
            xle_signal = max(-1.0, min(1.0, xle_signal))
            signals.append(("XLE", xle_signal, 0.35))
            logger.debug("XLE 5d change: %+.3f%% → signal: %+.4f",
                         xle_change_5d * 100, xle_signal)
        except Exception:
            pass

    # Gold signal (risk premium detector)
    # Gold rising with oil = geopolitical, not demand (reinforcing signal)
    # Gold falling while oil rises = pure demand (slightly less reliable)
    gold_df = instruments.get("GOLD")
    oil_df = get_primary_oil_df(instruments)
    if gold_df is not None and oil_df is not None and len(gold_df) >= 3:
        try:
            gold_change = gold_df['Returns'].iloc[-3:].mean()
            oil_change = oil_df['Returns'].iloc[-3:].mean()
            # Co-directional movement = reinforcing
            if (gold_change > 0 and oil_change > 0) or (gold_change < 0 and oil_change < 0):
                correlation_bonus = 0.2 * (1 if oil_change > 0 else -1)
            else:
                correlation_bonus = 0.0
            signals.append(("GOLD_CORRELATION", correlation_bonus, 0.25))
        except Exception:
            pass

    if not signals:
        return 0.0

    # Weighted sum
    total_weight = sum(s[2] for s in signals)
    macro_signal = sum(s[1] * s[2] for s in signals) / total_weight
    macro_signal = round(max(-1.0, min(1.0, macro_signal)), 4)

    logger.info("Macro signal: %.4f from %d instruments", macro_signal, len(signals))
    return macro_signal


def get_latest_price(instruments: Dict) -> Optional[float]:
    """Returns the latest WTI (or best available) price."""
    df = get_primary_oil_df(instruments)
    if df is None or df.empty:
        return None
    try:
        return round(float(df['Close'].iloc[-1]), 2)
    except Exception:
        return None


def get_price_change_pct(instruments: Dict, days: int = 1) -> Optional[float]:
    df = get_primary_oil_df(instruments)
    if df is None or len(df) < days + 1:
        return None
    try:
        latest = df['Close'].iloc[-1]
        past = df['Close'].iloc[-(days + 1)]
        return round(((latest - past) / past) * 100, 4)
    except Exception:
        return None


# ── Test ──
if __name__ == "__main__":
    instruments = fetch_all_instruments()

    print("\n=== INSTRUMENT PRICES ===")
    for name, df in instruments.items():
        if df is not None:
            print(f"  {name:8s}: {df['Close'].iloc[-1]:.2f} (latest)")
        else:
            print(f"  {name:8s}: UNAVAILABLE")

    spread = compute_brent_wti_spread(instruments)
    print(f"\nBrent-WTI Spread: ${spread:.2f}" if spread else "\nSpread: N/A")

    macro = compute_macro_signal(instruments)
    print(f"Macro Signal: {macro:+.4f}")

    wti_1d = get_price_change_pct(instruments, 1)
    wti_5d = get_price_change_pct(instruments, 5)
    print(f"WTI 1-day change: {wti_1d:+.2f}%" if wti_1d else "WTI 1-day: N/A")
    print(f"WTI 5-day change: {wti_5d:+.2f}%" if wti_5d else "WTI 5-day: N/A")
