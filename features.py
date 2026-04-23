"""
features.py — Oil Market Feature Engineering
=============================================
Oil-specific improvements:
1. OPEC meeting calendar awareness
2. Term structure proxy (contango/backwardation detection from return patterns)
3. Oil-specific volatility regimes (oil is 2–3× more volatile than equity indices)
4. ATR (Average True Range) — more robust for futures than standard deviation
5. Momentum with oil-specific normalization
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List

import config
from logger import get_logger
from quant_stack import compute_quant_stack

logger = get_logger(__name__)

FeatureDict = Dict[str, Any]


def _load_opec_meeting_dates(as_of_date: Optional[date] = None) -> List[date]:
    """Loads OPEC meeting dates for the current and next year from JSON config."""
    path = Path(config.OPEC_CALENDAR_FILE)
    if not path.exists():
        logger.warning("OPEC calendar file missing: %s", path)
        return []

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.error("Failed to read OPEC calendar file %s: %s", path, e)
        return []

    today = as_of_date or date.today()
    years_to_use = {today.year, today.year + 1}
    dates: List[date] = []

    for year in years_to_use:
        raw_dates = payload.get(str(year), []) if isinstance(payload, dict) else []
        if not isinstance(raw_dates, list):
            continue
        for raw in raw_dates:
            try:
                dates.append(date.fromisoformat(raw))
            except (TypeError, ValueError):
                logger.warning("Invalid OPEC date %r in %s", raw, path)

    dates = sorted(set(dates))
    if not dates:
        logger.info("No OPEC dates available for %s/%s in %s", today.year, today.year + 1, path)
    return dates


def _days_to_nearest_opec_meeting(as_of_date: Optional[date] = None) -> Optional[int]:
    """
    Returns days until (or since) the nearest OPEC+ meeting.
    Negative = days since last meeting. Positive = days until next.
    """
    today = as_of_date or date.today()
    meeting_dates = _load_opec_meeting_dates(as_of_date=today)
    if not meeting_dates:
        return None

    differences = [(d - today).days for d in meeting_dates]
    return min(differences, key=abs)


def _is_opec_uncertainty_window(as_of_date: Optional[date] = None) -> bool:
    """Returns True if we're within OPEC_MEETING_UNCERTAINTY_DAYS of a meeting."""
    days = _days_to_nearest_opec_meeting(as_of_date=as_of_date)
    if days is None:
        return False
    return abs(days) <= config.OPEC_MEETING_UNCERTAINTY_DAYS


def _compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Average True Range — measures price volatility including gaps.

    WHY ATR over simple StdDev for oil:
    - Oil frequently has overnight gaps due to OPEC announcements
    - Standard deviation only measures close-to-close; ATR captures High-Low range
    - ATR is the industry standard for futures position sizing

    ATR = mean of True Range over N periods
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    """
    if len(df) < period + 1:
        return None

    try:
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        tr_list = []
        for i in range(1, len(close)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return None

        atr = np.mean(tr_list[-period:])
        current_price = close[-1]

        # Return as percentage of price (normalized ATR)
        if current_price > 0:
            return round((atr / current_price) * 100, 4)  # As % of price
        return None

    except Exception as e:
        logger.error("ATR computation failed: %s", e)
        return None


def _compute_trend_score(df: pd.DataFrame) -> Optional[float]:
    """
    Linear regression slope of closing prices over TREND_WINDOW.
    Same logic as Nifty system but with oil-calibrated normalization.

    For oil: cap at ±3% daily implied slope (oil trends stronger than indices).
    """
    window = config.TREND_WINDOW
    if len(df) < window:
        return None

    try:
        prices = df['Close'].iloc[-window:].values
        baseline = prices[0]
        if baseline == 0:
            return None

        normalized = (prices / baseline - 1) * 100
        x = np.arange(window, dtype=float)
        coeffs = np.polyfit(x, normalized, 1)
        slope = coeffs[0]

        # Oil can trend ±3% per day in strong moves
        trend_score = slope / 3.0
        return round(max(-1.0, min(1.0, trend_score)), 4)

    except Exception:
        return None


def _compute_volatility_regime(df: pd.DataFrame) -> Optional[str]:
    """
    Classifies oil volatility vs historical baseline.

    Oil-specific calibration: uses longer windows because oil has
    longer-cycle volatility regimes than equity indices.
    """
    vol_window = config.VOLATILITY_WINDOW
    baseline_window = 40  # WHY 40: Oil vol cycles run 4–6 weeks

    if len(df) < baseline_window:
        return None

    try:
        returns = df['Returns'].dropna()
        recent_vol = returns.iloc[-vol_window:].std() * np.sqrt(252)
        baseline_vol = returns.iloc[-baseline_window:].std() * np.sqrt(252)

        if baseline_vol == 0 or np.isnan(baseline_vol):
            return None

        ratio = recent_vol / baseline_vol
        logger.debug("Oil vol: recent=%.4f baseline=%.4f ratio=%.2f",
                     recent_vol, baseline_vol, ratio)

        if ratio >= config.HIGH_VOLATILITY_MULTIPLIER:
            return "HIGH"
        elif ratio <= (1 / config.HIGH_VOLATILITY_MULTIPLIER):
            return "LOW"
        else:
            return "NORMAL"

    except Exception:
        return None


def _compute_rsi(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Standard RSI calculation."""
    if len(df) < period + 1:
        return None

    try:
        closes = df['Close'].iloc[-(period + 5):]
        delta = closes.diff().dropna()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        avg_gain = gains.iloc[-period:].mean()
        avg_loss = losses.iloc[-period:].mean()

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    except Exception:
        return None


def _rsi_to_signal(rsi: Optional[float]) -> float:
    if rsi is None:
        return 0.0
    if rsi < 30: return 0.8
    elif rsi < 40: return 0.4
    elif rsi > 70: return -0.8
    elif rsi > 60: return -0.4
    else: return 0.0


def _compute_volume_trend(df: pd.DataFrame) -> Optional[float]:
    """
    Checks if volume is rising or falling vs recent baseline.

    WHY for oil: Volume on WTI futures is a reliable leading indicator.
    Rising price + rising volume = confirmed trend.
    Rising price + falling volume = weakening trend (potential reversal).

    Returns:
        float in [-1, +1]: positive = volume confirming trend
    """
    if 'Volume' not in df.columns or len(df) < 10:
        return None

    try:
        recent_vol = df['Volume'].iloc[-5:].mean()
        baseline_vol = df['Volume'].iloc[-15:-5].mean()

        if baseline_vol == 0 or np.isnan(baseline_vol):
            return None

        vol_ratio = recent_vol / baseline_vol - 1  # 0 = same, positive = rising

        # Rising volume (up to +50% above baseline) = bullish confirmation
        # If price is also rising, the signal is reinforced in features combination
        signal = max(-1.0, min(1.0, vol_ratio / 0.5))
        return round(signal, 4)

    except Exception:
        return None


def compute_features(instruments: Dict, macro_signal: float, as_of_date: Optional[date] = None) -> FeatureDict:
    """
    Main entry point. Computes all oil market features.

    Args:
        instruments: Dict from market_data.fetch_all_instruments()
        macro_signal: Float from market_data.compute_macro_signal()

    Returns:
        FeatureDict with all computed features. None values = unavailable.
    """
    # Get primary oil price data
    from market_data import get_primary_oil_df, get_latest_price, get_price_change_pct, compute_brent_wti_spread

    df = get_primary_oil_df(instruments)

    if df is None or df.empty:
        logger.error("No oil price data for feature computation")
        return _empty_features(macro_signal)

    logger.info("Computing oil features from %d trading days...", len(df))

    # Each feature in independent try block
    try:
        trend_score = _compute_trend_score(df)
    except Exception as e:
        logger.error("Trend error: %s", e)
        trend_score = None

    try:
        vol_regime = _compute_volatility_regime(df)
        vol_score = {"HIGH": -0.6, "NORMAL": 0.0, "LOW": 0.3}.get(vol_regime, 0.0)
    except Exception as e:
        logger.error("Volatility error: %s", e)
        vol_regime = None
        vol_score = 0.0

    try:
        rsi = _compute_rsi(df)
        rsi_signal = _rsi_to_signal(rsi)
    except Exception as e:
        logger.error("RSI error: %s", e)
        rsi = None
        rsi_signal = 0.0

    try:
        atr_pct = _compute_atr(df)
    except Exception as e:
        logger.error("ATR error: %s", e)
        atr_pct = None

    try:
        volume_signal = _compute_volume_trend(df)
    except Exception as e:
        logger.error("Volume signal error: %s", e)
        volume_signal = None

    # OPEC meeting context
    try:
        opec_days = _days_to_nearest_opec_meeting(as_of_date=as_of_date)
        opec_uncertainty = _is_opec_uncertainty_window(as_of_date=as_of_date)
    except Exception:
        opec_days = None
        opec_uncertainty = False

    # Price levels
    latest_close = get_latest_price(instruments)
    price_1d = get_price_change_pct(instruments, 1)
    price_5d = get_price_change_pct(instruments, 5)
    price_10d = get_price_change_pct(instruments, 10)
    brent_wti_spread = compute_brent_wti_spread(instruments)

    # Brent/WTI prices separately for Telegram
    wti_price = None
    brent_price = None
    try:
        if instruments.get("WTI") is not None:
            wti_price = round(float(instruments["WTI"]['Close'].iloc[-1]), 2)
        if instruments.get("BRENT") is not None:
            brent_price = round(float(instruments["BRENT"]['Close'].iloc[-1]), 2)
    except Exception:
        pass

    quant_score = 0.0
    quant_diagnostics = {}
    try:
        quant = compute_quant_stack(instruments)
        quant_score = quant.score
        quant_diagnostics = quant.diagnostics
    except Exception as e:
        logger.error("Quant stack error: %s", e)

    features = {
        'trend_score': trend_score,
        'vol_score': vol_score,
        'vol_regime': vol_regime,
        'rsi': rsi,
        'rsi_signal': rsi_signal,
        'atr_pct': atr_pct,
        'volume_signal': volume_signal,
        'macro_signal': macro_signal,
        'opec_days': opec_days,
        'opec_uncertainty': opec_uncertainty,
        'latest_close': latest_close,
        'wti_price': wti_price,
        'brent_price': brent_price,
        'brent_wti_spread': brent_wti_spread,
        'price_1d': price_1d,
        'price_5d': price_5d,
        'price_10d': price_10d,
        'data_rows': len(df),
        'quant_score': quant_score,
        'quant_diagnostics': quant_diagnostics,
    }

    logger.info(
        "Features: trend=%.4f | vol=%s | RSI=%.1f | ATR=%.2f%% | OPEC≈%s days",
        trend_score or 0, vol_regime or 'N/A', rsi or 0,
        atr_pct or 0, opec_days
    )

    return features


def _empty_features(macro_signal: float = 0.0) -> FeatureDict:
    return {k: None for k in [
        'trend_score', 'vol_score', 'vol_regime', 'rsi', 'rsi_signal',
        'atr_pct', 'volume_signal', 'latest_close', 'wti_price',
        'brent_price', 'brent_wti_spread', 'price_1d', 'price_5d', 'price_10d',
    ]} | {'macro_signal': macro_signal, 'opec_uncertainty': False,
          'opec_days': None, 'data_rows': 0, 'vol_score': 0.0, 'rsi_signal': 0.0,
          'quant_score': 0.0, 'quant_diagnostics': {}}


if __name__ == "__main__":
    from market_data import fetch_all_instruments, compute_macro_signal
    instruments = fetch_all_instruments()
    macro = compute_macro_signal(instruments)
    features = compute_features(instruments, macro)
    print("\nComputed Oil Features:")
    for k, v in features.items():
        print(f"  {k:25s}: {v}")
