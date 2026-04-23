"""Walk-forward backtest for the oil signal system (fully free-data)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import config
from features import compute_features
from logger import get_logger
from market_data import compute_macro_signal
from signal_engine import generate_signal

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    metrics: Dict[str, float]


def _fetch_ticker_history(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)
        if df is None or df.empty:
            return None
        df.columns = [c.capitalize() for c in df.columns]
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.dropna(subset=['Close'])
        if len(df) < 10:
            return None
        df['Returns'] = df['Close'].pct_change()
        return df.dropna(subset=['Returns'])
    except Exception as e:
        logger.warning("Backtest fetch failed for %s: %s", ticker, e)
        return None


def _fetch_backtest_universe(start: str, end: str) -> Dict[str, Optional[pd.DataFrame]]:
    mapping = {
        'WTI': config.WTI_TICKER,
        'BRENT': config.BRENT_TICKER,
        'USD': config.USD_INDEX_TICKER,
        'XLE': config.XLE_TICKER,
        'GOLD': config.GOLD_TICKER,
        'NATGAS': config.NAT_GAS_TICKER,
    }
    data = {}
    for name, ticker in mapping.items():
        logger.info("Backtest loading %s (%s)", name, ticker)
        data[name] = _fetch_ticker_history(ticker, start, end)
    return data


def _slice_instruments(universe: Dict[str, Optional[pd.DataFrame]], as_of: pd.Timestamp) -> Dict[str, Optional[pd.DataFrame]]:
    out = {}
    for k, df in universe.items():
        if df is None:
            out[k] = None
            continue
        sliced = df[df.index <= as_of].copy()
        out[k] = sliced if not sliced.empty else None
    return out


def _signal_to_position(sig: str) -> int:
    return 1 if sig == 'BULLISH' else -1 if sig == 'BEARISH' else 0


def run_backtest(start_date: str = config.BACKTEST_START_DATE, end_date: Optional[str] = config.BACKTEST_END_DATE) -> BacktestResult:
    end = end_date or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    universe = _fetch_backtest_universe(start_date, end)

    wti = universe.get('WTI')
    if wti is None or len(wti) < config.BACKTEST_WARMUP_BARS + 30:
        raise RuntimeError("Insufficient WTI history for backtest")

    dates = wti.index
    rows = []
    prev_pos = 0
    tx_cost = config.BACKTEST_TX_COST_BPS / 10_000.0

    for i in range(config.BACKTEST_WARMUP_BARS, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]

        inst = _slice_instruments(universe, d)
        macro = compute_macro_signal(inst)
        feats = compute_features(inst, macro, as_of_date=d.date())

        # Historical backtest uses only end-of-day observable factors.
        res = generate_signal(
            polymarket_score=0.0,
            polymarket_markets=[],
            sentiment_score=0.0,
            analyzed_articles=[],
            features=feats,
        )
        pos = _signal_to_position(res['signal'])

        ret_next = float((wti.loc[d_next, 'Close'] / wti.loc[d, 'Close']) - 1.0)
        turnover = abs(pos - prev_pos)
        strat_ret = pos * ret_next - (turnover * tx_cost)

        rows.append({
            'date': d.date().isoformat(),
            'next_date': d_next.date().isoformat(),
            'signal': res['signal'],
            'confidence': res['confidence'],
            'position': pos,
            'wti_ret_next_1d': ret_next,
            'strategy_ret_1d': strat_ret,
            'turnover': turnover,
            'quant_score': feats.get('quant_score', 0.0),
            'raw_score': res.get('raw_score', 0.0),
        })
        prev_pos = pos

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Backtest generated no rows")

    eq = (1 + df['strategy_ret_1d']).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1

    active = df[df['position'] != 0]
    hit = float((np.sign(active['position']) == np.sign(active['wti_ret_next_1d'])).mean()) if len(active) else 0.0

    ann_ret = float((eq.iloc[-1] ** (252 / len(df))) - 1)
    ann_vol = float(df['strategy_ret_1d'].std() * np.sqrt(252))
    sharpe = float((df['strategy_ret_1d'].mean() / (df['strategy_ret_1d'].std() + 1e-12)) * np.sqrt(252))

    metrics = {
        'rows': float(len(df)),
        'active_trade_days': float(len(active)),
        'hit_ratio': hit,
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': float(dd.min()),
        'avg_turnover': float(df['turnover'].mean()),
        'avg_confidence': float(df['confidence'].mean()),
        'avg_quant_score': float(df['quant_score'].mean()),
    }

    df.to_csv('backtest_results.csv', index=False)
    logger.info("Backtest complete: rows=%d sharpe=%.2f hit=%.1f%%", len(df), sharpe, hit * 100)

    return BacktestResult(trades=df, metrics=metrics)


def print_backtest_report(result: BacktestResult) -> None:
    m = result.metrics
    print("\n" + "=" * 60)
    print("OIL SIGNAL BACKTEST REPORT (1D NEXT-DAY)")
    print("=" * 60)
    print(f"Rows:                 {int(m['rows'])}")
    print(f"Active trade days:    {int(m['active_trade_days'])}")
    print(f"Hit ratio:            {m['hit_ratio'] * 100:.1f}%")
    print(f"Annual return:        {m['annual_return'] * 100:.2f}%")
    print(f"Annual volatility:    {m['annual_volatility'] * 100:.2f}%")
    print(f"Sharpe:               {m['sharpe']:.2f}")
    print(f"Max drawdown:         {m['max_drawdown'] * 100:.2f}%")
    print(f"Avg turnover:         {m['avg_turnover']:.2f}")
    print(f"Avg confidence:       {m['avg_confidence'] * 100:.1f}%")
    print(f"Avg quant score:      {m['avg_quant_score']:.3f}")
    print("Results CSV:          backtest_results.csv")


if __name__ == '__main__':
    result = run_backtest()
    print_backtest_report(result)
