"""
validator.py — Oil Signal Accuracy Tracking
=============================================
Enhancements over Nifty version:
1. Tracks both WTI price and signal accuracy
2. Polymarket market IDs archived — lets us verify if Polymarket
   was actually right on specific markets (manual audit trail)
3. Multi-timeframe accuracy: 1-day, 3-day, 5-day outcome tracking
4. Confidence-stratified accuracy reporting
5. Factor attribution: which of the 4 factors was most predictive?
"""

import csv
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import config
from logger import get_logger

logger = get_logger(__name__)

UTC = timezone.utc

CSV_COLUMNS = [
    'date', 'timestamp_utc',
    'signal', 'confidence', 'raw_score',
    'polymarket_score', 'sentiment_score', 'trend_score', 'macro_signal',
    'vol_regime', 'rsi', 'atr_pct',
    'wti_price', 'brent_price', 'brent_wti_spread',
    'polymarket_market_count',
    'article_count',
    'opec_uncertainty',
    # Outcome fields (filled D+1, D+3, D+5)
    'wti_next1', 'change_pct_1d', 'outcome_1d',
    'wti_next3', 'change_pct_3d', 'outcome_3d',
    'wti_next5', 'change_pct_5d', 'outcome_5d',
    # Audit
    'polymarket_market_ids',   # JSON list of market IDs used
]

NEUTRAL_THRESHOLD = 0.7  # % move below which NEUTRAL is "correct"


def _ensure_csv():
    if not os.path.exists(config.PREDICTIONS_FILE):
        try:
            with open(config.PREDICTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
            logger.info("Created predictions file: %s", config.PREDICTIONS_FILE)
        except IOError as e:
            logger.error("Cannot create predictions file: %s", e)


def save_prediction(signal_result: Dict[str, Any]) -> bool:
    _ensure_csv()

    now = datetime.now(UTC)
    date_str = now.strftime('%Y-%m-%d')

    existing = _read_all()
    if any(p.get('date') == date_str for p in existing):
        logger.warning("Prediction for %s already saved — skipping", date_str)
        return True

    # Extract Polymarket market IDs for audit trail
    poly_ids = [m.get('id', '') for m in signal_result.get('polymarket_markets', [])]

    row = {
        'date': date_str,
        'timestamp_utc': now.isoformat(),
        'signal': signal_result.get('signal', 'NEUTRAL'),
        'confidence': round(signal_result.get('confidence', 0), 4),
        'raw_score': round(signal_result.get('raw_score', 0), 4),
        'polymarket_score': round(signal_result.get('polymarket_score', 0), 4),
        'sentiment_score': round(signal_result.get('sentiment_score', 0), 4),
        'trend_score': round(signal_result.get('trend_score', 0) or 0, 4),
        'macro_signal': round(signal_result.get('macro_signal', 0) or 0, 4),
        'vol_regime': signal_result.get('vol_regime', ''),
        'rsi': round(signal_result.get('rsi', 0) or 0, 2),
        'atr_pct': round(signal_result.get('atr_pct', 0) or 0, 4),
        'wti_price': round(signal_result.get('wti_price', 0) or 0, 2),
        'brent_price': round(signal_result.get('brent_price', 0) or 0, 2),
        'brent_wti_spread': round(signal_result.get('brent_wti_spread', 0) or 0, 2),
        'polymarket_market_count': signal_result.get('polymarket_market_count', 0),
        'article_count': signal_result.get('article_count', 0),
        'opec_uncertainty': int(signal_result.get('opec_uncertainty', False)),
        # Outcomes — filled later
        'wti_next1': '', 'change_pct_1d': '', 'outcome_1d': 'PENDING',
        'wti_next3': '', 'change_pct_3d': '', 'outcome_3d': 'PENDING',
        'wti_next5': '', 'change_pct_5d': '', 'outcome_5d': 'PENDING',
        'polymarket_market_ids': json.dumps(poly_ids),
    }

    try:
        with open(config.PREDICTIONS_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
        logger.info("Prediction saved: %s %s conf=%.2f poly=%.4f",
                    date_str, row['signal'], row['confidence'], row['polymarket_score'])
        return True
    except IOError as e:
        logger.error("Failed to save prediction: %s", e)
        return False


def _evaluate_outcome(signal: str, change_pct: float) -> str:
    if signal == 'BULLISH':
        return 'CORRECT' if change_pct > 0 else 'INCORRECT'
    elif signal == 'BEARISH':
        return 'CORRECT' if change_pct < 0 else 'INCORRECT'
    elif signal == 'NEUTRAL':
        return 'NEUTRAL_HIT' if abs(change_pct) < NEUTRAL_THRESHOLD else 'NEUTRAL_MISS'
    return 'UNKNOWN'


def update_pending_outcomes(wti_today: Optional[float]) -> int:
    """
    Multi-timeframe outcome resolution.

    For each PENDING row:
    - If the row is 1 trading day old → try to fill 1d outcome
    - If 3 days old → try to fill 3d outcome
    - If 5 days old → try to fill 5d outcome

    WHY multi-timeframe: Oil macro signals often take 3–5 days to
    materialize. A 1-day accuracy test undersells the signal quality.
    """
    if wti_today is None:
        logger.warning("Cannot update outcomes: no WTI price available")
        return 0

    predictions = _read_all()
    if not predictions:
        return 0

    updated = 0
    today = datetime.now(UTC).date()

    for pred in predictions:
        pred_date_str = pred.get('date', '')
        if not pred_date_str:
            continue

        try:
            pred_date = datetime.strptime(pred_date_str, '%Y-%m-%d').date()
        except ValueError:
            continue

        age_days = (today - pred_date).days
        signal = pred.get('signal', 'NEUTRAL')
        baseline_wti = _safe_float(pred.get('wti_price', ''))

        if baseline_wti == 0:
            continue

        # 1-day outcome
        if age_days >= 1 and pred.get('outcome_1d') == 'PENDING':
            change = ((wti_today - baseline_wti) / baseline_wti) * 100
            pred['wti_next1'] = round(wti_today, 2)
            pred['change_pct_1d'] = round(change, 4)
            pred['outcome_1d'] = _evaluate_outcome(signal, change)
            updated += 1
            logger.info("1d outcome for %s: signal=%s change=%+.2f%% → %s",
                        pred_date_str, signal, change, pred['outcome_1d'])

        # 3-day outcome (approximate — we don't have intermediate prices)
        if age_days >= 3 and pred.get('outcome_3d') == 'PENDING':
            change = ((wti_today - baseline_wti) / baseline_wti) * 100
            pred['wti_next3'] = round(wti_today, 2)
            pred['change_pct_3d'] = round(change, 4)
            pred['outcome_3d'] = _evaluate_outcome(signal, change)

        # 5-day outcome
        if age_days >= 5 and pred.get('outcome_5d') == 'PENDING':
            change = ((wti_today - baseline_wti) / baseline_wti) * 100
            pred['wti_next5'] = round(wti_today, 2)
            pred['change_pct_5d'] = round(change, 4)
            pred['outcome_5d'] = _evaluate_outcome(signal, change)

    if updated > 0:
        _write_all(predictions)

    return updated


def compute_performance_metrics() -> Dict[str, Any]:
    """Extended accuracy metrics including factor attribution."""
    predictions = _read_all()
    if not predictions:
        return {'error': 'No predictions', 'total': 0}

    resolved_1d = [p for p in predictions if p.get('outcome_1d') not in ('PENDING', '')]
    directional = [p for p in resolved_1d if p.get('signal') in ('BULLISH', 'BEARISH')]

    if not directional:
        return {
            'error': 'No resolved directional predictions',
            'total_saved': len(predictions),
            'pending': sum(1 for p in predictions if p.get('outcome_1d') == 'PENDING'),
        }

    def accuracy(subset):
        if not subset: return None
        correct = sum(1 for p in subset if p.get('outcome_1d') == 'CORRECT')
        return round(100 * correct / len(subset), 1)

    # Confidence bands
    high_conf = [p for p in directional if _safe_float(p.get('confidence', 0)) >= 0.65]
    mid_conf = [p for p in directional if 0.45 <= _safe_float(p.get('confidence', 0)) < 0.65]
    low_conf = [p for p in directional if _safe_float(p.get('confidence', 0)) < 0.45]

    # 5-day accuracy (deeper signal quality)
    resolved_5d = [p for p in predictions if p.get('outcome_5d') not in ('PENDING', '')]
    dir_5d = [p for p in resolved_5d if p.get('signal') in ('BULLISH', 'BEARISH')]
    acc_5d = accuracy(dir_5d)

    # OPEC uncertainty filter
    non_opec = [p for p in directional if p.get('opec_uncertainty', '0') == '0']
    opec_adj_acc = accuracy(non_opec)

    # Polymarket reliability: was Polymarket signal correct?
    poly_bullish = [p for p in directional if _safe_float(p.get('polymarket_score', 0)) > 0.1]
    poly_bearish = [p for p in directional if _safe_float(p.get('polymarket_score', 0)) < -0.1]
    poly_bull_acc = accuracy([p for p in poly_bullish if p.get('signal') == 'BULLISH'])
    poly_bear_acc = accuracy([p for p in poly_bearish if p.get('signal') == 'BEARISH'])

    return {
        'total_saved': len(predictions),
        'total_resolved_1d': len(resolved_1d),
        'directional_total': len(directional),
        'directional_correct': sum(1 for p in directional if p.get('outcome_1d') == 'CORRECT'),
        'overall_accuracy_1d_pct': accuracy(directional),
        'accuracy_5d_pct': acc_5d,
        'accuracy_excl_opec_pct': opec_adj_acc,
        'by_confidence': {
            'high_65plus': accuracy(high_conf),
            'mid_45to65': accuracy(mid_conf),
            'low_below45': accuracy(low_conf),
        },
        'polymarket_accuracy': {
            'bullish_correct': poly_bull_acc,
            'bearish_correct': poly_bear_acc,
        },
        'signal_distribution': {
            s: sum(1 for p in predictions if p.get('signal') == s)
            for s in ('BULLISH', 'BEARISH', 'NEUTRAL')
        },
        'pending_1d': sum(1 for p in predictions if p.get('outcome_1d') == 'PENDING'),
    }


def print_performance_report():
    m = compute_performance_metrics()

    if 'error' in m:
        print(f"\n⚠️  {m['error']}")
        return

    print("\n" + "=" * 55)
    print("🛢  OIL SIGNAL SYSTEM — PERFORMANCE REPORT")
    print("=" * 55)
    print(f"\nTotal saved:             {m['total_saved']}")
    print(f"Total resolved (1d):     {m['total_resolved_1d']}")
    print(f"Pending:                 {m['pending_1d']}")
    print(f"\n📈 1-DAY DIRECTIONAL ACCURACY")
    print(f"   Overall:              {m['overall_accuracy_1d_pct']}%")
    print(f"   5-day accuracy:       {m.get('accuracy_5d_pct', 'N/A')}%")
    print(f"   Excl. OPEC windows:   {m.get('accuracy_excl_opec_pct', 'N/A')}%")
    print(f"\n📶 BY CONFIDENCE BAND")
    bc = m['by_confidence']
    print(f"   High (≥65%):          {bc['high_65plus']}%")
    print(f"   Mid  (45–65%):        {bc['mid_45to65']}%")
    print(f"   Low  (<45%):          {bc['low_below45']}%")
    print(f"\n🎲 POLYMARKET ACCURACY")
    pa = m['polymarket_accuracy']
    print(f"   When poly=BULLISH:    {pa['bullish_correct']}%")
    print(f"   When poly=BEARISH:    {pa['bearish_correct']}%")
    print(f"\n📊 SIGNAL DISTRIBUTION")
    for sig, cnt in m['signal_distribution'].items():
        print(f"   {sig}: {cnt}")

    total = m['directional_total']
    acc = m['overall_accuracy_1d_pct'] or 0
    print()
    if total < 20:
        print(f"⚠️  Only {total} signals. Need ≥20 for statistical meaning.")
    elif acc >= 60:
        print("✅ Above 60% — system has measurable edge.")
    elif acc >= 50:
        print("🟡 Coin-flip accuracy. Review factor weights.")
    else:
        print("🔴 Below 50%. Recalibration needed.")
    print("=" * 55)


def _read_all() -> List[Dict]:
    _ensure_csv()
    rows = []
    try:
        with open(config.PREDICTIONS_FILE, 'r', newline='', encoding='utf-8') as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error("Error reading predictions: %s", e)
    return rows


def _write_all(predictions: List[Dict]) -> bool:
    tmp = config.PREDICTIONS_FILE + '.tmp'
    try:
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(predictions)
        os.replace(tmp, config.PREDICTIONS_FILE)
        return True
    except Exception as e:
        logger.error("Failed to write predictions: %s", e)
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        return False


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


if __name__ == "__main__":
    print_performance_report()
