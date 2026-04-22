"""
signal_engine.py — Oil Signal Generation Engine
=================================================
Four-factor model:
    1. Polymarket (0.35) — forward-looking crowd probabilities
    2. News Sentiment (0.25) — current news flow direction
    3. Technical Trend (0.25) — price momentum and structure
    4. Macro Context (0.15) — USD, energy equities, correlations

Why this weight ordering:
- Polymarket leads because it's forward-looking and has real money behind it
- News and technical are roughly co-equal; both are lagging but informative
- Macro is a multiplier/modifier — it rarely drives oil but confirms/denies

OPEC uncertainty adjustment:
    When within 7 days of an OPEC meeting, confidence is automatically
    reduced by 30%. The outcome is binary and unpredictable; any model
    claiming high confidence before OPEC is lying.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import math

import config
from logger import get_logger

logger = get_logger(__name__)

SignalResult = Dict[str, Any]


def _safe(val: Optional[float], default: float = 0.0) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


def _compute_raw_score(
    polymarket_score: float,
    sentiment_score: float,
    features: Dict[str, Any],
) -> tuple[float, Dict[str, float], int]:
    """
    Computes the weighted composite score from all four factors.

    Trend sub-model: 55% pure trend + 25% RSI signal + 20% volume confirmation
    Macro: direct from market_data.compute_macro_signal()
    """
    trend = _safe(features.get('trend_score'))
    rsi_signal = _safe(features.get('rsi_signal'))
    volume_signal = _safe(features.get('volume_signal'))
    macro = _safe(features.get('macro_signal'))
    quant = _safe(features.get('quant_score'))

    # Technical sub-score
    technical = (
        0.55 * trend +
        0.25 * rsi_signal +
        0.20 * (volume_signal if volume_signal is not None else 0.0)
    )
    # Modulate by volatility regime
    # HIGH vol = uncertainty, dampen signal. LOW vol = trend more reliable.
    vol_modifier = {"HIGH": 0.7, "NORMAL": 1.0, "LOW": 1.1}.get(
        features.get('vol_regime'), 1.0
    )
    technical *= vol_modifier

    factor_scores = {
        'polymarket': polymarket_score,
        'sentiment': sentiment_score,
        'technical': technical,
        'macro': macro,
        'quant': quant,
    }
    active_weights = {
        'polymarket': config.WEIGHT_POLYMARKET if abs(polymarket_score) > 0.0001 else 0.0,
        'sentiment': config.WEIGHT_SENTIMENT if abs(sentiment_score) > 0.0001 else 0.0,
        'technical': config.WEIGHT_TREND if features.get('trend_score') is not None else 0.0,
        'macro': config.WEIGHT_MACRO if features.get('macro_signal') is not None else 0.0,
        'quant': config.WEIGHT_QUANT if features.get('quant_score') is not None else 0.0,
    }
    active_weight_sum = sum(active_weights.values())
    if active_weight_sum == 0:
        raw = 0.0
        participation = 0
    else:
        normalized_weights = {k: v / active_weight_sum for k, v in active_weights.items()}
        raw = sum(factor_scores[k] * normalized_weights[k] for k in factor_scores)
        participation = sum(1 for w in active_weights.values() if w > 0)

    # Consensus overlay: reward broad agreement, punish conflict.
    active_scores = [v for k, v in factor_scores.items() if active_weights.get(k, 0) > 0 and abs(v) >= 0.12]
    bulls = sum(1 for v in active_scores if v > 0)
    bears = sum(1 for v in active_scores if v < 0)
    if max(bulls, bears) >= 3:
        raw *= config.CONSENSUS_BOOST_STRONG
    elif bulls > 0 and bears > 0:
        raw *= config.CONSENSUS_DAMPEN_MIXED

    logger.debug(
        "Raw score: poly=%.4f sent=%.4f tech=%.4f macro=%.4f quant=%.4f | participation=%d => %.4f",
        polymarket_score,
        sentiment_score,
        technical,
        macro,
        quant,
        participation,
        raw
    )

    return round(max(-1.0, min(1.0, raw)), 4), factor_scores, participation


def _classify(normalized: float) -> str:
    if normalized >= config.BULLISH_THRESHOLD:
        return "BULLISH"
    elif normalized <= config.BEARISH_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def _compute_confidence(
    raw_score: float,
    normalized: float,
    has_polymarket: bool,
    has_sentiment: bool,
    has_market: bool,
    opec_uncertainty: bool,
    consensus_strength: float,
    participation: int,
) -> float:
    """
    Confidence with oil-specific penalty structure.

    Penalties:
    - No Polymarket data: -25% (our most valuable source missing)
    - No news sentiment: -15%
    - No market price: -30%
    - OPEC meeting within 7 days: -30% (binary outcome risk)
    """
    base = abs(normalized - 0.5) * 2  # Distance from neutral

    penalty = 1.0
    if not has_polymarket:
        penalty *= 0.75
        logger.warning("Confidence -25%: no Polymarket data")
    if not has_sentiment:
        penalty *= 0.85
    if not has_market:
        penalty *= 0.70
    if opec_uncertainty:
        penalty *= 0.70
        logger.warning("Confidence -30%: OPEC meeting within %d days",
                       config.OPEC_MEETING_UNCERTAINTY_DAYS)

    participation_penalty = 1.0
    if participation < config.MIN_FACTOR_PARTICIPATION:
        participation_penalty = 0.65
    elif participation == 2:
        participation_penalty = 0.82

    consensus_boost = 0.85 + (0.30 * max(0.0, min(1.0, consensus_strength)))
    conf = max(0.05, min(0.90, base * penalty * participation_penalty * consensus_boost))
    return round(conf, 4)


def _consensus_strength(factor_scores: Dict[str, float]) -> float:
    active = [v for v in factor_scores.values() if abs(v) >= 0.12]
    if len(active) < 2:
        return 0.0
    same_sign_pairs = 0
    total_pairs = 0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            total_pairs += 1
            if active[i] * active[j] > 0:
                same_sign_pairs += 1
    return round(same_sign_pairs / total_pairs if total_pairs else 0.0, 4)


def _build_reasoning(
    signal: str,
    polymarket_score: float,
    sentiment_score: float,
    features: Dict,
    polymarket_markets: List[Dict],
) -> List[str]:
    reasons = []

    # Polymarket reasoning
    n_markets = len(polymarket_markets)
    if n_markets == 0:
        reasons.append("Prediction markets: no data available")
    else:
        bullish_m = sum(1 for m in polymarket_markets if m['direction'] == 'BULLISH')
        bearish_m = n_markets - bullish_m
        direction = "lean bullish" if polymarket_score > 0.1 else \
                    "lean bearish" if polymarket_score < -0.1 else "are balanced"
        reasons.append(
            f"Prediction markets ({n_markets} active: {bullish_m} bullish/"
            f"{bearish_m} bearish) {direction} on oil (score {polymarket_score:+.2f})"
        )
        # Show top market
        top = sorted(polymarket_markets, key=lambda m: m['volume_usd'], reverse=True)
        if top:
            reasons.append(
                f"Highest-liquidity market: \"{top[0]['title'][:55]}...\" "
                f"YES={top[0]['yes_prob']:.0%}"
            )

    # Sentiment reasoning
    if abs(sentiment_score) < 0.05:
        reasons.append("News sentiment is neutral/mixed for oil")
    elif sentiment_score > 0:
        s = "strongly" if sentiment_score > 0.3 else "mildly"
        reasons.append(f"Oil news flow is {s} bullish (supply constraints / demand strength)")
    else:
        s = "strongly" if sentiment_score < -0.3 else "mildly"
        reasons.append(f"Oil news flow is {s} bearish (supply surplus / demand weakness)")

    # Technical reasoning
    trend = features.get('trend_score')
    p5d = features.get('price_5d')
    if trend is not None:
        if trend > 0.2:
            reasons.append(f"10-day price trend is upward ({p5d:+.1f}% 5-day)" if p5d else "10-day trend is upward")
        elif trend < -0.2:
            reasons.append(f"10-day price trend is downward ({p5d:+.1f}% 5-day)" if p5d else "10-day trend is downward")
        else:
            reasons.append("Price trend is largely flat (no directional conviction)")

    # Macro reasoning
    macro = features.get('macro_signal', 0)
    if macro > 0.2:
        reasons.append("Macro context bullish: USD weakening and/or energy equities rising")
    elif macro < -0.2:
        reasons.append("Macro context bearish: USD strengthening and/or energy equities falling")

    # Spread
    spread = features.get('brent_wti_spread')
    if spread is not None:
        if spread > 8:
            reasons.append(f"Brent-WTI spread is wide (${spread:.1f}) — Atlantic basin supply tight")
        elif spread < 2:
            reasons.append(f"Brent-WTI spread is narrow (${spread:.1f}) — US-global parity, no regional stress")

    # OPEC warning
    opec_days = features.get('opec_days')
    if opec_days is not None and abs(opec_days) <= config.OPEC_MEETING_UNCERTAINTY_DAYS:
        if opec_days >= 0:
            reasons.append(f"⚠️ OPEC+ meeting in {opec_days} day(s) — binary outcome risk, confidence reduced")
        else:
            reasons.append(f"⚠️ OPEC+ met {abs(opec_days)} day(s) ago — market still repricing")

    # Volatility
    vol = features.get('vol_regime')
    if vol == "HIGH":
        reasons.append("Oil volatility is elevated — signal reliability reduced")
    elif vol == "LOW":
        reasons.append("Volatility is low — price action in steady trending regime")

    quant_score = features.get('quant_score')
    if isinstance(quant_score, (int, float)):
        if quant_score > 0.2:
            reasons.append("Quant stack is bullish across carry/regime/risk/ML layers")
        elif quant_score < -0.2:
            reasons.append("Quant stack is bearish across carry/regime/risk/ML layers")

    consensus = features.get('consensus_strength')
    if isinstance(consensus, (int, float)):
        if consensus >= 0.66:
            reasons.append("Model factors are strongly aligned (high cross-factor consensus)")
        elif consensus <= 0.34:
            reasons.append("Model factors disagree materially (mixed regime, lower conviction)")

    return reasons


def generate_signal(
    polymarket_score: float,
    polymarket_markets: List[Dict],
    sentiment_score: float,
    analyzed_articles: List[Dict],
    features: Dict[str, Any],
) -> SignalResult:
    """
    Main entry point. Generates final oil market signal.

    Args:
        polymarket_score: From polymarket.fetch_polymarket_signal()
        polymarket_markets: Market list from same function
        sentiment_score: From sentiment.analyze_sentiment()
        analyzed_articles: Analyzed article list
        features: From features.compute_features()

    Returns:
        Full SignalResult dict for Telegram and validator.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    has_polymarket = len(polymarket_markets) > 0
    has_sentiment = len(analyzed_articles) > 0
    has_market = features.get('data_rows', 0) > 0
    opec_uncertainty = features.get('opec_uncertainty', False)

    try:
        raw_score, factor_scores, participation = _compute_raw_score(polymarket_score, sentiment_score, features)
        if abs(raw_score) < config.MIN_DIRECTIONAL_SCORE:
            raw_score = 0.0
        normalized = (raw_score + 1) / 2
        signal = _classify(normalized)
        consensus_strength = _consensus_strength(factor_scores)
        features['consensus_strength'] = consensus_strength
        confidence = _compute_confidence(
            raw_score, normalized,
            has_polymarket, has_sentiment, has_market, opec_uncertainty,
            consensus_strength, participation
        )
        reasons = _build_reasoning(
            signal, polymarket_score, sentiment_score, features, polymarket_markets
        )

        pos_arts = sum(1 for a in analyzed_articles if a.get('sentiment_label') == 'POSITIVE')
        neg_arts = sum(1 for a in analyzed_articles if a.get('sentiment_label') == 'NEGATIVE')

        logger.info("OIL SIGNAL: %s | confidence=%.4f | raw=%.4f | "
                    "poly=%.4f | sent=%.4f | market=%s | OPEC=%s",
                    signal, confidence, raw_score,
                    polymarket_score, sentiment_score,
                    "✓" if has_market else "✗",
                    "⚠️" if opec_uncertainty else "OK")

        return {
            'signal': signal,
            'confidence': confidence,
            'confidence_pct': round(confidence * 100, 1),
            'raw_score': raw_score,
            'normalized_score': round(normalized, 4),

            # Factor scores
            'polymarket_score': round(polymarket_score, 4),
            'sentiment_score': round(sentiment_score, 4),
            'trend_score': features.get('trend_score'),
            'macro_signal': features.get('macro_signal'),
            'quant_score': features.get('quant_score'),

            # Market context
            'vol_regime': features.get('vol_regime'),
            'rsi': features.get('rsi'),
            'atr_pct': features.get('atr_pct'),
            'wti_price': features.get('wti_price'),
            'brent_price': features.get('brent_price'),
            'brent_wti_spread': features.get('brent_wti_spread'),
            'price_1d': features.get('price_1d'),
            'price_5d': features.get('price_5d'),
            'price_10d': features.get('price_10d'),

            # Article stats
            'article_count': len(analyzed_articles),
            'positive_articles': pos_arts,
            'negative_articles': neg_arts,

            # Polymarket
            'polymarket_market_count': len(polymarket_markets),
            'polymarket_markets': polymarket_markets[:5],  # Top 5 for Telegram

            # OPEC
            'opec_days': features.get('opec_days'),
            'opec_uncertainty': opec_uncertainty,

            'reasons': reasons,
            'timestamp': timestamp,
            'data_quality': {
                'has_polymarket': has_polymarket,
                'has_sentiment': has_sentiment,
                'has_market': has_market,
                'opec_uncertainty': opec_uncertainty,
                'factor_participation': participation,
                'consensus_strength': consensus_strength,
                'quant_score': features.get('quant_score'),
            }
        }

    except Exception as e:
        logger.error("Signal engine failed: %s", e, exc_info=True)
        return {
            'signal': 'NEUTRAL', 'confidence': 0.05, 'confidence_pct': 5.0,
            'raw_score': 0.0, 'normalized_score': 0.5,
            'polymarket_score': polymarket_score, 'sentiment_score': sentiment_score,
            'trend_score': None, 'macro_signal': None, 'quant_score': 0.0,
            'vol_regime': None, 'rsi': None, 'atr_pct': None,
            'wti_price': None, 'brent_price': None, 'brent_wti_spread': None,
            'price_1d': None, 'price_5d': None, 'price_10d': None,
            'article_count': 0, 'positive_articles': 0, 'negative_articles': 0,
            'polymarket_market_count': 0, 'polymarket_markets': [],
            'opec_days': None, 'opec_uncertainty': False,
            'reasons': [f"Signal engine error: {str(e)[:100]}"],
            'timestamp': timestamp,
            'data_quality': {'has_polymarket': False, 'has_sentiment': False,
                             'has_market': False, 'opec_uncertainty': False},
        }
