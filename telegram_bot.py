"""
telegram_bot.py — Oil Signal Telegram Output
=============================================
Optimized message format:
- compact "trade card" at top for fast decisioning
- scorecard with major factors + quant + consensus quality
- concise risk and data-quality warnings
- keeps payload comfortably below Telegram hard limits
"""

import requests
import json
from datetime import datetime, timezone, timedelta
from math import log2
from typing import Dict, Any, List, Optional

import config
from logger import get_logger

logger = get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
UTC_PLUS_530 = timedelta(hours=5, minutes=30)   # IST (for reference)
UTC = timezone.utc


def _signal_emoji(signal: str) -> str:
    return {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '🟡'}.get(signal, '⚪')


def _confidence_bar(pct: float) -> str:
    filled = round(pct / 10)
    return '▓' * filled + '░' * (10 - filled)


def _safe_pct(value: Optional[float], scale: float = 100.0) -> float:
    if value is None:
        return 0.0
    return float(value) * scale


def _ist_time_str(r: Dict[str, Any]) -> str:
    raw_ts = r.get('timestamp')
    dt = None
    if raw_ts:
        try:
            dt = datetime.fromisoformat(str(raw_ts).replace('Z', '+00:00'))
        except ValueError:
            dt = None
    if dt is None:
        dt = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    ist = dt.astimezone(timezone(UTC_PLUS_530))
    return ist.strftime('%d %b %Y, %I:%M %p IST')


def _regime_bundle(r: Dict[str, Any]) -> Dict[str, float | str]:
    high_vol = r.get('regime_prob_high_vol')
    if high_vol is None:
        high_vol = r.get('model_risk', {}).get('regime_prob_high_vol', 0.0)
    hv = min(1.0, max(0.0, float(high_vol or 0.0)))
    diffusion = max(0.0, 1.0 - hv)
    transition = min(diffusion, hv * 0.45)
    stress = max(0.0, 1.0 - diffusion - transition)

    if hv < 0.33:
        regime = "LOW_VOL_DIFFUSION"
    elif hv < 0.66:
        regime = "TRANSITION"
    else:
        regime = "STRESS"

    return {
        'regime': regime,
        'd': diffusion,
        't': transition,
        's': stress,
    }


def _entropy_from_mix(d: float, t: float, s: float) -> float:
    probs = [max(1e-9, p) for p in (d, t, s)]
    h = -sum(p * log2(p) for p in probs)
    return h / log2(3)


def _change_arrow(change: Optional[float]) -> str:
    if change is None: return '↔'
    return '↗' if change > 0.5 else ('↘' if change < -0.5 else '↔')


def _factor_bar(score: float) -> str:
    """Visualizes a score in [-1, +1] as a directional bar."""
    normalized = (score + 1) / 2  # to [0, 1]
    filled = round(normalized * 10)
    bar = '░' * filled + '·' * (10 - filled)
    sign = '+' if score >= 0 else ''
    return f"[{bar}] {sign}{score:.2f}"


def _position_hint(conf_pct: float, vol_regime: Optional[str], quant_score: float) -> str:
    """
    Human-readable sizing guidance (not execution advice).
    """
    base = 0.25 if conf_pct < 45 else (0.50 if conf_pct < 65 else 0.75)
    if vol_regime == "HIGH":
        base *= 0.65
    elif vol_regime == "LOW":
        base *= 1.10

    if quant_score > 0.25:
        base *= 1.10
    elif quant_score < -0.25:
        base *= 0.95

    pct = int(round(max(10, min(100, base * 100))))
    return f"{pct}% of normal risk budget"


def _shorten(text: str, n: int = 72) -> str:
    if len(text) <= n:
        return text
    return text[:n - 1].rstrip() + "…"


def _format_message(r: Dict[str, Any]) -> str:
    """Formats the full signal result into a Telegram message."""
    time_str = _ist_time_str(r)

    signal = r.get('signal', 'NEUTRAL')
    conf_pct = r.get('confidence_pct', 0)
    emoji = _signal_emoji(signal)
    bar = _confidence_bar(conf_pct)

    wti = r.get('wti_price')
    brent = r.get('brent_price')
    spread = r.get('brent_wti_spread')
    p1d = r.get('price_1d')
    p5d = r.get('price_5d')
    p10d = r.get('price_10d')
    vol = r.get('vol_regime', 'N/A')
    rsi = r.get('rsi')
    atr = r.get('atr_pct')

    poly_score = r.get('polymarket_score', 0)
    sent_score = r.get('sentiment_score', 0)
    trend_score = r.get('trend_score')
    macro_score = r.get('macro_signal', 0)
    quant_score = r.get('quant_score', 0.0)

    poly_markets = r.get('polymarket_markets', [])
    n_poly = r.get('polymarket_market_count', 0)
    articles = r.get('article_count', 0)
    pos_arts = r.get('positive_articles', 0)
    neg_arts = r.get('negative_articles', 0)

    opec_days = r.get('opec_days')
    opec_flag = r.get('opec_uncertainty', False)
    reasons = r.get('reasons', [])
    dq = r.get('data_quality', {})
    participation = dq.get('factor_participation')
    consensus = dq.get('consensus_strength')

    posterior_up = _safe_pct(r.get('posterior_up', r.get('normalized_score', 0.5)))
    long_only_weight = _safe_pct(r.get('recommended_weight', r.get('kelly_fractional', 0.0)))
    if signal == 'BEARISH':
        long_only_weight = 0.0
    long_only_weight = max(0.0, min(100.0, long_only_weight))

    regime = _regime_bundle(r)
    disagreement = (1.0 - float(consensus or 0.0)) if consensus is not None else 0.0
    uncertainty = float(r.get('model_uncertainty', r.get('garch_vol', 0.0)) or 0.0)
    uncertainty = max(0.0, min(1.0, uncertainty))
    energy = max(0.0, min(2.0, 1.0 + float(quant_score or 0.0)))
    entropy = _entropy_from_mix(regime['d'], regime['t'], regime['s'])

    headline = "Bullish ✅" if sent_score > 0.08 else ("Bearish ❗" if sent_score < -0.08 else "Mixed 📰")

    if rsi is not None:
        rsi_note = " (Overbought ⚠️)" if rsi > 70 else (" (Oversold ⚠️)" if rsi < 30 else "")
        rsi_line = f"   RSI(14): {rsi:.1f}{rsi_note}"
    else:
        rsi_line = "   RSI(14): N/A"

    price_label = "NIFTY 50 DATA" if r.get('market_label') == 'NIFTY50' else "MARKET DATA"
    close_price = r.get('close_price', r.get('latest_close', wti or brent))

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "📊 OIL MARKET SIGNAL REPORT",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"🕐 {time_str}",
        "",
        f"{emoji}  Signal:  {signal}",
        f"📶  Confidence: {conf_pct:.1f}%",
        f"🎯  Posterior P(up): {posterior_up:.1f}%",
        f"     [{bar}]",
        f"⚖️  Recommended Weight: {long_only_weight:.2f}%",
        "",
        "🧠 MODEL RISK",
        f"   Regime: {regime['regime']}",
        f"   Regime Mix: D {regime['d'] * 100:.1f}% | T {regime['t'] * 100:.1f}% | S {regime['s'] * 100:.1f}%",
        f"   Disagreement: {disagreement:.3f}",
        f"   Uncertainty: {uncertainty:.3f}",
        f"   Energy / Entropy: {energy:.2f} / {entropy:.2f}",
        "",
        f"📈 {price_label}",
        f"   Close: {close_price:,.2f}" if isinstance(close_price, (int, float)) else "   Close: N/A",
        f"   1-Day: {p1d:+.2f}%  {_change_arrow(p1d)}" if p1d is not None else "   1-Day: N/A",
        f"   5-Day: {p5d:+.2f}%  {_change_arrow(p5d)}" if p5d is not None else "   5-Day: N/A",
        rsi_line,
        "",
        "📰 NEWS SENTIMENT",
        f"   Overall: {headline}",
        f"   Score: {sent_score:+.3f} (range: -1 to +1)",
        f"   Articles: {articles} analyzed",
        f"   Positive: {pos_arts} | Negative: {neg_arts}",
    ]

    if reasons:
        lines.extend(["", "🔍 KEY FACTORS"])
        for r_text in reasons[:4]:
            lines.append(f"   • {_shorten(r_text, 100)}")

    lines.extend([
        "",
        "🧭 SCENARIO PLAYBOOK",
        "   • Base: keep suggested weight while shock/uncertainty stay contained.",
        "   • Upside: trend persistence with low shock supports full risk budget.",
    ])

    missing = []
    if not dq.get('has_polymarket'):
        missing.append("Polymarket")
    if not dq.get('has_market'):
        missing.append("price data")
    if not dq.get('has_sentiment'):
        missing.append("news")
    if missing or opec_flag:
        lines.append("")
        if missing:
            lines.append(f"⚠️  Missing data: {', '.join(missing)} — reliability reduced")
        if opec_flag:
            day_count = opec_days if opec_days and opec_days >= 0 else abs(opec_days or 0)
            lines.append(f"⚠️  OPEC event window: {day_count} day(s)")

    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️  NOT financial advice.",
        "   Always do your own research.",
        "   Past signals don't guarantee future accuracy.",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ])

    return '\n'.join(lines)


def send_telegram_message(message: str, chat_id: Optional[str] = None) -> bool:
    url = TELEGRAM_API.format(token=config.TELEGRAM_TOKEN, method="sendMessage")
    payload = {
        'chat_id': (chat_id or config.TELEGRAM_CHAT_ID).strip(),
        'text': message,
        'disable_web_page_preview': True,
    }

    for attempt in range(1, config.REQUEST_RETRIES + 1):
        try:
            response = requests.post(url, data=payload, timeout=config.REQUEST_TIMEOUT)
            data = response.json()

            if response.status_code == 200 and data.get('ok'):
                logger.info("Telegram: message sent")
                return True

            err_code = data.get('error_code', 0)
            desc = data.get('description', '')

            if err_code == 401:
                logger.error("Telegram: invalid token")
                return False
            if err_code == 400 and 'chat not found' in desc.lower():
                logger.error("Telegram: invalid chat ID")
                return False
            if err_code == 429:
                import time
                wait = data.get('parameters', {}).get('retry_after', 30)
                time.sleep(min(wait, 60))
                continue

            logger.warning("Telegram error %d: %s (attempt %d)", err_code, desc, attempt)

        except requests.exceptions.Timeout:
            logger.warning("Telegram timeout (attempt %d)", attempt)
        except Exception as e:
            logger.error("Telegram unexpected error: %s", e)
            return False

        if attempt < config.REQUEST_RETRIES:
            import time
            time.sleep(config.RETRY_DELAY_BASE ** attempt)

    return False


def send_signal(signal_result: Dict[str, Any]) -> bool:
    try:
        message = _format_message(signal_result)
        chat_ids = [cid.strip() for cid in config.TELEGRAM_CHAT_ID.split(',') if cid.strip()]

        if not chat_ids:
            logger.error("No Telegram chat IDs configured")
            return False

        # Send once per unique chat ID (preserve input order).
        unique_chat_ids = list(dict.fromkeys(chat_ids))
        if len(unique_chat_ids) < len(chat_ids):
            logger.warning("Duplicate Telegram chat IDs detected; deduplicating before send")

        sent_count = 0
        failed_ids = []
        for chat_id in unique_chat_ids:
            ok = send_telegram_message(message, chat_id=chat_id)
            if ok:
                sent_count += 1
            else:
                failed_ids.append(chat_id)

        if failed_ids:
            logger.error("Telegram send incomplete: sent=%d failed=%d failed_ids=%s",
                         sent_count, len(failed_ids), failed_ids)
            return False

        logger.info("Telegram send complete: sent=%d", sent_count)
        return True
    except Exception as e:
        logger.error("Failed to format/send signal: %s", e, exc_info=True)
        return False



def send_error_alert(error_summary: str) -> bool:
    now = datetime.now(UTC).strftime('%d %b %Y %H:%M UTC')
    message = (
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️  OIL SIGNAL SYSTEM ALERT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {now}\n\n"
        f"Error: {error_summary[:500]}\n\n"
        "No signal generated. Check oil_signal.log\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    return send_telegram_message(message)


def test_connection() -> bool:
    msg = (
        "✅ Oil Signal System — Connection OK\n"
        f"Time: {datetime.now(UTC).strftime('%d %b %Y %H:%M UTC')}\n\n"
        "Bot configured. Daily oil signals will arrive here."
    )
    success = send_telegram_message(msg)
    if success:
        logger.info("Telegram test PASSED")
    else:
        logger.error("Telegram test FAILED")
    return success


if __name__ == "__main__":
    errors = config.validate_config()
    if errors:
        for e in errors:
            print(f"✗ {e}")
    else:
        print("Testing connection...")
        test_connection()
