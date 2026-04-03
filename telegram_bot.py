"""
telegram_bot.py — Oil Signal Telegram Output
=============================================
Enhanced message format:
- Four-factor score breakdown (unique to oil system)
- Polymarket top markets displayed directly in message
- Brent/WTI price comparison
- ATR-based risk context
- OPEC meeting warnings
"""

import requests
import json
from datetime import datetime, timezone, timedelta
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


def _format_message(r: Dict[str, Any]) -> str:
    """Formats the full signal result into a Telegram message."""
    now_utc = datetime.now(UTC)
    time_str = now_utc.strftime('%d %b %Y %H:%M UTC')

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

    poly_markets = r.get('polymarket_markets', [])
    n_poly = r.get('polymarket_market_count', 0)
    articles = r.get('article_count', 0)
    pos_arts = r.get('positive_articles', 0)
    neg_arts = r.get('negative_articles', 0)

    opec_days = r.get('opec_days')
    opec_flag = r.get('opec_uncertainty', False)
    reasons = r.get('reasons', [])
    dq = r.get('data_quality', {})

    lines = []

    # ── Header ──
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("🛢  OIL MARKET SIGNAL")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🕐 {time_str}")
    lines.append("")

    # ── Main Signal ──
    lines.append(f"{emoji}  Signal:  {signal}")
    lines.append(f"📶  Confidence: {conf_pct:.1f}%")
    lines.append(f"    [{bar}]")
    if opec_flag:
        lines.append(f"⚠️  OPEC meeting in {opec_days if opec_days and opec_days >= 0 else abs(opec_days or 0)} days — uncertainty elevated")
    lines.append("")

    # ── Four-Factor Breakdown ──
    lines.append("📊 FACTOR SCORES  (−1 bearish ↔ +1 bullish)")
    lines.append(f"  🎲 Polymarket: {_factor_bar(poly_score)}")
    lines.append(f"  📰 News Sent:  {_factor_bar(sent_score)}")
    if trend_score is not None:
        lines.append(f"  📈 Tech Trend: {_factor_bar(trend_score)}")
    lines.append(f"  🌐 Macro:      {_factor_bar(macro_score)}")
    lines.append("")

    # ── Price Data ──
    if wti or brent:
        lines.append("💲 CRUDE PRICES")
        if wti:
            sign = f"{_change_arrow(p1d)}"
            lines.append(f"  WTI:   ${wti:,.2f}  {sign}")
        if brent:
            lines.append(f"  Brent: ${brent:,.2f}")
        if spread is not None:
            lines.append(f"  Brent-WTI spread: ${spread:.2f}")
        if p1d is not None:
            lines.append(f"  1-day change:  {p1d:+.2f}%")
        if p5d is not None:
            lines.append(f"  5-day change:  {p5d:+.2f}%  {_change_arrow(p5d)}")
        if p10d is not None:
            lines.append(f"  10-day change: {p10d:+.2f}%")
        lines.append("")

    # ── Technical Context ──
    if vol != 'N/A' or rsi or atr:
        lines.append("⚙️ TECHNICALS")
        vol_map = {'HIGH': '⚡ High', 'NORMAL': '〜 Normal', 'LOW': '🧘 Low'}
        if vol and vol != 'N/A':
            lines.append(f"  Volatility: {vol_map.get(vol, vol)}")
        if atr:
            lines.append(f"  ATR: {atr:.2f}% of price")
        if rsi:
            note = " (Overbought ⚠️)" if rsi > 70 else (" (Oversold ⚠️)" if rsi < 30 else "")
            lines.append(f"  RSI(14): {rsi:.1f}{note}")
        lines.append("")

    # ── Polymarket Markets ──
    if poly_markets:
        lines.append(f"🎲 PREDICTION MARKETS ({n_poly} active)")
        for m in poly_markets[:3]:
            d_emoji = "🟢" if m['direction'] == 'BULLISH' else "🔴"
            lines.append(
                f"  {d_emoji} YES={m['yes_prob']:.0%} | ${m['volume_usd']:,.0f} | "
                f"{m['title'][:45]}"
            )
        lines.append("")
    elif not dq.get('has_polymarket'):
        lines.append("🎲 Prediction markets: no oil markets active")
        lines.append("")

    # ── News Sentiment ──
    if dq.get('has_sentiment'):
        lines.append(f"📰 NEWS ({articles} articles)")
        lines.append(f"  Positive: {pos_arts}  |  Negative: {neg_arts}")
        lines.append(f"  Net sentiment: {sent_score:+.3f}")
        lines.append("")

    # ── Reasoning ──
    if reasons:
        lines.append("🔍 KEY FACTORS")
        for r_text in reasons[:6]:
            lines.append(f"  • {r_text}")
        lines.append("")

    # ── Data Quality ──
    missing = []
    if not dq.get('has_polymarket'): missing.append("Polymarket")
    if not dq.get('has_market'): missing.append("price data")
    if not dq.get('has_sentiment'): missing.append("news")
    if missing:
        lines.append(f"⚠️  Missing data: {', '.join(missing)} — reliability reduced")
        lines.append("")

    # ── Disclaimer ──
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("⚠️  NOT trading advice.")
    lines.append("    Oil is extremely volatile. Risk accordingly.")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")

    return '\n'.join(lines)


def send_telegram_message(message: str) -> bool:
    url = TELEGRAM_API.format(token=config.TELEGRAM_TOKEN, method="sendMessage")
    payload = {
        'chat_id': config.TELEGRAM_CHAT_ID,
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
        # This splits your secret by the comma and sends to each person
        chat_ids = config.TELEGRAM_CHAT_ID.split(',')
        success = True
        for chat_id in chat_ids:
            # We temporarily swap the ID to send to each friend
            url = TELEGRAM_API.format(token=config.TELEGRAM_TOKEN, method="sendMessage")
            payload = {
                'chat_id': chat_id.strip(),
                'text': message,
                'disable_web_page_preview': True,
            }
            res = requests.post(url, data=payload, timeout=config.REQUEST_TIMEOUT)
            if not res.json().get('ok'):
                success = False
                logger.error(f"Failed to send to {chat_id}: {res.text}")
        return success
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
