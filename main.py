"""
main.py — Oil Signal Pipeline Orchestrator
==========================================
Five-stage pipeline:
  1. Market data (WTI, Brent, USD, XLE, Gold, NatGas)
  2. Polymarket prediction markets
  3. News sentiment (oil-filtered)
  4. Feature engineering
  5. Signal generation → Telegram + validator

Usage:
  python main.py          → full pipeline
  python main.py test     → Telegram connection test
  python main.py performance → accuracy report
"""

import sys
import traceback
from datetime import datetime, timezone

try:
    import config
    from logger import get_logger
    from news_fetcher import fetch_news, get_top_entities
    from sentiment import analyze_sentiment
    from market_data import fetch_all_instruments, compute_macro_signal, get_latest_price
    from features import compute_features
    from polymarket import fetch_polymarket_signal, get_polymarket_summary
    from signal_engine import generate_signal
    from telegram_bot import send_signal, send_error_alert, test_connection
    from validator import save_prediction, update_pending_outcomes, compute_performance_metrics, print_performance_report
except ImportError as e:
    print(f"\n[FATAL] Import failed: {e}")
    print("Install: pip install feedparser yfinance requests textblob pandas numpy")
    sys.exit(1)

logger = get_logger(__name__)
UTC = timezone.utc


def run_pipeline() -> bool:
    start = datetime.now(UTC)
    logger.info("=" * 65)
    logger.info("OIL SIGNAL SYSTEM — Run started %s",
                start.strftime('%Y-%m-%d %H:%M:%S UTC'))
    logger.info("=" * 65)

    # ── Step 0: Config ──
    errors = config.validate_config()
    if errors:
        for e in errors:
            logger.error("Config: %s", e)
        print("\n❌ Config errors in config.py:")
        for e in errors:
            print(f"  • {e}")
        return False

    # ── Step 1: Market Data ──
    logger.info("[Step 1] Fetching oil market instruments...")
    instruments = {}
    macro_signal = 0.0

    try:
        instruments = fetch_all_instruments()
        macro_signal = compute_macro_signal(instruments)
        logger.info("[Step 1] Macro signal: %.4f", macro_signal)
    except Exception as e:
        logger.error("[Step 1] Market data failed: %s", e, exc_info=True)

    # ── Step 2: Update outcomes from yesterday ──
    logger.info("[Step 2] Updating pending outcomes...")
    try:
        wti_today = get_latest_price(instruments)
        if wti_today:
            n = update_pending_outcomes(wti_today)
            logger.info("[Step 2] %d outcome(s) updated", n)
    except Exception as e:
        logger.error("[Step 2] Outcome update failed: %s", e, exc_info=True)

    # ── Step 3: Polymarket ──
    logger.info("[Step 3] Fetching Polymarket prediction markets...")
    polymarket_score = 0.0
    polymarket_markets = []

    try:
        polymarket_score, polymarket_markets = fetch_polymarket_signal()
        logger.info("[Step 3] Polymarket score=%.4f from %d markets",
                    polymarket_score, len(polymarket_markets))
    except Exception as e:
        logger.error("[Step 3] Polymarket failed: %s", e, exc_info=True)

    # ── Step 4: News ──
    logger.info("[Step 4] Fetching oil news...")
    articles = []
    try:
        articles = fetch_news()
        logger.info("[Step 4] %d oil-relevant articles", len(articles))
        if articles:
            top_entities = get_top_entities(articles)
            logger.info("[Step 4] Top entities: %s", top_entities[:3])
    except Exception as e:
        logger.error("[Step 4] News fetch failed: %s", e, exc_info=True)

    # ── Step 5: Sentiment ──
    logger.info("[Step 5] Analyzing sentiment...")
    sentiment_score = 0.0
    analyzed_articles = []
    try:
        if articles:
            sentiment_score, analyzed_articles = analyze_sentiment(articles)
            logger.info("[Step 5] Sentiment: %.4f", sentiment_score)
    except Exception as e:
        logger.error("[Step 5] Sentiment failed: %s", e, exc_info=True)

    # ── Step 6: Features ──
    logger.info("[Step 6] Computing features...")
    features = {}
    try:
        features = compute_features(instruments, macro_signal)
    except Exception as e:
        logger.error("[Step 6] Features failed: %s", e, exc_info=True)
        features = {}

    # ── Step 7: Signal ──
    logger.info("[Step 7] Generating signal...")
    signal_result = None
    try:
        signal_result = generate_signal(
            polymarket_score, polymarket_markets,
            sentiment_score, analyzed_articles,
            features
        )
        logger.info("[Step 7] SIGNAL=%s | conf=%.1f%% | poly=%.4f | sent=%.4f",
                    signal_result['signal'], signal_result['confidence_pct'],
                    polymarket_score, sentiment_score)
    except Exception as e:
        logger.error("[Step 7] Signal generation failed: %s", e, exc_info=True)
        # Emergency neutral
        from datetime import timezone
        signal_result = {
            'signal': 'NEUTRAL', 'confidence': 0.05, 'confidence_pct': 5.0,
            'raw_score': 0.0, 'normalized_score': 0.5,
            'polymarket_score': polymarket_score, 'sentiment_score': sentiment_score,
            'trend_score': None, 'macro_signal': macro_signal,
            'vol_regime': None, 'rsi': None, 'atr_pct': None,
            'wti_price': None, 'brent_price': None, 'brent_wti_spread': None,
            'price_1d': None, 'price_5d': None, 'price_10d': None,
            'article_count': len(analyzed_articles), 'positive_articles': 0, 'negative_articles': 0,
            'polymarket_market_count': len(polymarket_markets), 'polymarket_markets': [],
            'opec_days': None, 'opec_uncertainty': False,
            'reasons': ["Emergency fallback — signal engine error"],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_quality': {'has_polymarket': False, 'has_sentiment': bool(articles),
                             'has_market': bool(instruments), 'opec_uncertainty': False},
        }

    # ── Step 8: Save ──
    logger.info("[Step 8] Saving prediction...")
    try:
        save_prediction(signal_result)
    except Exception as e:
        logger.error("[Step 8] Save failed: %s", e, exc_info=True)

    # ── Step 9: Telegram ──
    logger.info("[Step 9] Sending to Telegram...")
    success = False
    try:
        success = send_signal(signal_result)
        logger.info("[Step 9] Telegram: %s", '✓' if success else '✗')
    except Exception as e:
        logger.error("[Step 9] Telegram failed: %s", e, exc_info=True)

    elapsed = (datetime.now(UTC) - start).total_seconds()
    logger.info("Pipeline complete in %.1fs | %s %.1f%% | Telegram=%s",
                elapsed, signal_result['signal'],
                signal_result['confidence_pct'], '✓' if success else '✗')

    return success


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'run'

    if mode == 'test':
        errors = config.validate_config()
        if errors:
            for e in errors:
                print(f"✗ {e}")
        else:
            test_connection()

    elif mode == 'performance':
        print_performance_report()

    elif mode == 'run':
        try:
            success = run_pipeline()
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            logger.info("Interrupted")
            sys.exit(0)
        except Exception as e:
            logger.error("CRITICAL: %s", traceback.format_exc())
            try:
                send_error_alert(str(e))
            except Exception:
                pass
            sys.exit(1)

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [run|test|performance]")
        sys.exit(1)
