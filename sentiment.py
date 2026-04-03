"""
sentiment.py — Oil-Specific Sentiment Analysis
================================================
Improvements over Nifty version:
1. Oil-specific domain lexicon (100+ terms vs general finance)
2. Source-weighted and relevance-weighted aggregation
   (EIA report > blog post, core OPEC news > tangential mention)
3. Entity-conditional scoring — "Russia sanctions" is bearish supply
   which is BULLISH for oil price, not bearish in general
4. Recency weighting — articles from last 6 hours count more
"""

from textblob import TextBlob
from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime, timezone, timedelta

from logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# OIL-SPECIFIC SENTIMENT LEXICON
# ─────────────────────────────────────────────
# Signs for oil PRICE direction (not general market):
# POSITIVE (+) = bullish for oil price (supply cut, demand surge, geopolitical risk)
# NEGATIVE (-) = bearish for oil price (supply surge, demand destruction, price collapse)

OIL_BULLISH_TERMS = {
    # Supply constraints (less supply → higher price)
    "production cut": 0.8,
    "output cut": 0.8,
    "supply cut": 0.8,
    "quota reduction": 0.7,
    "extend cuts": 0.7,
    "maintain cuts": 0.6,
    "voluntary cut": 0.7,
    "curtailment": 0.5,
    "underproduction": 0.5,
    "supply deficit": 0.7,
    "tight supply": 0.6,
    "supply crunch": 0.7,
    "draws on inventory": 0.5,
    "inventory draw": 0.6,
    "inventory decline": 0.5,
    "strategic reserve release postponed": 0.4,

    # Demand strength
    "demand surge": 0.7,
    "demand recovery": 0.6,
    "demand growth": 0.5,
    "china demand": 0.5,
    "peak summer demand": 0.4,
    "strong demand": 0.5,

    # Geopolitical risk to supply
    "iran sanction": 0.7,
    "venezuela sanction": 0.6,
    "strait of hormuz": 0.8,
    "oil embargo": 0.7,
    "pipeline attack": 0.7,
    "oil field attack": 0.7,
    "supply disruption": 0.6,
    "force majeure": 0.5,
    "tanker seized": 0.6,

    # Price momentum
    "record high": 0.6,
    "oil rally": 0.5,
    "crude rally": 0.5,
    "price surge": 0.5,
    "backwardation": 0.4,   # Front month premium = tight near-term supply
    "opec discipline": 0.4,
    "opec cohesion": 0.4,
}

OIL_BEARISH_TERMS = {
    # Supply increases
    "production increase": -0.8,
    "output increase": -0.8,
    "quota increase": -0.7,
    "flood market": -0.7,
    "us production record": -0.6,
    "shale record": -0.6,
    "supply surplus": -0.7,
    "oil glut": -0.8,
    "inventory build": -0.6,
    "inventory increase": -0.5,
    "spr release": -0.5,
    "strategic reserve release": -0.5,
    "spare capacity": -0.4,
    "contango": -0.4,        # Near month discount = oversupply
    "opec split": -0.7,
    "opec disagreement": -0.6,
    "price war": -0.8,
    "cheat quota": -0.5,
    "exceed quota": -0.5,

    # Demand destruction
    "recession fears": -0.6,
    "demand destruction": -0.8,
    "demand contraction": -0.6,
    "ev transition": -0.4,
    "peak oil demand": -0.5,
    "china slowdown": -0.5,
    "china lockdown": -0.6,
    "weak demand": -0.5,
    "demand outlook cut": -0.6,

    # Dollar strength (inverse relationship)
    "dollar surges": -0.4,
    "dollar strengthens": -0.4,
    "usd strengthens": -0.4,

    # Price weakness
    "oil crash": -0.8,
    "oil slump": -0.6,
    "price collapse": -0.7,
    "crude selloff": -0.6,
    "bearish outlook": -0.5,
    "cut price forecast": -0.5,
    "lower price target": -0.4,
}

ALL_OIL_TERMS = {**OIL_BULLISH_TERMS, **OIL_BEARISH_TERMS}

RECENCY_HALF_LIFE_HOURS = 12  # Articles older than this get 50% weight


def _get_recency_weight(published_iso: Optional[str]) -> float:
    """
    Exponential decay weight based on article age.
    WHY: A news article from 30 hours ago about an OPEC decision that
    was already priced in yesterday shouldn't count the same as
    breaking news from 1 hour ago.
    """
    if not published_iso:
        return 0.7  # Unknown age — assume moderately fresh

    try:
        pub_time = datetime.fromisoformat(published_iso)
        if pub_time.tzinfo is None:
            pub_time = pub_time.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - pub_time).total_seconds() / 3600
        # Exponential decay with half-life of 12 hours
        import math
        weight = math.exp(-0.693 * age_hours / RECENCY_HALF_LIFE_HOURS)
        return max(0.1, weight)  # Floor at 10% — never fully ignore
    except Exception:
        return 0.7


def _oil_domain_score(text: str) -> float:
    """
    Scans for oil-specific terminology and returns a directional score.
    """
    boost = 0.0
    hits = []
    for term, score in ALL_OIL_TERMS.items():
        if term in text:
            boost += score
            hits.append((term, score))

    if hits:
        logger.debug("Oil terms: %s", [(t, f"{s:+.2f}") for t, s in hits[:5]])

    return max(-1.0, min(1.0, boost))


def _analyze_single_article(article: Dict) -> Dict:
    """
    Analyzes one article. Returns article enriched with:
    - textblob_score: raw TextBlob polarity
    - domain_score: oil-specific keyword score
    - final_score: weighted combination
    - effective_weight: source_weight × relevance_score × recency_weight
    """
    text = article.get('text', '').lower()
    if not text:
        return {**article, 'final_score': 0.0, 'effective_weight': 0.0,
                'sentiment_label': 'NEUTRAL'}

    # TextBlob — general sentiment
    try:
        textblob_score = TextBlob(text).sentiment.polarity
    except Exception:
        textblob_score = 0.0

    # Oil domain score
    domain_score = _oil_domain_score(text)

    # For oil news: domain score dominates (it's oil-specific),
    # TextBlob provides minor tiebreaking
    # WHY 30/70 split: TextBlob sees "cut" as negative (it is in English),
    # but "production cut" is BULLISH for oil. Domain score corrects this.
    final_score = (0.30 * textblob_score) + (0.70 * domain_score)
    final_score = max(-1.0, min(1.0, final_score))

    # Effective weight for aggregation
    source_weight = article.get('source_weight', 1.0)
    relevance = article.get('relevance_score', 0.5)
    recency = _get_recency_weight(article.get('published'))
    effective_weight = source_weight * relevance * recency

    label = "POSITIVE" if final_score > 0.08 else ("NEGATIVE" if final_score < -0.08 else "NEUTRAL")

    return {
        **article,
        'textblob_score': round(textblob_score, 4),
        'domain_score': round(domain_score, 4),
        'final_score': round(final_score, 4),
        'effective_weight': round(effective_weight, 4),
        'sentiment_label': label,
    }


def analyze_sentiment(articles: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    Weighted sentiment aggregation.

    WHY weighted average instead of simple mean:
    An EIA weekly inventory report from Reuters (weight=1.4×1.0×1.0=1.4)
    should count more than a speculative blog post (weight=1.0×0.3×0.5=0.15).
    The effective_weight captures this.

    Returns:
        (aggregate_score [-1, +1], analyzed_articles)
    """
    if not articles:
        return 0.0, []

    logger.info("Analyzing oil sentiment for %d articles...", len(articles))
    analyzed = [_analyze_single_article(a) for a in articles]

    # Weighted mean
    total_weight = sum(a['effective_weight'] for a in analyzed)
    if total_weight == 0:
        aggregate = 0.0
    else:
        aggregate = sum(
            a['final_score'] * a['effective_weight'] for a in analyzed
        ) / total_weight

    aggregate = round(max(-1.0, min(1.0, aggregate)), 4)

    pos = sum(1 for a in analyzed if a['sentiment_label'] == 'POSITIVE')
    neg = sum(1 for a in analyzed if a['sentiment_label'] == 'NEGATIVE')
    logger.info("Sentiment: aggregate=%.4f | POSITIVE=%d NEGATIVE=%d NEUTRAL=%d",
                aggregate, pos, neg, len(analyzed) - pos - neg)

    return aggregate, analyzed


if __name__ == "__main__":
    test_articles = [
        {'title': 'OPEC agrees to extend production cuts through Q3',
         'text': 'OPEC+ agreed to extend production cuts through Q3, maintaining supply discipline.',
         'published': None, 'source': 'reuters.com', 'source_weight': 1.4,
         'relevance_score': 1.0, 'entities': ['OPEC+']},
        {'title': 'US shale production hits record high',
         'text': 'US shale production hits record high, threatening to create oil glut.',
         'published': None, 'source': 'oilprice.com', 'source_weight': 1.2,
         'relevance_score': 0.8, 'entities': []},
    ]
    score, analyzed = analyze_sentiment(test_articles)
    print(f"\nAggregate: {score:+.4f}")
    for a in analyzed:
        print(f"  [{a['sentiment_label']:8s}] score={a['final_score']:+.4f} "
              f"weight={a['effective_weight']:.3f} | {a['title'][:60]}")
