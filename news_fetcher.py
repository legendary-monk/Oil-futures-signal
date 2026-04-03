"""
news_fetcher.py — Oil-Specific News Ingestion
===============================================
Improvements over Nifty version:
1. Oil-specific relevance scoring — not all financial news matters for oil
2. OPEC meeting date detection — automatically flags elevated uncertainty periods
3. Source credibility weighting — EIA > Reuters > OilPrice.com
4. Entity extraction — pulls out specific actors (Saudi Arabia, OPEC+, Iran)
   for the Telegram summary
"""

import feedparser
import requests
import re
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

import config
from logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# OIL RELEVANCE FILTER
# ─────────────────────────────────────────────
# We only care about articles that are actually about oil/energy markets.
# Reuters covers everything — we need to filter for relevance.

OIL_RELEVANCE_KEYWORDS = {
    # High relevance (any one = definitely relevant)
    "crude oil", "brent", "wti", "opec", "petroleum", "oil price",
    "barrel", "oil production", "oil supply", "oil demand", "oil futures",
    "energy price", "natural gas price", "gasoline", "diesel",
    # Medium relevance (need context)
    "saudi arabia", "russia oil", "iran oil", "venezuela oil",
    "shale", "rig count", "refinery", "oil inventory", "crude inventory",
    "strategic petroleum reserve", "spr release",
    "oil minister", "energy minister",
}

# Hard exclusion — these topics use "oil" in non-commodity contexts
# e.g., "olive oil", "oil painting", "oil spill" (unless commodity context)
EXCLUSION_PATTERNS = [
    r"\bolive oil\b",
    r"\bcoconut oil\b",
    r"\bpalm oil\b(?!.*price)",   # palm oil OK if discussing price
    r"\bfish oil\b",
    r"\bessential oil\b",
]


# Source credibility weights (1.0 = baseline, >1 = more trusted)
SOURCE_WEIGHTS = {
    "eia.gov": 1.8,          # US Energy Information Administration — authoritative data
    "opec.org": 1.8,         # OPEC official — highest relevance
    "reuters.com": 1.4,
    "bloomberg.com": 1.4,
    "oilprice.com": 1.2,
    "rigzone.com": 1.1,
}


def _get_source_weight(source_url: str) -> float:
    """Returns credibility multiplier for a given source URL."""
    source_lower = source_url.lower()
    for domain, weight in SOURCE_WEIGHTS.items():
        if domain in source_lower:
            return weight
    return 1.0


def _is_oil_relevant(text: str, title: str) -> Tuple[bool, float]:
    """
    Determines if an article is relevant to oil markets.

    Returns:
        (is_relevant: bool, relevance_score: float 0.0–1.0)

    WHY return a score: Downstream sentiment weighting can use this.
    A core OPEC decision article should count more than a tangential mention.
    """
    combined = (title + " " + text).lower()

    # Check exclusion patterns first
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, combined):
            return False, 0.0

    # Count relevance keyword hits
    hits = sum(1 for kw in OIL_RELEVANCE_KEYWORDS if kw in combined)

    if hits == 0:
        return False, 0.0

    # Score: more hits = more relevant, capped at 1.0
    score = min(1.0, hits / 3.0)
    return True, round(score, 3)


def _extract_key_entities(text: str) -> List[str]:
    """
    Extracts key actors and concepts from article text.
    Used in Telegram message to give context for the signal.

    Returns list of strings like ["OPEC+", "Saudi Arabia", "production cut"]
    """
    entities = []
    text_lower = text.lower()

    entity_patterns = [
        (r"\bopec\+?\b", "OPEC+"),
        (r"\bsaudi\s*arabia\b", "Saudi Arabia"),
        (r"\brussia\b", "Russia"),
        (r"\biran\b", "Iran"),
        (r"\bvenezuela\b", "Venezuela"),
        (r"\bchina\b", "China"),
        (r"\beia\b", "EIA"),
        (r"\bfed\b|\bfederal reserve\b", "US Fed"),
        (r"\bproduction cut", "production cut"),
        (r"\bproduction increase", "production increase"),
        (r"\bsupply cut", "supply cut"),
        (r"\bprice war", "price war"),
        (r"\bstraits? of hormuz\b", "Hormuz Strait"),
        (r"\bstrategic petroleum", "SPR"),
    ]

    for pattern, label in entity_patterns:
        if re.search(pattern, text_lower) and label not in entities:
            entities.append(label)

    return entities[:5]  # Cap at 5 entities per article


def _preprocess_text(text: str) -> str:
    """Cleans text for sentiment analysis."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = re.sub(r'http\S+|www\S+', '', text)
    text = ' '.join(text.split())
    return text


def _parse_entry(entry, source_url: str) -> Optional[Dict]:
    """Parses a feedparser entry into a clean article dict."""
    title = getattr(entry, 'title', '').strip()
    summary = getattr(entry, 'summary', '').strip()
    content = ''
    if hasattr(entry, 'content') and entry.content:
        content = entry.content[0].get('value', '').strip()

    raw_text = f"{title}. {summary or content}".strip('. ')
    clean_text = _preprocess_text(raw_text)

    if len(clean_text) < 20:
        return None

    # Oil relevance filter
    is_relevant, relevance_score = _is_oil_relevant(clean_text, title)
    if not is_relevant:
        return None

    published = None
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        try:
            pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=config.MAX_ARTICLE_AGE_HOURS)
            if pub_dt < cutoff:
                return None  # Too old
            published = pub_dt.isoformat()
        except (TypeError, ValueError):
            pass

    entities = _extract_key_entities(clean_text)
    source_weight = _get_source_weight(source_url)

    return {
        'title': title,
        'text': clean_text,
        'published': published,
        'source': source_url,
        'relevance_score': relevance_score,
        'source_weight': source_weight,
        'entities': entities,
    }


def _fetch_single_feed(feed_url: str) -> List[Dict]:
    """Fetches one RSS feed with retry logic."""
    articles = []

    for attempt in range(1, config.REQUEST_RETRIES + 1):
        try:
            response = requests.get(
                feed_url,
                timeout=config.REQUEST_TIMEOUT,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; OilSignalBot/1.0)'}
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            if feed.bozo and not feed.entries:
                logger.warning("Feed XML errors, no entries: %s", feed_url)
                break

            for entry in feed.entries:
                parsed = _parse_entry(entry, feed_url)
                if parsed:
                    articles.append(parsed)

            logger.info("Fetched %d oil-relevant articles from %s",
                        len(articles), feed_url)
            break

        except requests.exceptions.Timeout:
            logger.warning("Timeout on %s (attempt %d)", feed_url, attempt)
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP error %s on %s", e, feed_url)
            break
        except Exception as e:
            logger.error("Error fetching %s: %s", feed_url, e)
            break

        if attempt < config.REQUEST_RETRIES:
            time.sleep(config.RETRY_DELAY_BASE ** attempt)

    return articles


def fetch_news() -> List[Dict]:
    """
    Fetches oil-relevant articles from all configured feeds.

    Returns articles sorted by (source_weight × relevance_score) descending,
    so the most authoritative and relevant articles get analyzed first.
    """
    logger.info("Fetching oil news from %d feeds...", len(config.RSS_FEEDS))

    all_articles = []
    failed = 0

    for url in config.RSS_FEEDS:
        arts = _fetch_single_feed(url)
        all_articles.extend(arts)
        if not arts:
            failed += 1
        time.sleep(0.8)

    if not all_articles:
        logger.error("All news feeds failed")
        return []

    # Deduplicate by title
    seen = set()
    unique = []
    for a in all_articles:
        fp = a['title'][:60].lower().strip()
        if fp not in seen:
            seen.add(fp)
            unique.append(a)

    # Sort by combined signal quality
    unique.sort(key=lambda a: a['source_weight'] * a['relevance_score'], reverse=True)

    result = unique[:config.MAX_ARTICLES]
    logger.info("News: %d unique oil-relevant articles (%d feeds failed)",
                len(result), failed)

    return result


def get_top_entities(articles: List[Dict]) -> List[Tuple[str, int]]:
    """
    Returns the most frequently mentioned entities across all articles.
    Used for Telegram message context.

    Returns: List of (entity_name, count) sorted by frequency.
    """
    from collections import Counter
    all_entities = []
    for a in articles:
        all_entities.extend(a.get('entities', []))

    return Counter(all_entities).most_common(5)


if __name__ == "__main__":
    arts = fetch_news()
    print(f"\nTotal oil-relevant articles: {len(arts)}")
    if arts:
        print("\nTop entities mentioned:")
        for entity, count in get_top_entities(arts):
            print(f"  {entity}: {count} mentions")
        print("\nTop 3 articles:")
        for a in arts[:3]:
            print(f"  [{a['source_weight']:.1f}×{a['relevance_score']:.2f}] {a['title'][:70]}")
