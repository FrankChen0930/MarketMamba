"""
MarketMamba V5.5 — 英文新聞爬蟲 (多來源架構)
透過多個 RSS Feed 擷取國際財經 / 地緣政治新聞

來源清單：
  1. Google News RSS (主力，覆蓋最廣)
  2. Yahoo Finance RSS (美股即時)
  3. Finviz RSS (量化圈常用)

去重機制：標題前 40 字元相似度比對，自動過濾重複報導
"""

import logging
from datetime import datetime, timedelta
from xml.etree import ElementTree
from difflib import SequenceMatcher

import requests

from marketmamba.config import (
    NEWS_SOURCES_WHITELIST_EN, NEWS_LOOKBACK_DAYS,
    MARKET_KEYWORDS_US, GEOPOLITICAL_KEYWORDS_EN,
)

logger = logging.getLogger('MarketMamba.crawler_en')

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
YAHOO_FINANCE_RSS = "https://finance.yahoo.com/news/rssindex"
FINVIZ_RSS = "https://finviz.com/news_export.ashx?v=3"

# 模擬瀏覽器標頭，防止 403 Forbidden / 429 Too Many Requests
BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


# ==========================================
# 去重引擎
# ==========================================
def _deduplicate(articles: list[dict], threshold: float = 0.7) -> list[dict]:
    """
    基於標題相似度的去重

    策略：取標題前 40 字元比對，相似度 > threshold 視為重複
    保留最早出現的一篇 (先到先得)
    """
    unique = []
    seen_fps = []

    for article in articles:
        title = article.get('title', '')
        fp = title[:40].lower().strip()

        is_dup = False
        for seen_fp in seen_fps:
            if SequenceMatcher(None, fp, seen_fp).ratio() > threshold:
                is_dup = True
                break

        if not is_dup:
            unique.append(article)
            seen_fps.append(fp)

    deduped = len(articles) - len(unique)
    if deduped > 0:
        logger.info(f"  🔄 去重: 移除 {deduped} 篇重複新聞")
    return unique


# ==========================================
# 通用 RSS 解析器
# ==========================================
def _parse_rss(url: str, params: dict = None, days: int = 3,
               source_label: str = "RSS") -> list[dict]:
    """通用 RSS Feed 解析，支援多種日期格式"""
    try:
        # 使用自定義 Headers 避免被阻擋
        res = requests.get(url, params=params, headers=BROWSER_HEADERS, timeout=10)
        res.raise_for_status()
    except Exception as e:
        logger.warning(f"⚠️ {source_label} 請求失敗: {e} for url: {url}")
        return []

    articles = []
    cutoff = datetime.now() - timedelta(days=days)

    try:
        root = ElementTree.fromstring(res.content)
        for item in root.findall('.//item'):
            title = item.findtext('title', '').strip()
            link = item.findtext('link', '')
            pub_date_str = item.findtext('pubDate', '')
            source = item.findtext('source', source_label)

            if not title:
                continue

            # 多種日期格式容錯解析
            pub_date = None
            for fmt in [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S%z',
            ]:
                try:
                    pub_date = datetime.strptime(pub_date_str[:30].strip(), fmt)
                    if pub_date.tzinfo:
                        pub_date = pub_date.replace(tzinfo=None)
                    break
                except (ValueError, TypeError):
                    continue
            if pub_date is None:
                pub_date = datetime.now()

            if pub_date < cutoff:
                continue

            source_domain = (source or '').lower()
            link_lower = link.lower()
            is_whitelisted = any(
                ws in source_domain or ws in link_lower
                for ws in NEWS_SOURCES_WHITELIST_EN
            )

            articles.append({
                'title': title,
                'source': source or source_label,
                'pub_date': pub_date.strftime('%Y-%m-%d'),
                'link': link,
                'whitelisted': is_whitelisted,
                'feed': source_label,
            })
    except ElementTree.ParseError as e:
        logger.warning(f"⚠️ {source_label} XML 解析錯誤: {e}")

    return articles


# ==========================================
# 各來源擷取器
# ==========================================
def _parse_google_news_rss(query: str, days: int = 3,
                           lang: str = 'en', gl: str = 'US') -> list[dict]:
    """Google News RSS"""
    return _parse_rss(
        GOOGLE_NEWS_RSS,
        params={'q': query, 'hl': lang, 'gl': gl, 'ceid': f'{gl}:{lang}'},
        days=days,
        source_label="GoogleNews",
    )


def _fetch_yahoo_finance_rss(days: int = 3) -> list[dict]:
    """Yahoo Finance RSS (美股焦點)"""
    return _parse_rss(YAHOO_FINANCE_RSS, days=days, source_label="YahooFinance")


def _fetch_finviz_rss(days: int = 3) -> list[dict]:
    """Finviz RSS (量化圈常用)"""
    return _parse_rss(FINVIZ_RSS, days=days, source_label="Finviz")


# ==========================================
# 公開 API
# ==========================================
def crawl_global_market_news(days: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """
    多來源擷取全球市場新聞

    來源：Google News + Yahoo Finance + Finviz → 自動去重
    """
    query = ' OR '.join(f'"{kw}"' for kw in MARKET_KEYWORDS_US[:5])

    all_articles = []
    all_articles.extend(_parse_google_news_rss(query, days=days))
    all_articles.extend(_fetch_yahoo_finance_rss(days=days))
    all_articles.extend(_fetch_finviz_rss(days=days))

    unique = _deduplicate(all_articles)
    feeds_used = len(set(a.get('feed', '') for a in unique))
    logger.info(f"📰 全球市場新聞: {len(unique)} 篇 (來自 {feeds_used} 個來源)")
    return unique


def crawl_geopolitical_news(days: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """擷取地緣政治相關新聞 (戰爭/制裁/關稅)"""
    query = ' OR '.join(f'"{kw}"' for kw in GEOPOLITICAL_KEYWORDS_EN[:5])
    articles = _parse_google_news_rss(query, days=days)
    logger.info(f"🌍 地緣政治新聞: 擷取 {len(articles)} 篇")
    return articles


def crawl_stock_news_en(stock_name_en: str,
                        days: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """擷取特定公司的英文新聞"""
    articles = _parse_google_news_rss(f'"{stock_name_en}" stock', days=days)
    logger.info(f"📰 {stock_name_en} 英文新聞: {len(articles)} 篇")
    return articles


def crawl_all_en_news(days: int = NEWS_LOOKBACK_DAYS) -> dict:
    """
    一次擷取所有英文新聞 (多來源 + 去重)

    Returns:
        {"market": [...], "geopolitical": [...]}
    """
    return {
        "market": crawl_global_market_news(days),
        "geopolitical": crawl_geopolitical_news(days),
    }
