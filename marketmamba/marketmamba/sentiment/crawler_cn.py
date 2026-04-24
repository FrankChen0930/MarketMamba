"""
MarketMamba V5.5 — 中文新聞爬蟲
透過 Google News RSS (TW) 擷取台股個股 / 大盤 / 政策新聞
來源：鉅亨網、UDN 經濟日報、工商時報、MoneyDJ 等
"""

import logging
from datetime import datetime, timedelta
from xml.etree import ElementTree

import requests

from marketmamba.config import (
    NEWS_SOURCES_WHITELIST_CN, NEWS_LOOKBACK_DAYS,
    MARKET_KEYWORDS_TW,
)

logger = logging.getLogger('MarketMamba.crawler_cn')

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def _parse_google_news_rss_tw(query: str, days: int = 3) -> list[dict]:
    """
    解析 Google News RSS Feed (台灣繁體中文)

    Args:
        query: 搜尋關鍵字
        days: 回看天數

    Returns:
        [{"title": str, "source": str, "pub_date": str, "link": str}]
    """
    params = {
        'q': query,
        'hl': 'zh-TW',
        'gl': 'TW',
        'ceid': 'TW:zh-Hant',
    }

    try:
        res = requests.get(GOOGLE_NEWS_RSS, params=params, timeout=10)
        res.raise_for_status()
    except Exception as e:
        logger.warning(f"⚠️ Google News RSS (TW) 請求失敗: {e}")
        return []

    articles = []
    cutoff = datetime.now() - timedelta(days=days)

    try:
        root = ElementTree.fromstring(res.content)
        for item in root.findall('.//item'):
            title = item.findtext('title', '')
            link = item.findtext('link', '')
            pub_date_str = item.findtext('pubDate', '')
            source = item.findtext('source', '')

            try:
                pub_date = datetime.strptime(
                    pub_date_str, '%a, %d %b %Y %H:%M:%S %Z'
                )
            except (ValueError, TypeError):
                pub_date = datetime.now()

            if pub_date < cutoff:
                continue

            source_domain = source.lower() if source else ''
            link_lower = link.lower()
            is_whitelisted = any(
                ws in source_domain or ws in link_lower
                for ws in NEWS_SOURCES_WHITELIST_CN
            )

            articles.append({
                'title': title.strip(),
                'source': source,
                'pub_date': pub_date.strftime('%Y-%m-%d'),
                'link': link,
                'whitelisted': is_whitelisted,
            })
    except ElementTree.ParseError as e:
        logger.warning(f"⚠️ RSS XML 解析錯誤: {e}")

    return articles


def crawl_tw_market_news(days: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """
    擷取台股大盤 / 總經相關中文新聞
    """
    query = ' OR '.join(f'"{kw}"' for kw in MARKET_KEYWORDS_TW[:5])
    articles = _parse_google_news_rss_tw(query, days=days)
    logger.info(f"🇹🇼 台股大盤新聞: 擷取 {len(articles)} 篇")
    return articles


def crawl_stock_news_cn(stock_id: str, stock_name: str,
                        days: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """
    擷取特定個股的中文新聞

    Args:
        stock_id: 股票代號 (如 "2330")
        stock_name: 公司名稱 (如 "台積電")
        days: 回看天數
    """
    query = f'"{stock_id}" OR "{stock_name}"'
    articles = _parse_google_news_rss_tw(query, days=days)
    logger.info(f"📰 {stock_id} {stock_name} 中文新聞: {len(articles)} 篇")
    return articles


def crawl_all_cn_news(stock_list: list[tuple] = None,
                      days: int = NEWS_LOOKBACK_DAYS) -> dict:
    """
    一次擷取所有中文新聞

    Args:
        stock_list: [(stock_id, stock_name), ...] 要爬取個股新聞的清單
                    如未傳入則只爬大盤新聞

    Returns:
        {
            "market_tw": [...],
            "stocks": {"2330": [...], "2317": [...], ...},
        }
    """
    result = {
        "market_tw": crawl_tw_market_news(days),
        "stocks": {},
    }

    if stock_list:
        for stock_id, stock_name in stock_list:
            result["stocks"][stock_id] = crawl_stock_news_cn(
                stock_id, stock_name, days
            )

    return result
