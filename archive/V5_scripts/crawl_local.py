"""
MarketMamba V5.5 — 本機新聞爬蟲 + 情緒分析
在本機多核心 CPU 上執行爬蟲，可選擇性執行 FinBERT (需要本機有 GPU/CPU)
產出的快取目錄可直接上傳 Google Drive，供 Colab 訓練使用

使用方式：
  cd MarketMamba
  python scripts/crawl_local.py --days 30 --output ./News_Cache

  # 完成後，將 News_Cache 資料夾上傳到 Google Drive:
  #   MyDrive/MarketMamba_V5/News_Cache/
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 確保能匯入 marketmamba
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marketmamba.sentiment.crawler_en import (
    crawl_global_market_news, crawl_geopolitical_news,
)
from marketmamba.sentiment.crawler_cn import (
    crawl_tw_market_news, crawl_stock_news_cn,
)
from marketmamba.config import NEWS_CACHE_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('LocalCrawler')

# ==========================================
# 台股熱門個股名稱對照表 (用於個股新聞爬取)
# ==========================================
TOP_STOCK_NAMES = {
    '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2308': '台達電',
    '2382': '廣達', '2412': '中華電', '2881': '富邦金', '2882': '國泰金',
    '2303': '聯電', '3711': '日月光', '2886': '兆豐金', '2891': '中信金',
    '2884': '玉山金', '2885': '元大金', '2887': '台新金', '2603': '長榮',
    '2609': '陽明', '2615': '萬海', '2002': '中鋼', '1301': '台塑',
    '1303': '南亞', '1326': '台化', '2207': '和泰車', '3034': '聯詠',
    '2357': '華碩', '2301': '光寶科', '6505': '台塑化', '3037': '欣興',
    '2345': '智邦', '5274': '信驊', '6415': '矽力-KY', '3661': '世芯-KY',
    '2327': '國巨', '2356': '英業達', '3231': '緯創', '2353': '宏碁',
    '2344': '華邦電', '2449': '京元電', '3443': '創意', '6770': '力積電',
    '2492': '華新科', '3665': '貿聯-KY', '4904': '遠傳', '1216': '統一',
    '2801': '彰銀', '5880': '合庫金', '2890': '永豐金', '2892': '第一金',
    '9958': '世紀鋼', '3045': '台灣大',
}


def crawl_all_news(days: int = 30, include_stocks: bool = True,
                   max_workers: int = 4) -> dict:
    """
    多執行緒批次爬取所有新聞

    Args:
        days: 回看天數
        include_stocks: 是否爬取個股新聞
        max_workers: 最大併發數 (建議 2~8，太多會被 Google 封鎖)

    Returns:
        {"market_en": [...], "geopolitical": [...], "market_tw": [...], "stocks": {...}}
    """
    all_news = {
        'market_en': [],
        'geopolitical': [],
        'market_tw': [],
        'stocks': {},
    }

    # 1. 市場 + 地緣 (序列執行，量少不需並行)
    logger.info(f"LOCAL: Crawling Global Market News (Lookback {days} days)...")
    all_news['market_en'] = crawl_global_market_news(days)
    time.sleep(1)

    logger.info(f"LOCAL: Crawling Geopolitical News...")
    all_news['geopolitical'] = crawl_geopolitical_news(days)
    time.sleep(1)

    logger.info(f"LOCAL: Crawling Taiwan Market News...")
    all_news['market_tw'] = crawl_tw_market_news(days)
    time.sleep(1)

    # 2. 個股新聞 (並行爬取，因為數量多)
    if include_stocks:
        logger.info(f"LOCAL: Crawling {len(TOP_STOCK_NAMES)} Stock News (Parallel {max_workers} threads)...")

        def _crawl_one_stock(stock_id, stock_name):
            articles = crawl_stock_news_cn(stock_id, stock_name, days)
            time.sleep(0.5)  # 禮貌延遲
            return stock_id, articles

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_crawl_one_stock, sid, name): sid
                for sid, name in TOP_STOCK_NAMES.items()
            }
            done_count = 0
            for future in as_completed(futures):
                stock_id, articles = future.result()
                all_news['stocks'][stock_id] = articles
                done_count += 1
                if done_count % 10 == 0:
                    logger.info(f"  進度: {done_count}/{len(TOP_STOCK_NAMES)}")

    # 統計
    total = (
        len(all_news['market_en']) +
        len(all_news['geopolitical']) +
        len(all_news['market_tw']) +
        sum(len(v) for v in all_news['stocks'].values())
    )
    logger.info(f"DONE: Crawler finished! Total {total} articles.")

    return all_news


def save_news_cache(all_news: dict, output_dir: str) -> str:
    """
    儲存新聞快取到本機目錄

    目錄結構：
      News_Cache/
      └── raw/
          └── YYYY-MM-DD_news_bundle.json  (所有新聞合存一個檔案)
    """
    raw_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    today_str = datetime.now().strftime('%Y-%m-%d')
    save_path = os.path.join(raw_dir, f'{today_str}_news_bundle.json')

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_news, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 新聞快取已儲存: {save_path}")
    return save_path


def run_local_finbert(all_news: dict, output_dir: str) -> None:
    """
    在本機執行 FinBERT 分析 (需要安裝 transformers + torch)

    如果本機沒有 GPU，會用 CPU 跑 (慢但可以)
    產出：sentiment_scores.json + embeddings_en.npy + embeddings_cn.npy
    """
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🧠 開始本機 FinBERT 分析 (設備: {device})...")
    except ImportError:
        logger.warning("⚠️ 未安裝 torch/transformers，跳過 FinBERT 分析")
        logger.warning("   請在 Colab 上執行 FinBERT，或安裝: pip install torch transformers")
        return

    import numpy as np
    from marketmamba.sentiment.finbert_en import FinBERTEnglish
    from marketmamba.sentiment.finbert_cn import FinBERTChinese

    processed_dir = os.path.join(output_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # EN
    finbert_en = FinBERTEnglish(device=device)

    market_en_titles = [a['title'] for a in all_news.get('market_en', [])]
    geo_titles = [a['title'] for a in all_news.get('geopolitical', [])]
    all_en_titles = market_en_titles + geo_titles

    en_scores = finbert_en.get_sentiment(all_en_titles) if all_en_titles else []
    en_embeds = finbert_en.get_embedding(all_en_titles) if all_en_titles else np.zeros((0, 16))

    # CN
    finbert_cn = FinBERTChinese(device=device)

    market_tw_titles = [a['title'] for a in all_news.get('market_tw', [])]
    all_cn_titles = market_tw_titles

    stock_article_map = {}  # {stock_id: [titles]}
    for stock_id, articles in all_news.get('stocks', {}).items():
        titles = [a['title'] for a in articles]
        if titles:
            stock_article_map[stock_id] = titles
            all_cn_titles.extend(titles)

    cn_scores = finbert_cn.get_sentiment(all_cn_titles) if all_cn_titles else []
    cn_embeds = finbert_cn.get_embedding(all_cn_titles) if all_cn_titles else np.zeros((0, 16))

    # 聚合結果
    n_market_en = len(market_en_titles)
    n_geo = len(geo_titles)
    n_market_tw = len(market_tw_titles)

    results = {
        'market_us_score': float(np.mean(en_scores[:n_market_en])) if n_market_en > 0 else 0.0,
        'geopolitical_score': float(np.mean(en_scores[n_market_en:n_market_en + n_geo])) if n_geo > 0 else 0.0,
        'market_tw_score': float(np.mean(cn_scores[:n_market_tw])) if n_market_tw > 0 else 0.0,
        'stocks': {},
    }

    # 個股情緒
    cn_offset = n_market_tw
    for stock_id, titles in stock_article_map.items():
        n = len(titles)
        stock_scores = cn_scores[cn_offset:cn_offset + n]
        results['stocks'][stock_id] = {
            'score': float(np.mean(stock_scores)) if stock_scores else 0.0,
            'count': n,
        }
        cn_offset += n

    today_str = datetime.now().strftime('%Y-%m-%d')

    # 儲存分數
    scores_path = os.path.join(processed_dir, f'{today_str}_sentiment_scores.json')
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 儲存 embeddings
    if len(en_embeds) > 0:
        np.save(os.path.join(processed_dir, f'{today_str}_embeddings_en.npy'), en_embeds)
    if len(cn_embeds) > 0:
        np.save(os.path.join(processed_dir, f'{today_str}_embeddings_cn.npy'), cn_embeds)

    del finbert_en, finbert_cn
    logger.info(f"✅ FinBERT 分析完成！結果已儲存至 {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description='MarketMamba V5.5 本機新聞爬蟲')
    parser.add_argument('--days', type=int, default=30,
                        help='回看天數 (預設 30)')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出目錄 (預設: config 中的 NEWS_CACHE_DIR)')
    parser.add_argument('--workers', type=int, default=4,
                        help='爬蟲並行數 (預設 4，建議 2~8)')
    parser.add_argument('--no-stocks', action='store_true',
                        help='不爬取個股新聞 (只爬大盤/地緣)')
    parser.add_argument('--run-finbert', action='store_true',
                        help='在本機執行 FinBERT 分析 (需要 torch + transformers)')
    args = parser.parse_args()

    output_dir = args.output or NEWS_CACHE_DIR

    print("=" * 60)
    print("MarketMamba V5.5 — 本機新聞爬蟲")
    print(f"DAYS: {args.days} | WORKERS: {args.workers}")
    print(f"OUTPUT: {output_dir}")
    print("=" * 60)

    # 1. 爬取
    all_news = crawl_all_news(
        days=args.days,
        include_stocks=not args.no_stocks,
        max_workers=args.workers,
    )

    # 2. 儲存
    save_news_cache(all_news, output_dir)

    # 3. FinBERT (選配)
    if args.run_finbert:
        run_local_finbert(all_news, output_dir)

    print("\n" + "=" * 60)
    print("DONE! 接下來請將快取上傳到 Google Drive:")
    print(f"   本機: {output_dir}")
    print(f"   目標: MyDrive/MarketMamba_V5/News_Cache/")
    print("=" * 60)


if __name__ == '__main__':
    main()
