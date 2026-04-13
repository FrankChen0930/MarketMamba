"""
MarketMamba V5.5 — 情緒特徵整合模組
負責：新聞爬取 → FinBERT 分析 → 特徵聚合 → Forward Fill + 指數衰減 → 併入主矩陣

優化：
  - 合併所有中文標題為一次 forward pass (不再逐股跑 FinBERT)
  - 個股新聞預先收集、批次分析、再分配回去
  - 衰減計算向量化
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from marketmamba.config import (
    SENTIMENT_SCALAR_COLS, SENTIMENT_EMBED_EN_COLS, SENTIMENT_EMBED_CN_COLS,
    SENTIMENT_HALF_LIFE, FINBERT_EMBED_DIM, NEWS_CACHE_DIR,
    GEOPOLITICAL_KEYWORDS_EN, GEOPOLITICAL_KEYWORDS_CN,
)

logger = logging.getLogger('MarketMamba.integrator')


def _decay_sentiment(series: pd.Series, half_life: int = SENTIMENT_HALF_LIFE) -> pd.Series:
    """
    情緒分數指數衰減：模擬消息效應隨時間遞減

    score_t = score_{t-1} × 0.5^(1/half_life)

    無新聞的日子繼承前日情緒但逐步衰減至 0
    """
    decay_factor = 0.5 ** (1 / half_life)
    vals = series.values.copy()

    for i in range(1, len(vals)):
        if np.isnan(vals[i]) or vals[i] == 0:
            prev = vals[i - 1]
            if not np.isnan(prev) and prev != 0:
                vals[i] = prev * decay_factor

    return pd.Series(vals, index=series.index).fillna(0)


def _is_geopolitical(title: str) -> bool:
    """判斷標題是否為地緣政治相關"""
    title_lower = title.lower()
    return any(kw in title_lower for kw in GEOPOLITICAL_KEYWORDS_EN + GEOPOLITICAL_KEYWORDS_CN)


def compute_sentiment_features(df: pd.DataFrame,
                               ticker_name_map: dict = None,
                               top_n_stocks: int = 50,
                               precomputed_news: dict = None) -> pd.DataFrame:
    """
    完整的情緒特徵計算管線 (優化版)

    優化重點：
    - EN: 市場 + 地緣合併為單次 analyze() 呼叫
    - CN: 所有個股標題合併為單次 analyze() 呼叫，再用 offset 映射回各股
    - FP16 自動啟用 (GPU)

    Args:
        df: 主特徵矩陣
        ticker_name_map: {stock_id: stock_name} 映射
        top_n_stocks: Top N 熱門股票
        precomputed_news: 預載的新聞 dict (可選)

    Returns:
        加入情緒特徵後的 DataFrame
    """
    print("📰 啟動雙軌情緒引擎...")

    from marketmamba.sentiment.crawler_en import crawl_all_en_news
    from marketmamba.sentiment.crawler_cn import crawl_all_cn_news
    from marketmamba.sentiment.finbert_en import FinBERTEnglish
    from marketmamba.sentiment.finbert_cn import FinBERTChinese

    # === 1. 載入或爬取新聞 ===
    if precomputed_news:
        print("  ✅ 使用預載新聞資料")
        en_news = {
            'market': precomputed_news.get('market_en', []),
            'geopolitical': precomputed_news.get('geopolitical', [])
        }
        cn_news = {
            'market_tw': precomputed_news.get('market_tw', []),
            'stocks': precomputed_news.get('stocks', {})
        }
    else:
        print("  🌐 爬取英文新聞...")
        en_news = crawl_all_en_news()

        stock_list = []
        if ticker_name_map:
            top_stocks = (
                df.groupby('stock_id')['Volume'].mean()
                .nlargest(top_n_stocks).index.tolist()
            )
            stock_list = [
                (sid, ticker_name_map.get(sid, sid))
                for sid in top_stocks if sid in ticker_name_map
            ]

        print("  🇹🇼 爬取中文新聞...")
        cn_news = crawl_all_cn_news(stock_list=stock_list)

    # === 2. FinBERT-EN: 市場 + 地緣合併為單次推論 ===
    market_en_titles = [a['title'] for a in en_news.get('market', [])]
    geo_titles = [a['title'] for a in en_news.get('geopolitical', [])]
    all_en_titles = market_en_titles + geo_titles

    n_market_en = len(market_en_titles)
    n_geo = len(geo_titles)

    print(f"  🧠 FinBERT-EN 情緒分析 ({len(all_en_titles)} 篇)...")
    finbert_en = FinBERTEnglish()

    if all_en_titles:
        en_result = finbert_en.analyze(all_en_titles)
        en_scores = en_result["scores"]
        en_embeds = en_result["embeddings"]

        market_en_scores = en_scores[:n_market_en]
        geo_scores = en_scores[n_market_en:]
        market_en_embed = en_embeds[:n_market_en]
    else:
        market_en_scores, geo_scores = [], []
        market_en_embed = np.zeros((0, FINBERT_EMBED_DIM))

    # 釋放 EN 模型
    del finbert_en
    _clear_gpu_cache()

    # === 3. FinBERT-CN: 所有中文標題合併為單次推論 ===
    market_tw_titles = [a['title'] for a in cn_news.get('market_tw', [])]

    # 收集所有個股標題，記錄 offset 用於映射回去
    stock_title_ranges = {}  # {stock_id: (start_idx, end_idx)}
    all_stock_titles = []
    for stock_id, articles in cn_news.get('stocks', {}).items():
        if articles:
            titles = [a['title'] for a in articles]
            start = len(all_stock_titles)
            all_stock_titles.extend(titles)
            stock_title_ranges[stock_id] = (start, start + len(titles), len(titles))

    all_cn_titles = market_tw_titles + all_stock_titles
    n_market_tw = len(market_tw_titles)

    print(f"  🧠 FinBERT-CN 情緒分析 ({len(all_cn_titles)} 篇)...")
    finbert_cn = FinBERTChinese()

    if all_cn_titles:
        cn_result = finbert_cn.analyze(all_cn_titles)
        cn_scores = cn_result["scores"]
        cn_embeds = cn_result["embeddings"]

        market_tw_scores = cn_scores[:n_market_tw]
    else:
        market_tw_scores = []
        cn_scores = []
        cn_embeds = np.zeros((0, FINBERT_EMBED_DIM))

    # 釋放 CN 模型
    del finbert_cn
    _clear_gpu_cache()

    # === 4. 聚合為特徵 ===
    latest_date = df['Date'].max()

    sent_market_us = float(np.mean(market_en_scores)) if market_en_scores else 0.0
    sent_geopolitical = float(np.mean(geo_scores)) if geo_scores else 0.0
    sent_market_tw = float(np.mean(market_tw_scores)) if market_tw_scores else 0.0

    market_embed_en = market_en_embed.mean(axis=0) if len(market_en_embed) > 0 else np.zeros(FINBERT_EMBED_DIM)

    # 個股情緒：從合併結果中用 offset 切回
    stock_sentiments = {}
    for stock_id, (start, end, count) in stock_title_ranges.items():
        offset = n_market_tw  # 個股在 all_cn_titles 中的偏移
        s_scores = cn_scores[offset + start:offset + end]
        s_embeds = cn_embeds[offset + start:offset + end]

        stock_sentiments[stock_id] = {
            'cn_score': float(np.mean(s_scores)) if s_scores else 0.0,
            'news_count': count,
            'embed_cn': s_embeds.mean(axis=0) if len(s_embeds) > 0 else np.zeros(FINBERT_EMBED_DIM),
        }

    # === 5. 併入 DataFrame ===
    for col in SENTIMENT_SCALAR_COLS:
        if col not in df.columns:
            df[col] = 0.0
    for col in SENTIMENT_EMBED_EN_COLS + SENTIMENT_EMBED_CN_COLS:
        if col not in df.columns:
            df[col] = 0.0

    mask = df['Date'] == latest_date

    df.loc[mask, 'Sent_Market_US'] = sent_market_us
    df.loc[mask, 'Sent_Geopolitical'] = sent_geopolitical
    df.loc[mask, 'Sent_Market_TW'] = sent_market_tw

    for i, col in enumerate(SENTIMENT_EMBED_EN_COLS):
        df.loc[mask, col] = market_embed_en[i]

    print(f"  📊 填入個股情緒 ({len(stock_sentiments)} 支)...")
    for stock_id, sent_data in stock_sentiments.items():
        stock_mask = mask & (df['stock_id'].astype(str) == stock_id)
        df.loc[stock_mask, 'Sent_Stock_CN'] = sent_data['cn_score']
        df.loc[stock_mask, 'News_Volume_Stock'] = np.log1p(sent_data['news_count'])

        for i, col in enumerate(SENTIMENT_EMBED_CN_COLS):
            df.loc[stock_mask, col] = sent_data['embed_cn'][i]

    # === 6. Forward Fill + 指數衰減 ===
    print("  ⏳ 情緒衰減計算...")
    all_sentiment_cols = SENTIMENT_SCALAR_COLS + SENTIMENT_EMBED_EN_COLS + SENTIMENT_EMBED_CN_COLS
    for col in tqdm(all_sentiment_cols, desc="  衰減處理", unit="col"):
        if col in df.columns:
            df[col] = df.groupby('stock_id')[col].transform(
                lambda x: _decay_sentiment(x)
            )

    df[all_sentiment_cols] = df[all_sentiment_cols].fillna(0)

    # === 7. 輸出摘要 ===
    _save_sentiment_summary(sent_market_us, sent_market_tw, sent_geopolitical,
                            len(stock_sentiments))

    print(f"✅ 情緒特徵整合完成！")
    print(f"   市場情緒 (US): {sent_market_us:+.3f}")
    print(f"   地緣政治情緒: {sent_geopolitical:+.3f}")
    print(f"   台股大盤情緒: {sent_market_tw:+.3f}")
    print(f"   個股新聞覆蓋: {len(stock_sentiments)} 支")

    return df


def _clear_gpu_cache():
    """釋放 GPU 記憶體"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _save_sentiment_summary(market_us: float, market_tw: float,
                             geopolitical: float, stock_count: int) -> None:
    """儲存情緒摘要 JSON，供 Streamlit 前端 sidebar 顯示"""
    import json
    from marketmamba.config import get_repo_output_dir, get_now_str

    summary = {
        "market_us": round(market_us, 4),
        "market_tw": round(market_tw, 4),
        "geopolitical": round(geopolitical, 4),
        "stock_coverage": stock_count,
        "updated_at": get_now_str(),
    }

    output_dir = get_repo_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'sentiment_summary.json')

    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 情緒摘要已儲存: {path}")
