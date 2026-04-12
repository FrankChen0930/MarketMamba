"""
MarketMamba V5.5 — 情緒特徵整合模組
負責：新聞爬取 → FinBERT 分析 → 特徵聚合 → Forward Fill + 指數衰減 → 併入主矩陣
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

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
    (使用 Numpy 加速運算，避免 Pandas iloc 的效能瓶頸)
    """
    decay_factor = 0.5 ** (1 / half_life)
    vals = series.values.copy()

    for i in range(1, len(vals)):
        # 若為 0 或 NaN 代表當日無新聞，進行衰減
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
    完整的情緒特徵計算管線

    步驟：
    1. 爬取中英文新聞 (若無預載資料)
    2. 雙軌 FinBERT 分析
    3. 按 (stock_id, Date) 聚合情緒分數與 embedding
    4. 地緣政治分類
    5. Forward Fill + 指數衰減
    6. 併入主矩陣

    Args:
        df: 主特徵矩陣
        ticker_name_map: {stock_id: stock_name} 映射 (用於爬取個股新聞)
        top_n_stocks: 只對 Top N 熱門股票爬取個股新聞

    Returns:
        加入情緒特徵後的 DataFrame
    """
    print("📰 啟動雙軌情緒引擎...")

    # 延遲匯入 (避免未安裝 transformers 時報錯)
    from marketmamba.sentiment.crawler_en import crawl_all_en_news
    from marketmamba.sentiment.crawler_cn import crawl_all_cn_news
    from marketmamba.sentiment.finbert_en import FinBERTEnglish
    from marketmamba.sentiment.finbert_cn import FinBERTChinese

    # === 1. 載入或爬取新聞 ===
    if precomputed_news:
        print("  ✅ 使用本機預載的新聞資料，跳過重複爬取...")
        en_news = {
            'market': precomputed_news.get('market_en', []),
            'geopolitical': precomputed_news.get('geopolitical', [])
        }
        cn_news = {
            'market_tw': precomputed_news.get('market_tw', []),
            'stocks': precomputed_news.get('stocks', {})
        }
    else:
        print("  🌐 正在即時爬取新聞...")
        en_news = crawl_all_en_news()

        # 準備個股新聞爬取清單
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

    # === 2. FinBERT 分析 ===
    print("  🧠 FinBERT-EN 情緒分析...")
    finbert_en = FinBERTEnglish()

    # 全球市場情緒
    market_en_titles = [a['title'] for a in en_news.get('market', [])]
    market_en_scores = finbert_en.get_sentiment(market_en_titles) if market_en_titles else []
    market_en_embed = finbert_en.get_embedding(market_en_titles) if market_en_titles else np.zeros((0, FINBERT_EMBED_DIM))

    # 地緣政治情緒
    geo_titles = [a['title'] for a in en_news.get('geopolitical', [])]
    geo_scores = finbert_en.get_sentiment(geo_titles) if geo_titles else []

    print("  🧠 FinBERT-CN 情緒分析...")
    finbert_cn = FinBERTChinese()

    # 台股大盤情緒
    market_tw_titles = [a['title'] for a in cn_news.get('market_tw', [])]
    market_tw_scores = finbert_cn.get_sentiment(market_tw_titles) if market_tw_titles else []

    # === 3. 聚合為特徵 ===
    latest_date = df['Date'].max()

    # 全市場層級特徵 (所有股票共用)
    sent_market_us = float(np.mean(market_en_scores)) if market_en_scores else 0.0
    sent_geopolitical = float(np.mean(geo_scores)) if geo_scores else 0.0
    sent_market_tw = float(np.mean(market_tw_scores)) if market_tw_scores else 0.0

    # 全市場 Embedding 均值
    market_embed_en = market_en_embed.mean(axis=0) if len(market_en_embed) > 0 else np.zeros(FINBERT_EMBED_DIM)

    # === 4. 個股層級特徵 ===
    stock_sentiments = {}  # {stock_id: {"cn_score": float, "en_score": float, "volume": int, "embed_cn": ndarray}}

    for stock_id, articles in cn_news.get('stocks', {}).items():
        if articles:
            titles = [a['title'] for a in articles]
            scores = finbert_cn.get_sentiment(titles)
            embeds = finbert_cn.get_embedding(titles)

            stock_sentiments[stock_id] = {
                'cn_score': float(np.mean(scores)),
                'news_count': len(articles),
                'embed_cn': embeds.mean(axis=0),
            }

    # === 5. 併入 DataFrame ===
    # 初始化情緒欄位
    for col in SENTIMENT_SCALAR_COLS:
        if col not in df.columns:
            df[col] = 0.0
    for col in SENTIMENT_EMBED_EN_COLS + SENTIMENT_EMBED_CN_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # 只在最新日期填入情緒值 (歷史日用衰減填補)
    mask = df['Date'] == latest_date

    df.loc[mask, 'Sent_Market_US'] = sent_market_us
    df.loc[mask, 'Sent_Geopolitical'] = sent_geopolitical
    df.loc[mask, 'Sent_Market_TW'] = sent_market_tw

    # 市場 EN Embedding
    for i, col in enumerate(SENTIMENT_EMBED_EN_COLS):
        df.loc[mask, col] = market_embed_en[i]

    # 個股層級
    for stock_id, sent_data in stock_sentiments.items():
        stock_mask = mask & (df['stock_id'].astype(str) == stock_id)
        df.loc[stock_mask, 'Sent_Stock_CN'] = sent_data['cn_score']
        df.loc[stock_mask, 'News_Volume_Stock'] = np.log1p(sent_data['news_count'])

        # CN Embedding
        for i, col in enumerate(SENTIMENT_EMBED_CN_COLS):
            df.loc[stock_mask, col] = sent_data['embed_cn'][i]

    # === 6. Forward Fill + 指數衰減 ===
    for col in SENTIMENT_SCALAR_COLS:
        if col in df.columns:
            df[col] = df.groupby('stock_id')[col].transform(
                lambda x: _decay_sentiment(x)
            )

    # Embedding 欄位也做衰減
    for col in SENTIMENT_EMBED_EN_COLS + SENTIMENT_EMBED_CN_COLS:
        if col in df.columns:
            df[col] = df.groupby('stock_id')[col].transform(
                lambda x: _decay_sentiment(x)
            )

    # 最終填 0
    sentiment_cols = SENTIMENT_SCALAR_COLS + SENTIMENT_EMBED_EN_COLS + SENTIMENT_EMBED_CN_COLS
    df[sentiment_cols] = df[sentiment_cols].fillna(0)

    # 清理 FinBERT 模型釋放 GPU
    del finbert_en, finbert_cn
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # === 7. 輸出情緒摘要 JSON (供 Streamlit 前端顯示) ===
    _save_sentiment_summary(sent_market_us, sent_market_tw, sent_geopolitical,
                            len(stock_sentiments))

    print(f"✅ 情緒特徵整合完成！")
    print(f"   市場情緒 (US): {sent_market_us:.3f}")
    print(f"   地緣政治情緒: {sent_geopolitical:.3f}")
    print(f"   台股大盤情緒: {sent_market_tw:.3f}")
    print(f"   個股新聞覆蓋: {len(stock_sentiments)} 支")

    return df


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

