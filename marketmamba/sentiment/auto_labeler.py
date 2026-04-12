"""
MarketMamba V5.5 — 自動標籤產生器 (Auto-Labeler)
用股價反應自動標註新聞情緒，用於微調 FinBERT

策略：看新聞發布後 N 天的 Alpha 變化來判斷新聞好壞
避免主觀偏誤，以市場實際反應為唯一標準
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger('MarketMamba.auto_labeler')


def generate_sentiment_labels(df_master: pd.DataFrame,
                              news_records: list[dict],
                              forward_window: int = 5,
                              positive_threshold: float = 0.01,
                              negative_threshold: float = -0.01) -> pd.DataFrame:
    """
    股價反應自動標籤產生器

    流程：
    1. 對每條新聞，找到對應股票在 pub_date 後 forward_window 天的累積 Alpha
    2. Alpha > positive_threshold → label = 1 (positive)
       Alpha < negative_threshold → label = -1 (negative)
       else → label = 0 (neutral)

    Args:
        df_master: 主特徵矩陣 (需含 stock_id, Date, Alpha_1d)
        news_records: 新聞列表，每條需含:
                      {"stock_id": str, "pub_date": str, "title": str}
        forward_window: 看未來幾天的累積 Alpha
        positive_threshold: 正面標籤門檻
        negative_threshold: 負面標籤門檻

    Returns:
        帶標籤的新聞 DataFrame，可用於微調 FinBERT
        columns: [stock_id, pub_date, title, cumulative_alpha, label, label_text]
    """
    if not news_records:
        logger.warning("⚠️ 空的新聞列表，無法產生標籤")
        return pd.DataFrame()

    logger.info(f"🏷️ 開始產生自動標籤 (forward_window={forward_window}d)...")

    df_master['Date'] = pd.to_datetime(df_master['Date'])
    df_master = df_master.sort_values(['stock_id', 'Date'])

    results = []

    for record in news_records:
        stock_id = str(record.get('stock_id', ''))
        pub_date = pd.to_datetime(record.get('pub_date', ''))
        title = record.get('title', '')

        if not stock_id or pd.isna(pub_date):
            continue

        # 找到該股票在 pub_date 之後的資料
        stock_data = df_master[
            (df_master['stock_id'].astype(str) == stock_id) &
            (df_master['Date'] > pub_date)
        ].head(forward_window)

        if len(stock_data) < forward_window:
            continue  # 資料不足，跳過

        # 計算累積 Alpha
        cumulative_alpha = stock_data['Alpha_1d'].sum()

        # 標籤判定
        if cumulative_alpha > positive_threshold:
            label = 1
            label_text = 'positive'
        elif cumulative_alpha < negative_threshold:
            label = -1
            label_text = 'negative'
        else:
            label = 0
            label_text = 'neutral'

        results.append({
            'stock_id': stock_id,
            'pub_date': pub_date.strftime('%Y-%m-%d'),
            'title': title,
            'cumulative_alpha': round(cumulative_alpha, 6),
            'label': label,
            'label_text': label_text,
        })

    df_labels = pd.DataFrame(results)

    if not df_labels.empty:
        label_dist = df_labels['label_text'].value_counts()
        logger.info(f"🏷️ 標籤分佈: {label_dist.to_dict()}")
        logger.info(f"   共 {len(df_labels)} 條標註完成")
    else:
        logger.warning("⚠️ 無法產生任何標籤 (資料不足)")

    return df_labels


def finetune_finbert(base_model_name: str,
                     labeled_df: pd.DataFrame,
                     output_dir: str,
                     epochs: int = 3,
                     batch_size: int = 16,
                     lr: float = 2e-5) -> None:
    """
    用 Auto-Label 結果微調 FinBERT

    需要累積 ≥ 1000 條標註資料後再執行

    Args:
        base_model_name: HuggingFace 模型名稱
        labeled_df: generate_sentiment_labels 的產出
        output_dir: 微調後模型的儲存路徑
        epochs: 訓練 epoch 數
        batch_size: 批次大小
        lr: 學習率
    """
    if len(labeled_df) < 100:
        logger.warning(f"⚠️ 目前只有 {len(labeled_df)} 條標註，建議累積至 1000+ 條再微調")
        return

    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer
    )
    from torch.utils.data import Dataset
    import torch

    logger.info(f"🔧 開始微調 {base_model_name} (共 {len(labeled_df)} 條資料)...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=3
    )

    # 標籤映射: -1→0, 0→1, 1→2
    label_map = {-1: 0, 0: 1, 1: 2}

    class SentimentDataset(Dataset):
        def __init__(self, df, tokenizer):
            self.encodings = tokenizer(
                df['title'].tolist(), truncation=True,
                padding=True, max_length=128, return_tensors='pt'
            )
            self.labels = torch.tensor(
                [label_map[l] for l in df['label'].tolist()]
            )

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

    dataset = SentimentDataset(labeled_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy='epoch',
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✅ 微調完成！模型已儲存至: {output_dir}")
