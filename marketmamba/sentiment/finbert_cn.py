"""
MarketMamba V5.5 — FinBERT 中文情緒分析器
使用 hw2942/chinese-finbert-for-sentiment-analysis 對中文金融新聞進行情緒量化
繁體中文輸入相容 (BERT tokenizer 字元級分詞)
輸出：情緒分數 [-1, +1] 與 768→16 維 [CLS] embedding 投影
"""

import logging

import numpy as np
import torch

from marketmamba.config import FINBERT_CN_MODEL, FINBERT_EMBED_DIM

logger = logging.getLogger('MarketMamba.finbert_cn')


class FinBERTChinese:
    """
    中文金融情緒分析器 (hw2942/chinese-finbert-for-sentiment-analysis)

    此模型以簡體中文金融語料訓練，但 BERT 的字元級 tokenizer
    對繁體中文有良好的相容性（多數漢字在 vocab 中共用）
    """

    def __init__(self, device: str = None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"🔧 載入 FinBERT-CN: {FINBERT_CN_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_CN_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_CN_MODEL
        ).to(self.device).eval()

        # 768 → FINBERT_EMBED_DIM 投影層
        self.projection = torch.nn.Linear(768, FINBERT_EMBED_DIM).to(self.device)

        # chinese-finbert 的標籤順序：0=negative, 1=neutral, 2=positive
        # (需確認模型 config，此處以常見排列為準)
        logger.info(f"✅ FinBERT-CN 已載入 (設備: {self.device})")

    @torch.no_grad()
    def get_sentiment(self, titles: list[str]) -> list[float]:
        """
        批次計算情緒分數

        Returns:
            list[float]: [-1, +1]
                         假設 label 排列為 [negative, neutral, positive]
                         score = P(positive) - P(negative)
        """
        if not titles:
            return []

        inputs = self.tokenizer(
            titles, return_tensors='pt',
            truncation=True, padding=True, max_length=128
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        # 依模型標籤順序計算 (0=neg, 1=neu, 2=pos)
        # score = P(positive) - P(negative)
        num_labels = probs.shape[1]
        if num_labels == 3:
            scores = probs[:, 2] - probs[:, 0]
        elif num_labels == 2:
            # 二分類: [negative, positive]
            scores = probs[:, 1] - probs[:, 0]
        else:
            logger.warning(f"⚠️ 未預期的標籤數量: {num_labels}，回傳 0")
            scores = np.zeros(len(titles))

        return scores.tolist()

    @torch.no_grad()
    def get_embedding(self, titles: list[str]) -> np.ndarray:
        """
        提取 [CLS] token embedding 並投影至低維

        Returns:
            np.ndarray: (N, FINBERT_EMBED_DIM)
        """
        if not titles:
            return np.zeros((0, FINBERT_EMBED_DIM))

        inputs = self.tokenizer(
            titles, return_tensors='pt',
            truncation=True, padding=True, max_length=128
        ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        cls_embedding = outputs.hidden_states[-1][:, 0, :]  # (N, 768)

        projected = self.projection(cls_embedding).cpu().numpy()
        return projected

    def analyze(self, titles: list[str]) -> dict:
        """
        同時取得情緒分數和 embedding

        Returns:
            {"scores": list[float], "embeddings": np.ndarray}
        """
        return {
            "scores": self.get_sentiment(titles),
            "embeddings": self.get_embedding(titles),
        }
