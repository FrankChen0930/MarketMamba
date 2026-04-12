"""
MarketMamba V5.5 — FinBERT 英文情緒分析器
使用 ProsusAI/finbert 對英文金融新聞進行情緒量化
輸出：情緒分數 [-1, +1] 與 768→16 維 [CLS] embedding 投影
"""

import logging

import numpy as np
import torch

from marketmamba.config import FINBERT_EN_MODEL, FINBERT_EMBED_DIM

logger = logging.getLogger('MarketMamba.finbert_en')


class FinBERTEnglish:
    """英文金融情緒分析器 (ProsusAI/finbert)"""

    def __init__(self, device: str = None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"🔧 載入 FinBERT-EN: {FINBERT_EN_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_EN_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_EN_MODEL
        ).to(self.device).eval()

        # 768 → FINBERT_EMBED_DIM 投影層
        self.projection = torch.nn.Linear(768, FINBERT_EMBED_DIM).to(self.device)

        # ProsusAI/finbert: labels = ["positive", "negative", "neutral"]
        self.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

        logger.info(f"✅ FinBERT-EN 已載入 (設備: {self.device})")

    @torch.no_grad()
    def get_sentiment(self, titles: list[str]) -> list[float]:
        """
        批次計算情緒分數

        Returns:
            list[float]: 每條標題的情緒分數 [-1, +1]
                         = P(positive) - P(negative)
        """
        if not titles:
            return []

        inputs = self.tokenizer(
            titles, return_tensors='pt',
            truncation=True, padding=True, max_length=128
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        # sentiment = P(positive) - P(negative)
        scores = probs[:, 0] - probs[:, 1]
        return scores.tolist()

    @torch.no_grad()
    def get_embedding(self, titles: list[str]) -> np.ndarray:
        """
        提取 [CLS] token embedding 並投影至低維

        Returns:
            np.ndarray: (N, FINBERT_EMBED_DIM) 投影後的語義向量
        """
        if not titles:
            return np.zeros((0, FINBERT_EMBED_DIM))

        inputs = self.tokenizer(
            titles, return_tensors='pt',
            truncation=True, padding=True, max_length=128
        ).to(self.device)

        # 取得 hidden states
        outputs = self.model(**inputs, output_hidden_states=True)
        # [CLS] token 的最後一層 hidden state
        cls_embedding = outputs.hidden_states[-1][:, 0, :]  # (N, 768)

        # 投影至低維
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
