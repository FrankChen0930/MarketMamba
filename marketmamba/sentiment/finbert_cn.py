"""
MarketMamba V5.5 — 中文情緒分析器 (多語言 DistilBERT)
使用 lxyuan/distilbert-base-multilingual-cased-sentiments-student
支援：繁體/簡體中文、英文、日文等多語言

輸出：情緒分數 [-1, +1] 與 768→16 維 [CLS] embedding 投影
"""

import logging

import numpy as np
import torch

from marketmamba.config import FINBERT_CN_MODEL, FINBERT_EMBED_DIM

logger = logging.getLogger('MarketMamba.finbert_cn')


class FinBERTChinese:
    """
    多語言情緒分析器 (DistilBERT-based)

    模型標籤順序：positive / neutral / negative
    比原版 chinese-finbert 更穩定且不需要 HuggingFace Token
    """

    def __init__(self, device: str = None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"🔧 載入情緒模型: {FINBERT_CN_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_CN_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_CN_MODEL
        ).to(self.device).eval()

        # 取得模型 hidden size (DistilBERT = 768)
        if hasattr(self.model.config, 'dim'):
            hidden_size = self.model.config.dim  # DistilBERT
        elif hasattr(self.model.config, 'hidden_size'):
            hidden_size = self.model.config.hidden_size  # BERT
        else:
            hidden_size = 768

        # hidden → FINBERT_EMBED_DIM 投影層
        self.projection = torch.nn.Linear(hidden_size, FINBERT_EMBED_DIM).to(self.device)

        # 讀取 label 順序
        self.label2id = self.model.config.label2id
        logger.info(f"✅ 情緒模型已載入 (設備: {self.device}, 標籤: {self.label2id})")

    @torch.no_grad()
    def get_sentiment(self, titles: list[str]) -> list[float]:
        """
        批次計算情緒分數

        Returns:
            list[float]: [-1, +1]
                         score = P(positive) - P(negative)
        """
        if not titles:
            return []

        # 分批處理避免 OOM (每批 32 筆)
        all_scores = []
        batch_size = 32

        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]

            inputs = self.tokenizer(
                batch, return_tensors='pt',
                truncation=True, padding=True, max_length=128
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            num_labels = probs.shape[1]
            if num_labels == 3:
                # 自動偵測 positive/negative 的 index
                pos_idx = self.label2id.get('positive', 0)
                neg_idx = self.label2id.get('negative', 2)
                scores = probs[:, pos_idx] - probs[:, neg_idx]
            elif num_labels == 2:
                scores = probs[:, 1] - probs[:, 0]
            else:
                logger.warning(f"⚠️ 未預期的標籤數量: {num_labels}，回傳 0")
                scores = np.zeros(len(batch))

            all_scores.extend(scores.tolist())

        return all_scores

    @torch.no_grad()
    def get_embedding(self, titles: list[str]) -> np.ndarray:
        """
        提取 [CLS] token embedding 並投影至低維

        Returns:
            np.ndarray: (N, FINBERT_EMBED_DIM)
        """
        if not titles:
            return np.zeros((0, FINBERT_EMBED_DIM))

        all_embeds = []
        batch_size = 32

        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]

            inputs = self.tokenizer(
                batch, return_tensors='pt',
                truncation=True, padding=True, max_length=128
            ).to(self.device)

            outputs = self.model(**inputs, output_hidden_states=True)

            # DistilBERT: hidden_states 是 tuple，取最後一層的 [CLS]
            cls_embedding = outputs.hidden_states[-1][:, 0, :]  # (batch, hidden_size)

            projected = self.projection(cls_embedding).cpu().numpy()
            all_embeds.append(projected)

        return np.concatenate(all_embeds, axis=0)

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
