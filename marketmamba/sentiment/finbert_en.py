"""
MarketMamba V5.5 — FinBERT 英文情緒分析器
使用 ProsusAI/finbert 對英文金融新聞進行情緒量化
輸出：情緒分數 [-1, +1] 與 768→16 維 [CLS] embedding 投影

優化：
  - 分批推論 + tqdm 進度條
  - FP16 混合精度 (GPU 加速 ~2x)
  - 合併 sentiment + embedding 為單次 forward pass
"""

import logging

import numpy as np
import torch
from tqdm.auto import tqdm

from marketmamba.config import FINBERT_EN_MODEL, FINBERT_EMBED_DIM

logger = logging.getLogger('MarketMamba.finbert_en')


class FinBERTEnglish:
    """英文金融情緒分析器 (ProsusAI/finbert)"""

    def __init__(self, device: str = None, batch_size: int = 64):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        logger.info(f"🔧 載入 FinBERT-EN: {FINBERT_EN_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_EN_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_EN_MODEL
        ).to(self.device).eval()

        # FP16 加速 (GPU only)
        self.use_fp16 = (self.device != 'cpu' and torch.cuda.is_available())
        if self.use_fp16:
            self.model = self.model.half()
            logger.info("  ⚡ FP16 混合精度已啟用")

        # 768 → FINBERT_EMBED_DIM 投影層
        self.projection = torch.nn.Linear(768, FINBERT_EMBED_DIM).to(self.device)
        if self.use_fp16:
            self.projection = self.projection.half()

        # ProsusAI/finbert: labels = ["positive", "negative", "neutral"]
        self.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

        logger.info(f"✅ FinBERT-EN 已載入 (設備: {self.device}, batch: {batch_size})")

    @torch.no_grad()
    def analyze(self, titles: list[str], show_progress: bool = True) -> dict:
        """
        一次 forward pass 同時取得情緒分數和 embedding

        Returns:
            {"scores": list[float], "embeddings": np.ndarray}
        """
        if not titles:
            return {"scores": [], "embeddings": np.zeros((0, FINBERT_EMBED_DIM))}

        all_scores = []
        all_embeds = []

        batches = range(0, len(titles), self.batch_size)
        pbar = tqdm(batches, desc="  FinBERT-EN", disable=not show_progress,
                    total=len(batches), unit="batch")

        for i in pbar:
            batch = titles[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch, return_tensors='pt',
                truncation=True, padding=True, max_length=128
            ).to(self.device)

            # 單次 forward 同時取 logits + hidden states
            outputs = self.model(**inputs, output_hidden_states=True)

            # 情緒分數
            probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
            scores = probs[:, 0] - probs[:, 1]  # P(pos) - P(neg)
            all_scores.extend(scores.tolist())

            # [CLS] embedding → 投影
            cls_emb = outputs.hidden_states[-1][:, 0, :]
            projected = self.projection(cls_emb).float().cpu().numpy()
            all_embeds.append(projected)

            pbar.set_postfix({"articles": min(i + self.batch_size, len(titles))})

        return {
            "scores": all_scores,
            "embeddings": np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0, FINBERT_EMBED_DIM)),
        }

    @torch.no_grad()
    def get_sentiment(self, titles: list[str], show_progress: bool = True) -> list[float]:
        """批次計算情緒分數 (呼叫 analyze 的便捷方法)"""
        return self.analyze(titles, show_progress)["scores"]

    @torch.no_grad()
    def get_embedding(self, titles: list[str], show_progress: bool = False) -> np.ndarray:
        """提取 [CLS] embedding (呼叫 analyze 的便捷方法)"""
        return self.analyze(titles, show_progress)["embeddings"]
