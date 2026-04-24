"""
MarketMamba V5.5 — 中文情緒分析器 (多語言 DistilBERT)
使用 lxyuan/distilbert-base-multilingual-cased-sentiments-student
支援：繁體/簡體中文、英文、日文等多語言

優化：
  - 分批推論 + tqdm 進度條
  - FP16 混合精度 (GPU 加速 ~2x)
  - 合併 sentiment + embedding 為單次 forward pass
"""

import logging

import numpy as np
import torch
from tqdm.auto import tqdm

from marketmamba.config import FINBERT_CN_MODEL, FINBERT_EMBED_DIM

logger = logging.getLogger('MarketMamba.finbert_cn')


class FinBERTChinese:
    """
    多語言情緒分析器 (DistilBERT-based)

    模型標籤順序：positive / neutral / negative
    公開模型，不需要 HuggingFace Token
    """

    def __init__(self, device: str = None, batch_size: int = 64):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        logger.info(f"🔧 載入情緒模型: {FINBERT_CN_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_CN_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_CN_MODEL
        ).to(self.device).eval()

        # FP16 加速 (GPU only)
        self.use_fp16 = (self.device != 'cpu' and torch.cuda.is_available())
        if self.use_fp16:
            self.model = self.model.half()
            logger.info("  ⚡ FP16 混合精度已啟用")

        # 取得模型 hidden size
        if hasattr(self.model.config, 'dim'):
            hidden_size = self.model.config.dim  # DistilBERT
        elif hasattr(self.model.config, 'hidden_size'):
            hidden_size = self.model.config.hidden_size  # BERT
        else:
            hidden_size = 768

        # hidden → FINBERT_EMBED_DIM 投影層
        self.projection = torch.nn.Linear(hidden_size, FINBERT_EMBED_DIM).to(self.device)
        if self.use_fp16:
            self.projection = self.projection.half()

        # 讀取 label 順序
        self.label2id = self.model.config.label2id
        self.pos_idx = self.label2id.get('positive', 0)
        self.neg_idx = self.label2id.get('negative', 2)

        logger.info(f"✅ 情緒模型已載入 (設備: {self.device}, batch: {batch_size})")

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
        pbar = tqdm(batches, desc="  FinBERT-CN", disable=not show_progress,
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
            num_labels = probs.shape[1]
            if num_labels == 3:
                scores = probs[:, self.pos_idx] - probs[:, self.neg_idx]
            elif num_labels == 2:
                scores = probs[:, 1] - probs[:, 0]
            else:
                scores = np.zeros(len(batch))

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
        """批次計算情緒分數"""
        return self.analyze(titles, show_progress)["scores"]

    @torch.no_grad()
    def get_embedding(self, titles: list[str], show_progress: bool = False) -> np.ndarray:
        """提取 [CLS] embedding"""
        return self.analyze(titles, show_progress)["embeddings"]
