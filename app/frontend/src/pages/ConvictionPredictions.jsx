import React from 'react';
import ComingSoon from '../components/ComingSoon';

export default function ConvictionPredictions() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <ComingSoon
        icon="🗂️"
        title="模型預測結果 — Thesis 卡片牆"
        desc="每檔候選標的一張卡片：thesis 摘要、催化劑、估值判讀、風險檢查、DL 輔助訊號（小字附註）、證偽條件。型態學雙重確認結果會併入卡片作為技術面佐證。"
        bullets={[
          '需要研究層（LLM 結構化 prompt）與整合層（人工信念分級 S/A/B）的後端支援，屬於 app/backend 範圍',
          '此次僅完成頁面版位與導覽，尚未串接真實 thesis 資料',
        ]}
        note="規劃詳見 planing/雙模型架構重整計畫.md §2.1 研究層 / §2.3。"
      />
    </div>
  );
}
