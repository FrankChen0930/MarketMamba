import React from 'react';
import Dashboard from './Dashboard';
import ComingSoon from '../components/ComingSoon';

// 廣度模型「模型預測結果」= 既有 Dashboard（Alpha 排名 + 產業熱力圖）
// + 知識圖譜視覺化（計畫中優先項目，尚無前端可視化，先留版位）
export default function BreadthPredictions() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <ComingSoon
        icon="🕸️"
        title="知識圖譜視覺化"
        desc="互動式節點圖（產業 / 集團 / 供應鏈 / 相關性邊，~640K 條邊）。目前完全沒有前端可視化，是計畫中技術辨識度最高、優先要做的項目。"
        bullets={['資料來源 V6/marketmamba/knowledge/graph_builder.py 的 KG 快取，需後端新增匯出端點']}
      />
      <Dashboard />
    </div>
  );
}
