import React from 'react';
import { Network } from 'lucide-react';
import Dashboard from './Dashboard';
import ComingSoon from '../components/ComingSoon';

// 廣度模型「每日評分」= 既有 Dashboard（分數排行 + 產業熱力圖）
// + 知識圖譜視覺化（還沒有前端可視化，先留版位）
export default function BreadthPredictions() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <ComingSoon
        icon={<Network size={16} strokeWidth={1.75} />}
        title="股票關聯圖"
        desc="模型在打分數的時候，會參考同產業、同集團、有供應鏈往來的其他股票——目前這些關聯有 64 萬條，但只存在後端，畫面上看不到。之後想做成可以拖曳查看的互動圖。"
        bullets={['需要先把後端的關聯資料整理成可以傳給前端的格式']}
      />
      <Dashboard />
    </div>
  );
}
