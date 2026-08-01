import React from 'react';
import ComingSoon from '../components/ComingSoon';

export default function ConvictionVersions() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <ComingSoon
        icon="🕰️"
        title="版本紀錄"
        desc="記錄兩條軌跡：篩選規則的變更（門檻調整、新增訊號）與 Agent pipeline 的演進（prompt 版本、單一 prompt → 多智能體的過渡、模型分層策略調整）。"
        bullets={[
          '目前篩選層沿用既有 scanner / signal_conditions（V6.2 進場評分 + 四層退場，見 CLAUDE.md 訊號系統章節）',
          'Agent pipeline 尚未開發（研究層 LLM thesis 生成屬 app/backend 範圍）',
        ]}
        note="規劃詳見 planing/雙模型架構重整計畫.md §2.5。"
      />
    </div>
  );
}
