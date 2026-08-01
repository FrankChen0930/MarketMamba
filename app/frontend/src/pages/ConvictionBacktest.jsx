import React from 'react';
import ComingSoon from '../components/ComingSoon';

export default function ConvictionBacktest() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <ComingSoon
        icon="📉"
        title="回測結果"
        desc="誠實拆兩塊呈現：有足夠歷史事件數、結構化數據的候選子策略（財報意外後動能延續、法人籌碼轉向反轉、除權息規律）可以真的做 Walk-Forward；其餘只能標示「即時追蹤中」搭配研究日誌。"
        bullets={[
          '方向三（Conviction 萃取實驗）目前狀態：使用者已決定暫停模型實驗，等真倉先驗證有沒有賺錢再繼續（見 CLAUDE.md）',
          '候選子方案：事件驅動 / Meta-labeling / 不確定性驅動集中（優先度最高，可重用現有 MC-Dropout Uncertainty 與 Signal_Quality）',
          '研究日誌（校準工具）規劃存 Supabase，尚未建表',
        ]}
        note="規劃詳見 planing/雙模型架構重整計畫.md §2.4、§2.7；研究進度見 planing/研究計畫_方向三_Conviction萃取實驗.md。"
      />
    </div>
  );
}
