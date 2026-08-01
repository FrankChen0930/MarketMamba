import React from 'react';
import ComingSoon from '../components/ComingSoon';

// 模型分歧看板（新增，未來整合的地基）—— 純研究用途，不做自動合併邏輯
export default function CompareBoard() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header">
        <div>
          <div className="page-title">🔀 模型分歧看板</div>
          <div className="page-subtitle">高信念模型 vs 廣度模型 —— 未來整合的地基，純研究用途</div>
        </div>
      </div>

      <ComingSoon
        icon="📐"
        title="兩模型選股重疊率"
        desc="高信念模型的 S/A 級標的，有幾檔同時出現在廣度模型 Top 50。"
      />
      <ComingSoon
        icon="📈"
        title="兩模型信號相關係數"
        desc="近 20 / 60 日滾動相關係數。高信念線目前只有離散信念分級（S/A/B），需要先轉換成可比較的量尺才能與廣度模型的連續分數對比。"
      />
      <ComingSoon
        icon="🔍"
        title="分歧時期標記"
        desc="列出兩模型明顯不一致的日期與標的，附原因推測欄位（先留空手動填）。"
      />
      <ComingSoon
        icon="🗺️"
        title="未來整合路線圖"
        desc="純研究用途 TODO 區塊，不做自動合併邏輯。"
        note="規劃詳見 planing/雙模型架構重整計畫.md §4。此頁需要兩條線各自產出結構化資料（信念分級、Alpha 排名時序）後才能接上真實運算，目前僅為版位。"
      />
    </div>
  );
}
