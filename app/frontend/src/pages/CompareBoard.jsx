import React from 'react';
import { GitCompare, Percent, Search, Map } from 'lucide-react';
import ComingSoon from '../components/ComingSoon';

const ICON = { size: 16, strokeWidth: 1.75 };

// 模型分歧看板 —— 純研究用途，不做自動合併邏輯。
// 目前還沒東西可看，所以刻意不放進主導覽（路由保留，網址仍可直接開）。
export default function CompareBoard() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <div className="page-header">
        <div>
          <div className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <GitCompare size={18} strokeWidth={1.75} aria-hidden="true" />
            兩條線的分歧
          </div>
          <div className="page-subtitle">
            買 50 檔那一套和只買 10 檔那一套，什麼時候看法一致、什麼時候完全相反
          </div>
        </div>
      </div>

      <ComingSoon
        icon={<Percent {...ICON} />}
        title="重疊了幾檔"
        desc="高信念那條線最看好的股票，有幾檔同時出現在廣度模型的前 50 名。重疊越低，代表兩套方法看到的東西越不一樣——那才有互相補位的價值。"
      />
      <ComingSoon
        icon={<GitCompare {...ICON} />}
        title="看法有多接近"
        desc="用近 20 天和近 60 天的資料算兩邊的相關程度。這件事有個前置問題要先解決：廣度模型給的是連續分數，高信念那邊只有「很看好／看好／普通」三級，兩者要先換算到同一把尺才比得了。"
      />
      <ComingSoon
        icon={<Search {...ICON} />}
        title="吵起來的日子"
        desc="列出兩邊明顯不同調的日期和股票，旁邊留一欄手動填原因。分歧本身不是問題，看不出為什麼分歧才是。"
      />
      <ComingSoon
        icon={<Map {...ICON} />}
        title="之後打算怎麼合"
        desc="這裡只放想法，不做自動合併。兩套方法各有各的前提，用程式硬湊出一個「綜合分數」只會把兩邊的優點都稀釋掉。"
        note="要等兩條線各自都有穩定的每日產出，這頁才有東西可以算。目前只有廣度模型那一邊在跑。"
      />
    </div>
  );
}
