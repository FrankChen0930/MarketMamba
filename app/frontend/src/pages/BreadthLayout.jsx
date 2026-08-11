import React from 'react';
import { Outlet } from 'react-router-dom';
import { Compass, ChartLine, Package, History } from 'lucide-react';
import SectionSubNav from '../components/SectionSubNav';
import VersionBadge from '../components/VersionBadge';

const ICON = { size: 15, strokeWidth: 1.75 };

const SUB_TABS = [
  { to: '/breadth',             icon: <Compass {...ICON} />,   label: '總覽', end: true,
    hint: '這套模型在做什麼' },
  { to: '/breadth/predictions', icon: <ChartLine {...ICON} />, label: '每日評分',
    hint: '全市場每一檔今天拿到幾分' },
  // 「20 天」刻意寫進分頁名稱：這個模型每 20 個交易日才換一次股票，
  // 名稱不講清楚的話，「今天的清單」會被讀成「今天要買的清單」。
  { to: '/breadth/portfolio',   icon: <Package {...ICON} />,   label: '持股組合（20 天換一次）',
    hint: '目前持有哪 50 檔，以及上線後的實際表現' },
  { to: '/breadth/versions',    icon: <History {...ICON} />,   label: '版本紀錄',
    hint: '模型改過什麼、為什麼改' },
];

export default function BreadthLayout() {
  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
            廣度量化模型
            <VersionBadge state="live" />
          </div>
          <div className="page-subtitle">
            讓模型替全台股每一檔打分數，買進分數最高的 50 檔，每 20 個交易日換一次。
          </div>
        </div>
      </div>
      <SectionSubNav items={SUB_TABS} />
      <Outlet />
    </div>
  );
}
