import React from 'react';
import { Outlet } from 'react-router-dom';
import SectionSubNav from '../components/SectionSubNav';

const SUB_TABS = [
  { to: '/breadth',             icon: '🧭', label: '總覽', end: true },
  { to: '/breadth/predictions', icon: '📈', label: '模型預測結果' },
  // 「20 日」刻意寫進分頁名稱：這個模型每 20 個交易日才換一次股，
  // 名稱不講清楚的話「今天的清單」會被讀成「今天要買的清單」。
  { to: '/breadth/portfolio',   icon: '📦', label: '持股組合（20 日）' },
  { to: '/breadth/backtest',    icon: '📊', label: '回測結果' },
  { to: '/breadth/versions',    icon: '🕰️', label: '版本紀錄' },
  { to: '/breadth/compare',     icon: '🔀', label: '雙模型比較' },
];

export default function BreadthLayout() {
  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">🌐 廣度量化模型</div>
          <div className="page-subtitle">統計學習驅動 · DL 為核心引擎 · 作品展示主軸</div>
        </div>
      </div>
      <SectionSubNav items={SUB_TABS} />
      <Outlet />
    </div>
  );
}
