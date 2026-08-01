import React from 'react';
import { Outlet } from 'react-router-dom';
import SectionSubNav from '../components/SectionSubNav';

const USER_MODE = import.meta.env.VITE_USER_MODE || 'personal';

const SUB_TABS = [
  { to: '/conviction',             icon: '🧭', label: '總覽', end: true },
  { to: '/conviction/signals',     icon: '📋', label: '每日訊號' },
  { to: '/conviction/predictions', icon: '🗂️', label: '模型預測結果' },
  { to: '/conviction/backtest',    icon: '📉', label: '回測結果' },
  { to: '/conviction/versions',    icon: '🕰️', label: '版本紀錄' },
  ...(USER_MODE === 'personal'
    ? [{ to: '/conviction/portfolio', icon: '💼', label: '持倉追蹤' }]
    : []),
];

export default function ConvictionLayout() {
  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">🎯 高信念量化模型</div>
          <div className="page-subtitle">判斷驅動 · LLM 為工具層 · 個人實盤操作主軸</div>
        </div>
      </div>
      <SectionSubNav items={SUB_TABS} />
      <Outlet />
    </div>
  );
}
