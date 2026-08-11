import React from 'react';
import { Outlet } from 'react-router-dom';
import { Compass, ClipboardList, Gauge, GitCompare, TriangleAlert } from 'lucide-react';
import SectionSubNav from '../components/SectionSubNav';
import VersionBadge from '../components/VersionBadge';

const ICON = { size: 15, strokeWidth: 1.75 };

const SUB_TABS = [
  { to: '/legacy',         icon: <Compass {...ICON} />,       label: '總覽', end: true,
    hint: '這一版做了什麼、為什麼換掉' },
  { to: '/legacy/signals', icon: <ClipboardList {...ICON} />, label: '每日訊號',
    hint: '每天挑出值得買的股票，附進場評分' },
  { to: '/legacy/sim',     icon: <Gauge {...ICON} />,         label: '模擬操作',
    hint: '照這套規則買賣，帳面上會變成什麼樣子' },
  { to: '/legacy/dual',    icon: <GitCompare {...ICON} />,    label: '雙模型比較',
    hint: '短線版與趨勢版選出來的股票差在哪' },
];

export default function LegacyLayout() {
  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
            前一版（V6.1）
            <VersionBadge state="legacy" />
          </div>
          <div className="page-subtitle">
            2026 年 8 月之前每天在跑的版本。留著是為了對照，不是還在用。
          </div>
        </div>
      </div>

      {/* 凍結公告：進到這個區段的每一頁都看得到，避免有人把舊數字當成今天的建議 */}
      <div
        role="note"
        style={{
          display: 'flex', gap: 'var(--space-3)', alignItems: 'flex-start',
          padding: 'var(--space-3) var(--space-4)',
          marginBottom: 'var(--space-5)',
          background: 'rgba(255,165,0,0.06)',
          borderLeft: '3px solid var(--ver-legacy)',
          borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
        }}
      >
        <TriangleAlert size={16} strokeWidth={2} color="var(--ver-legacy)" style={{ flexShrink: 0, marginTop: 2 }} aria-hidden="true" />
        <p style={{ margin: 0, fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
          <b style={{ color: 'var(--text-primary)' }}>這一區的內容已經停止更新。</b>
          {' '}底下的數字可能是好幾天前算的，別拿來當今天的進出場依據。
          現在實際在跑的是 <a href="/breadth" style={{ color: 'var(--accent-blue)' }}>廣度模型（V6.2）</a>。
        </p>
      </div>

      <SectionSubNav items={SUB_TABS} />
      <Outlet />
    </div>
  );
}
