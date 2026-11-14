import React from 'react';
import { Outlet } from 'react-router-dom';
import { Compass, FileText, ChartLine, History, Wallet } from 'lucide-react';
import SectionSubNav from '../components/SectionSubNav';
import VersionBadge from '../components/VersionBadge';

const USER_MODE = import.meta.env.VITE_USER_MODE || 'personal';

const ICON = { size: 15, strokeWidth: 1.75 };

const SUB_TABS = [
  { to: '/conviction',             icon: <Compass {...ICON} />,   label: '總覽', end: true,
    hint: '這條線打算怎麼做' },
  { to: '/conviction/predictions', icon: <FileText {...ICON} />,  label: '個股研究',
    hint: '每檔一張卡：看好的理由、風險、什麼情況代表看錯了' },
  { to: '/conviction/backtest',    icon: <ChartLine {...ICON} />, label: '回測結果',
    hint: '哪些部分驗證得動、哪些只能往前追蹤' },
  { to: '/conviction/versions',    icon: <History {...ICON} />,   label: '版本紀錄',
    hint: '篩選規則和流程改過什麼' },
  ...(USER_MODE === 'personal'
    ? [{ to: '/conviction/portfolio', icon: <Wallet {...ICON} />, label: '持倉追蹤',
         hint: '個人帳戶部位，主要在 PersonalOS 桌面版使用' }]
    : []),
];

export default function ConvictionLayout() {
  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
            高信念量化模型
            <VersionBadge state="planned" showVersion={false} />
          </div>
          <div className="page-subtitle">
            另一條路：不押 50 檔，只挑 10 到 20 檔，但每一檔都要說得出理由。
          </div>
        </div>
      </div>
      <SectionSubNav items={SUB_TABS} />
      <Outlet />
    </div>
  );
}
