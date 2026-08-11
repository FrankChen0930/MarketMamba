import React, { useEffect, useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { House, Globe, Target, History, ChartLine, Bot, Microscope } from 'lucide-react';
import { fetchMarket, fetchTicker } from '../api/market';
import { versionOf } from '../versions';

// 圖示用 lucide SVG（不用 emoji）：emoji 在不同作業系統長得不一樣，
// 也沒辦法跟著 hover / active 換色。窄螢幕會把文字藏起來只剩圖示，
// 所以每個分頁都要帶 hint 當作 aria-label 之外的說明。
const ICON = { size: 15, strokeWidth: 1.75 };

const TABS = [
  { to: '/',           icon: <House {...ICON} />,     label: '首頁',       hint: '總覽與今日重點' },
  { to: '/breadth',    icon: <Globe {...ICON} />,     label: '廣度模型',   hint: 'V6.2 · 目前上線中的版本' },
  { to: '/conviction', icon: <Target {...ICON} />,    label: '高信念模型', hint: '少數幾檔、深入研究的那條線' },
  { to: '/legacy',     icon: <History {...ICON} />,   label: 'V6.1 前一版', hint: '已凍結的舊版，保留對照用' },
  { to: '/quant',      icon: <ChartLine {...ICON} />, label: '量化分析',   hint: '技術型態與產業強弱' },
  { to: '/market',     icon: <Bot {...ICON} />,       label: 'AI 日報',    hint: '每天一篇的市場解讀' },
  { to: '/research',   icon: <Microscope {...ICON} />, label: '研究紀錄',  hint: '做過的實驗與結論' },
];


// Fallback ticker items shown while API loads
const FALLBACK_TICKER = [
  { id: 'TAIEX', name: '加權',  price: '—', change: '—', pct: '—', up: true },
  { id: '2330',  name: '台積電', price: '—', change: '—', pct: '—', up: true },
  { id: '2454',  name: '聯發科', price: '—', change: '—', pct: '—', up: true },
];

function TickerBar() {
  const [items, setItems] = useState(FALLBACK_TICKER);

  useEffect(() => {
    fetchTicker()
      .then((res) => setItems(res.items || []))
      .catch(() => {}); // silently fall back
  }, []);

  const doubled = [...items, ...items];

  return (
    <div className="ticker-bar">
      <div className="ticker-track">
        {doubled.map((t, i) => (
          <div key={i} className="ticker-item">
            <span className="ti-name">{t.name}</span>
            <span className="ti-price">{t.price}</span>
            <span className={`ti-change ${t.up ? 'text-positive' : 'text-negative'}`}>{t.pct}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AppLayout() {
  const [market, setMarket] = useState(null);

  useEffect(() => {
    fetchMarket().then(setMarket).catch(() => {});
  }, []);

  return (
    <div className="app-shell">
      {/* ── Top Nav ── */}
      <nav className="topbar">
        <a href="/" className="topbar-brand" style={{ textDecoration: 'none' }}>
          <div className="logo-mark">M</div>
          <div className="brand-text">
            <span className="brand-name">MarketMamba</span>
            <span className="brand-sub">{versionOf('live')} · 台股每日更新</span>
          </div>
        </a>

        <div className="nav-tabs" role="navigation" aria-label="主導覽">
          {TABS.map(tab => (
            <NavLink
              key={tab.to} to={tab.to} end={tab.to === '/'}
              aria-label={tab.label}
              title={`${tab.label} — ${tab.hint}`}
              className={({ isActive }) => `nav-tab${isActive ? ' active' : ''}`}
            >
              <span className="tab-icon" aria-hidden="true">{tab.icon}</span>
              <span>{tab.label}</span>
            </NavLink>
          ))}
        </div>

        <div className="topbar-right">
          <div className="topbar-stat">
            <span className="ts-label">VIX</span>
            <span className={`ts-value mono ${(market?.vix || 0) > 20 ? 'text-negative' : 'text-positive'}`}>
              {market?.vix ? market.vix.toFixed(2) : '—'}
            </span>
          </div>
          <div className="topbar-stat">
            <span className="ts-label">USD/TWD</span>
            <span className="ts-value mono">
              {market?.usd_twd ? market.usd_twd.toFixed(3) : '—'}
            </span>
          </div>
          <div className="topbar-stat">
            <span className="ts-label">漲/跌</span>
            <span className="ts-value mono">
              <span className="text-positive">{market?.advancing ?? '—'}</span>
              <span style={{ color: 'var(--text-muted)' }}>/</span>
              <span className="text-negative">{market?.declining ?? '—'}</span>
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div className="status-dot" style={{
              background: market?.run_status === 'completed' ? 'var(--positive)' : 'var(--accent-amber)',
              boxShadow: `0 0 6px ${market?.run_status === 'completed' ? 'var(--positive)' : 'var(--accent-amber)'}`,
            }} />
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {market?.run_status === 'completed' ? 'Live' : 'Mock'}
            </span>
          </div>
        </div>
      </nav>

      {/* ── Mobile Status Strip ── */}
      <div className="mobile-status-bar">
        <div className="msb-left">
          <span className="msb-label">VIX</span>
          <span className={`msb-value ${(market?.vix || 0) > 20 ? 'text-negative' : 'text-positive'}`}>
            {market?.vix ? market.vix.toFixed(2) : '—'}
          </span>
        </div>
        <div className="msb-right">
          <div className="status-dot" style={{
            background: market?.run_status === 'completed' ? 'var(--positive)' : 'var(--accent-amber)',
            boxShadow: `0 0 5px ${market?.run_status === 'completed' ? 'var(--positive)' : 'var(--accent-amber)'}`,
            width: 6, height: 6,
          }} />
          <span className="msb-label">
            {market?.run_status === 'completed' ? 'Live' : 'Mock'}
          </span>
        </div>
      </div>

      {/* ── Ticker ── */}
      <TickerBar />

      {/* ── Page ── */}
      <main className="page-content">
        <Outlet />
      </main>
    </div>
  );
}
