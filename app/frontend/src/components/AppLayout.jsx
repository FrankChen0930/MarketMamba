import React, { useEffect, useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { fetchMarket, fetchTicker } from '../api/market';

const USER_MODE = import.meta.env.VITE_USER_MODE || 'personal';

const TABS = [
  { to: '/',         icon: '📊', label: '今日選股' },
  { to: '/quant',    icon: '📈', label: '量化分析' },
  { to: '/market',   icon: '🤖', label: 'AI 消息面' },
  ...(USER_MODE === 'personal'
    ? [{ to: '/portfolio', icon: '💼', label: '持倉追蹤' }]
    : []),
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
            <span className="brand-sub">V6 · ALPHA</span>
          </div>
        </a>

        <div className="nav-tabs">
          {TABS.map(tab => (
            <NavLink
              key={tab.to} to={tab.to} end={tab.to === '/'}
              className={({ isActive }) => `nav-tab${isActive ? ' active' : ''}`}
            >
              <span className="tab-icon">{tab.icon}</span>
              <span>{tab.label}</span>
            </NavLink>
          ))}
        </div>

        <div className="topbar-right">
          <div className="topbar-stat">
            <span className="ts-label">Model IC</span>
            <span className="ts-value text-positive mono">
              {market ? `+${market.model_ic.toFixed(4)}` : '—'}
            </span>
          </div>
          <div className="topbar-stat">
            <span className="ts-label">最後推論</span>
            <span className="ts-value mono" style={{ fontSize: 11 }}>
              {market?.last_run || '—'}
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

      {/* ── Ticker ── */}
      <TickerBar />

      {/* ── Page ── */}
      <main className="page-content">
        <Outlet />
      </main>
    </div>
  );
}
