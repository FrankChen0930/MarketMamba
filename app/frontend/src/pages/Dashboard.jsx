import React, { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { fetchSignals } from '../api/signals';
import { fetchMarket } from '../api/market';
import StockModal from '../components/StockModal';
import SectorHeatmap from '../components/SectorHeatmap';
import { SkeletonCard, SkeletonTable, SkeletonBlock, ApiError } from '../components/SkeletonLoader';
import MetricTooltip from '../components/MetricTooltip';


// ── Helpers ──────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, valueClass, accent }) {
  return (
    <div className="stat-card animate-fade-up" style={accent ? {
      borderColor: accent,
      background: `color-mix(in srgb, ${accent} 5%, var(--bg-panel))`
    } : {}}>
      <div className="label">{label}</div>
      <div className={`value ${valueClass || ''}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

function SignalBadge({ signal }) {
  const cls = signal === 'BUY' ? 'badge-positive' : signal === 'SELL' ? 'badge-negative' : 'badge-neutral';
  const label = signal === 'BUY' ? '多' : signal === 'SELL' ? '空' : '觀望';
  return <span className={`badge ${cls}`}>{label}</span>;
}

function ConfBadge({ conf }) {
  const color = conf === '高信心' ? 'var(--positive)' : conf === '中信心' ? 'var(--accent-amber)' : 'var(--text-muted)';
  return <span style={{ fontSize: 11, color }}>{conf}</span>;
}

function AlphaBar({ value, max = 0.3 }) {
  const pct = Math.min(Math.abs(value) / max * 100, 100);
  const color = value >= 0 ? 'var(--positive)' : 'var(--negative)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ width: 52, height: 5, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`, height: '100%', background: color,
          borderRadius: 3, boxShadow: `0 0 5px ${color}`
        }} />
      </div>
      <span className={`mono ${value >= 0 ? 'text-positive' : 'text-negative'}`} style={{ fontSize: 11, minWidth: 52 }}>
        {value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%
      </span>
    </div>
  );
}

function MacroRow({ label, value, change, isUp }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
      <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{label}</span>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <span className="mono" style={{ fontSize: 13, color: 'var(--text-primary)' }}>{value}</span>
        {change !== undefined && (
          <span className={`mono ${isUp ? 'text-positive' : 'text-negative'}`} style={{ fontSize: 11 }}>
            {isUp ? '▲' : '▼'} {Math.abs(change).toFixed(2)}%
          </span>
        )}
      </div>
    </div>
  );
}

// ── Main ─────────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('20d');
  const [selectedStock, setSelectedStock] = useState(null);

  const { data: signalData, loading: sigLoading, error: sigError, refetch: refetchSigs } = useApi(() => fetchSignals());
  const { data: market, loading: mktLoading } = useApi(fetchMarket);

  const signals    = signalData?.signals || [];
  const sig5d  = [...signals].sort((a, b) => (b.alpha_5d  ?? 0) - (a.alpha_5d  ?? 0)).slice(0, 50);
  const sig20d = [...signals].sort((a, b) => (b.alpha_20d ?? 0) - (a.alpha_20d ?? 0)).slice(0, 50);
  const sig60d = [...signals].sort((a, b) => (b.alpha_60d ?? 0) - (a.alpha_60d ?? 0)).slice(0, 50);
  const displayed  = activeTab === '5d' ? sig5d : activeTab === '60d' ? sig60d : sig20d;
  const topSignals = sig20d;

  const loading = sigLoading || mktLoading;
  const taiex   = market?.taiex;
  const taiexUp = (taiex?.change ?? 0) >= 0;

  if (sigError) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header"><div className="page-title">今日選股訊號</div></div>
      <ApiError message={sigError} onRetry={refetchSigs} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {selectedStock && <StockModal stock={selectedStock} onClose={() => setSelectedStock(null)} />}

      {/* ── Page Header ── */}
      <div className="page-header">
        <div>
          <div className="page-title">今日選股訊號</div>
          <div className="page-subtitle">
            MarketMamba V6 Alpha · 推論日期：{signalData?.date || market?.last_run || '—'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {signalData?.freshness_warning && (
            <span style={{
              fontSize: 11, padding: '3px 10px', borderRadius: 12,
              background: 'rgba(245, 158, 11, 0.12)',
              color: 'var(--accent-amber)',
              border: '1px solid rgba(245, 158, 11, 0.25)',
            }}>
              ⚠️ {signalData.freshness_warning}
            </span>
          )}
          <button className="btn btn-primary" onClick={refetchSigs}>🔄 重新整理</button>
        </div>
      </div>

      {/* ── Macro Stat Cards ── */}
      <div className="grid-4">
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)
        ) : <>
          <StatCard
            label="加權指數 TAIEX"
            value={taiex?.value ? taiex.value.toLocaleString('zh-TW', { maximumFractionDigits: 0 }) : '—'}
            sub={<span className={taiexUp ? 'text-positive' : 'text-negative'}>
              {taiexUp ? '▲' : '▼'} {Math.abs(taiex?.change_pct ?? 0).toFixed(2)}%
              &nbsp;({taiexUp ? '+' : ''}{(taiex?.change ?? 0).toFixed(0)})
            </span>}
            accent={taiexUp ? 'var(--positive)' : 'var(--negative)'}
          />
          <StatCard
            label="漲 / 跌家數"
            value={`${market?.advancing ?? '—'} / ${market?.declining ?? '—'}`}
            sub={`總計 ${((market?.advancing ?? 0) + (market?.declining ?? 0)).toLocaleString()} 檔`}
          />
          <StatCard
            label={<>VIX 恐慌指數 <MetricTooltip metricKey="vix" /></>}
            value={market?.vix ? market.vix.toFixed(2) : '—'}
            sub={<span style={{ color: market?.vix > 20 ? 'var(--negative)' : 'var(--positive)' }}>
              {market?.vix > 30 ? '⚠️ 高度恐慌' : market?.vix > 20 ? '⚡ 波動偏高' : '✓ 市場平靜'}
            </span>}
          />
          <StatCard
            label="USD / TWD"
            value={market?.usd_twd ? market.usd_twd.toFixed(3) : '—'}
            sub={<span className={market?.spx_change >= 0 ? 'text-positive' : 'text-negative'}>
              SPX {market?.spx_change >= 0 ? '▲' : '▼'} {Math.abs(market?.spx_change ?? 0).toFixed(2)}%
            </span>}
          />
        </>}
      </div>

      {/* ── Main Area ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 330px', gap: 16, alignItems: 'start' }}>

        {/* Signal Table */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span className="panel-title-icon">⚡</span> Alpha 排名訊號
              {!loading && <span className="badge badge-neutral" style={{ marginLeft: 8, fontSize: 10 }}>
                可投資 {signals.filter(s => s.alpha_20d > 0).length} 檔
              </span>}
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {[['5d','5日 Alpha'], ['20d','20日 Alpha'], ['60d','60日 Alpha']].map(([tab, label]) => (
                <button
                  key={tab}
                  className={`btn ${activeTab === tab ? 'btn-primary' : 'btn-ghost'}`}
                  style={{ padding: '4px 12px', fontSize: 12 }}
                  onClick={() => setActiveTab(tab)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="panel-body-flush">
            {loading ? <SkeletonTable rows={10} cols={7} /> : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th style={{ width: 32 }}>#</th>
                    <th>股票</th>
                    <th>產業</th>
                    <th>{activeTab} Alpha 強度 <MetricTooltip metricKey="alpha" /></th>
                    <th>Sharpe <MetricTooltip metricKey="sharpe" /></th>
                    <th>建倉比重 <MetricTooltip metricKey="kelly" /></th>
                    <th>訊號</th>
                  </tr>
                </thead>
                <tbody>
                  {displayed.map((s, i) => (
                    <tr
                      key={s.stock_id}
                      style={{ animationDelay: `${i * 0.035}s`, cursor: 'pointer' }}
                      className="animate-fade-up"
                      onClick={() => setSelectedStock(s)}
                    >
                      <td style={{ color: 'var(--text-muted)', fontSize: 11 }}>{s.rank}</td>
                      <td>
                        <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 13 }}>{s.name}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{s.stock_id}</div>
                      </td>
                      <td><span className="badge badge-neutral" style={{ fontSize: 10 }}>{s.sector}</span></td>
                      <td><AlphaBar value={activeTab === '5d' ? (s.alpha_5d ?? 0) : activeTab === '60d' ? (s.alpha_60d ?? 0) : (s.alpha_20d ?? 0)} /></td>
                      <td>
                        <span className={`mono ${s.uncertainty < 0.03 ? 'text-positive' : 'text-secondary'}`} style={{ fontSize: 12 }}>
                          {(s.alpha_20d / Math.max(s.uncertainty, 0.001)).toFixed(1)}
                        </span>
                      </td>
                      <td>
                        <span className="mono" style={{ fontSize: 12 }}>
                          {s.suggested_weight ? `${(s.suggested_weight * 100).toFixed(1)}%` : '—'}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                          <SignalBadge signal={s.signal} />
                          <ConfBadge conf={s.confidence} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
          {!loading && (
            <div style={{ padding: '8px 16px', fontSize: 11, color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
              點擊股票查看 5d / 20d / 60d Alpha 詳情 · 數據截止 {signalData?.date || '—'}
            </div>
          )}
        </div>

        {/* Right Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Macro indicators */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🌐</span> 全球市場</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>即時</span>
            </div>
            <div className="panel-body">
              {mktLoading ? <SkeletonBlock height={120} /> : (<>
                <MacroRow
                  label="S&P 500"
                  value="—"
                  change={market?.spx_change}
                  isUp={(market?.spx_change ?? 0) >= 0}
                />
                <MacroRow
                  label="黃金 (Gold)"
                  value="—"
                  change={market?.gold_change}
                  isUp={(market?.gold_change ?? 0) >= 0}
                />
                <MacroRow
                  label="VIX 恐慌"
                  value={market?.vix?.toFixed(2) ?? '—'}
                />
                <MacroRow
                  label="USD / TWD"
                  value={market?.usd_twd?.toFixed(3) ?? '—'}
                />
                <div style={{ paddingTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  <span className="badge badge-neutral" style={{ fontSize: 10 }}>
                    漲 {market?.advancing ?? '—'} 跌 {market?.declining ?? '—'}
                  </span>
                  <span className="badge badge-blue" style={{ fontSize: 10 }}>
                    {signalData?.date || market?.last_run || '—'}
                  </span>
                </div>
              </>)}
            </div>
          </div>

          {/* Sector Heatmap — driven by real signals */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🏭</span> 產業強弱</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>模型 Alpha 加權</span>
            </div>
            <div className="panel-body">
              {sigLoading ? <SkeletonBlock height={140} /> : (
                <SectorHeatmap signals={signals} />
              )}
            </div>
          </div>

          {/* Top 5 Alpha stocks condensed */}
          <div className="panel" style={{ borderColor: 'rgba(0,255,136,0.15)', background: 'rgba(0,255,136,0.02)' }}>
            <div className="panel-header">
              <div className="panel-title"><span>🏆</span> 今日 Top 5</div>
            </div>
            <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {sigLoading ? <SkeletonBlock height={100} /> : topSignals.slice(0, 5).map((s, i) => (
                <div key={s.stock_id} style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}
                  onClick={() => setSelectedStock(s)}>
                  <span style={{ color: 'var(--accent-amber)', fontSize: 12, minWidth: 16, fontFamily: 'var(--font-mono)' }}>
                    {i + 1}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, fontWeight: 600 }}>{s.name}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{s.stock_id} · {s.sector}</div>
                  </div>
                  <span className="text-positive mono" style={{ fontSize: 12 }}>
                    +{(s.alpha_20d * 100).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
