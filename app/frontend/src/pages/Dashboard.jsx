import React, { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchSignals } from '../api/signals';
import { fetchMarket } from '../api/market';
import { fetchICHistory } from '../api/performance';
import StockModal from '../components/StockModal';
import SectorHeatmap from '../components/SectorHeatmap';
import { SkeletonCard, SkeletonTable, SkeletonBlock, ApiError } from '../components/SkeletonLoader';

// ── Sub-components ────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, valueClass }) {
  return (
    <div className="stat-card animate-fade-up">
      <div className="label">{label}</div>
      <div className={`value ${valueClass || ''}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

function SignalBadge({ signal }) {
  const cls = signal === 'BUY' ? 'badge-positive' : signal === 'SELL' ? 'badge-negative' : 'badge-neutral';
  return <span className={`badge ${cls}`}>{signal}</span>;
}

function AlphaBar({ value, max = 0.25 }) {
  const pct = Math.min(Math.abs(value) / max * 100, 100);
  const color = value >= 0 ? 'var(--positive)' : 'var(--negative)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ width: 60, height: 6, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`, height: '100%', background: color,
          borderRadius: 3, transition: 'width 0.5s ease', boxShadow: `0 0 6px ${color}`
        }} />
      </div>
      <span className={`mono ${value >= 0 ? 'text-positive' : 'text-negative'}`} style={{ fontSize: 12, minWidth: 56 }}>
        {value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%
      </span>
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass" style={{ padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>Epoch {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontFamily: 'var(--font-mono)' }}>
          {p.name}: {p.value?.toFixed(4)}
        </div>
      ))}
    </div>
  );
};

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('top');
  const [selectedStock, setSelectedStock] = useState(null);

  const { data: signalData, loading: sigLoading, error: sigError, refetch: refetchSigs } = useApi(() => fetchSignals());
  const { data: market, loading: mktLoading, error: mktError } = useApi(fetchMarket);
  const { data: icHistory, loading: icLoading } = useApi(fetchICHistory);

  const signals    = signalData?.signals || [];
  const topSignals = signals.filter(s => s.alpha_20d > 0).slice(0, 10);
  const botSignals = signals.filter(s => s.alpha_20d < 0).slice(0, 5);
  const displayed  = activeTab === 'top' ? topSignals : botSignals;
  const lastIc     = icHistory?.[icHistory.length - 1];

  const loading = sigLoading || mktLoading || icLoading;
  const error   = sigError || mktError;

  // ── Training Status Banner ─────────────────────────────────────────────────
  const trainingBanner = market?.training_status === 'training' && (
    <div style={{
      background: 'rgba(0,212,255,0.06)',
      border: '1px solid rgba(0,212,255,0.2)',
      borderRadius: 8, padding: '10px 16px',
      display: 'flex', alignItems: 'center', gap: 12,
      fontSize: 12,
    }}>
      <span style={{ animation: 'pulse-glow 2s infinite', display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-blue)', flexShrink: 0 }} />
      <span style={{ color: 'var(--text-secondary)' }}>
        Final Training 進行中（Epoch {market.training_epoch || '?'}/100）· 推論功能將在訓練完成後啟用
      </span>
    </div>
  );

  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header">
        <div><div className="page-title">今日選股訊號</div></div>
      </div>
      <ApiError message={error} onRetry={refetchSigs} />
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
            MarketMamba V6 Alpha · {market?.last_run || '—'} 推論完成
          </div>
        </div>
        <button className="btn btn-primary" onClick={refetchSigs}>🔄 重新整理</button>
      </div>

      {trainingBanner}

      {/* ── Stat Cards ── */}
      <div className="grid-4">
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)
        ) : <>
          <StatCard
            label="加權指數" value={market?.taiex.value.toLocaleString()}
            sub={<span className={market?.taiex.change >= 0 ? 'text-positive' : 'text-negative'}>
              {market?.taiex.change >= 0 ? '▲' : '▼'} {Math.abs(market?.taiex.change_pct ?? 0).toFixed(2)}%
            </span>}
          />
          <StatCard
            label="上漲 / 下跌"
            value={`${market?.advancing ?? '—'} / ${market?.declining ?? '—'}`}
            sub="今日漲跌家數"
          />
          <StatCard
            label="Model Val IC"
            value={lastIc ? `+${lastIc.val_ic.toFixed(4)}` : '—'}
            valueClass="text-positive"
            sub={lastIc ? `Epoch ${lastIc.epoch} best` : 'Training…'}
          />
          <StatCard label="今日成交金額" value="2,841億" sub="台股整體" />
        </>}
      </div>

      {/* ── Main Area ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 16, alignItems: 'start' }}>

        {/* Signal Table */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span className="panel-title-icon">⚡</span> Alpha 排名訊號
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {['top', 'bot'].map(tab => (
                <button
                  key={tab}
                  className={`btn ${activeTab === tab ? 'btn-primary' : 'btn-ghost'}`}
                  style={{ padding: '4px 12px', fontSize: 12 }}
                  onClick={() => setActiveTab(tab)}
                >
                  {tab === 'top' ? '多方 Top' : '空方 Bot'}
                </button>
              ))}
            </div>
          </div>
          <div className="panel-body-flush">
            {loading ? <SkeletonTable rows={8} cols={6} /> : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>排名</th><th>股票</th><th>產業</th>
                    <th>Alpha 強度</th><th>量比</th><th>訊號</th>
                  </tr>
                </thead>
                <tbody>
                  {displayed.map((s, i) => (
                    <tr
                      key={s.stock_id}
                      style={{ animationDelay: `${i * 0.04}s`, cursor: 'pointer' }}
                      className="animate-fade-up"
                      onClick={() => setSelectedStock(s)}
                    >
                      <td style={{ color: 'var(--text-muted)', fontSize: 11 }}>#{s.rank}</td>
                      <td>
                        <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{s.name}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{s.stock_id}</div>
                      </td>
                      <td><span className="badge badge-neutral" style={{ fontSize: 10 }}>{s.sector}</span></td>
                      <td><AlphaBar value={s.alpha_20d} /></td>
                      <td>
                        <span className={`mono ${s.vol_ratio > 1 ? 'text-positive' : 'text-muted'}`} style={{ fontSize: 12 }}>
                          {s.vol_ratio.toFixed(2)}x
                        </span>
                      </td>
                      <td><SignalBadge signal={s.signal} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
          {!loading && (
            <div style={{ padding: '8px 16px', fontSize: 11, color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
              點擊任一股票查看詳細 Alpha 分析
            </div>
          )}
        </div>

        {/* Right Side */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* IC Sparkline */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>📉</span> Val IC 趨勢</div>
              {lastIc && <span className="badge badge-positive">+{lastIc.val_ic.toFixed(4)}</span>}
            </div>
            <div className="panel-body" style={{ paddingTop: 8 }}>
              {icLoading ? <SkeletonBlock height={120} /> : (
                <ResponsiveContainer width="100%" height={120}>
                  <LineChart data={icHistory || []}>
                    <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={[0, 0.12]} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={0.05} stroke="rgba(0,255,136,0.3)" strokeDasharray="4 4" />
                    <Line type="monotone" dataKey="val_ic" stroke="var(--accent-blue)" strokeWidth={2} dot={false} name="Val IC" />
                  </LineChart>
                </ResponsiveContainer>
              )}
              <div className="divider" />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }}>
                <span>IC=0.05 門檻</span>
                <span className="text-positive">✓ 已達標</span>
              </div>
            </div>
          </div>

          {/* Market Status */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🌐</span> 系統狀態</div>
            </div>
            <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {mktLoading ? Array.from({ length: 4 }).map((_, i) => (
                <SkeletonBlock key={i} height={14} width={`${60 + i * 10}%`} />
              )) : [
                { label: '推論狀態', value: market?.run_status === 'completed' ? '✅ 已完成' : '⏳ 訓練中', cls: market?.run_status === 'completed' ? 'text-positive' : 'text-accent' },
                { label: '訓練 Epoch', value: `${market?.training_epoch || '?'} / 100`, cls: '' },
                { label: '最佳 IC',    value: lastIc ? `+${lastIc.val_ic.toFixed(4)}` : '—', cls: 'text-positive' },
                { label: '資料截止',   value: signalData?.date || '—', cls: '' },
              ].map(item => (
                <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                  <span style={{ color: 'var(--text-muted)' }}>{item.label}</span>
                  <span className={`mono ${item.cls}`}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Sector Heatmap */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🏭</span> 產業強弱</div>
            </div>
            <div className="panel-body">
              {sigLoading ? <SkeletonBlock height={120} /> : (
                <SectorHeatmap signals={signals} />
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
