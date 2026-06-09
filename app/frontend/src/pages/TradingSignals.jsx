import React, { useState, useEffect } from 'react';
import { useApi } from '../hooks/useApi';
import { fetchScannerSignals } from '../api/signals';
import StockModal from '../components/StockModal';
import { SkeletonCard, SkeletonBlock, ApiError } from '../components/SkeletonLoader';
import MetricTooltip from '../components/MetricTooltip';

// ── Entry Rules Modal ────────────────────────────────────────────────────────

function EntryRulesModal({ regime, onClose }) {
  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  const S = { section: { marginBottom: 20 }, th: { fontSize: 12, color: 'var(--text-muted)', padding: '8px 12px', borderBottom: '1px solid var(--border)', textAlign: 'left' }, td: { fontSize: 12, padding: '8px 12px', borderBottom: '1px solid rgba(48,54,61,0.4)' } };

  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div onClick={e => e.stopPropagation()} className="panel" style={{ width: 600, maxWidth: '90vw', maxHeight: '85vh', overflowY: 'auto', border: '1px solid var(--border-bright)', boxShadow: 'var(--shadow-glow-blue)', animation: 'fadeInUp 0.2s ease forwards' }}>
        <div className="panel-header" style={{ justifyContent: 'space-between' }}>
          <div style={{ fontSize: 16, fontWeight: 700 }}>📋 入場 / 退場規則</div>
          <button onClick={onClose} className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: 16, lineHeight: 1 }}>✕</button>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>

          {/* Entry conditions */}
          <div style={S.section}>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--positive)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>🟢</span> 入場條件（滿足 {regime === 'CAUTIOUS' ? '3' : '2'}/4 觸發推薦）
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, padding: '6px 10px', background: 'rgba(0,255,136,0.04)', borderRadius: 6 }}>
              滿足任意 {regime === 'CAUTIOUS' ? '3' : '2'} 個條件即進入買入推薦；觀察清單顯示的「評分」為各條件分數加總，可比較同層股票的信號強度
            </div>
            <table className="data-table" style={{ width: '100%' }}>
              <thead><tr><th style={S.th}>#</th><th style={S.th}>條件</th><th style={{ ...S.th, textAlign: 'center' }}>參考分數</th><th style={S.th}>判斷邏輯</th></tr></thead>
              <tbody>
                {[
                  ['1', '排名穩定性', '30', 'Top 10 連續 ≥2 天 或 Top 50 連續 ≥3 天'],
                  ['2', '高信心', '25', 'Uncertainty < 當日 Q30 分位數（MC-Dropout）'],
                  ['3', '機構連續淨買', '25', '外資/投信 連續 2 天淨買入'],
                  ['4', '相對低點', '20', 'RSI < 40 或 當前價 < 20日均線'],
                ].map(([n, name, score, logic]) => (
                  <tr key={n}>
                    <td style={{ ...S.td, color: 'var(--accent-amber)', fontWeight: 600 }}>{n}</td>
                    <td style={{ ...S.td, fontWeight: 600, color: 'var(--text-primary)' }}>{name}</td>
                    <td style={{ ...S.td, textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-blue)' }}>+{score}</td>
                    <td style={{ ...S.td, color: 'var(--text-secondary)', fontSize: 11 }}>{logic}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pattern bonus */}
          <div style={S.section}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#a78bfa', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>🔮</span> 型態加分（疊加在 4 條件分數上，最高 +50）
            </div>
            <table className="data-table" style={{ width: '100%' }}>
              <thead><tr><th style={S.th}>型態分數</th><th style={{ ...S.th, textAlign: 'center' }}>加分</th><th style={S.th}>說明</th></tr></thead>
              <tbody>
                {[
                  ['60 – 74', '+20', '普通型態確認'],
                  ['75 – 89', '+30', '強型態訊號'],
                  ['≥ 90',    '+40', '完美型態'],
                  ['雙確認 (型態≥60 且 Alpha排名≤200)', '+10', '型態 + 模型共同確認'],
                ].map(([cond, pts, desc]) => (
                  <tr key={cond}>
                    <td style={{ ...S.td, fontFamily: 'var(--font-mono)', color: '#a78bfa' }}>{cond}</td>
                    <td style={{ ...S.td, textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-blue)' }}>{pts}</td>
                    <td style={{ ...S.td, color: 'var(--text-secondary)', fontSize: 11 }}>{desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6, padding: '5px 10px', background: 'rgba(139,92,246,0.05)', borderRadius: 6 }}>
              型態掃描器辨識 5 種多方型態（W底、彈簧W底、頭肩底、收斂三角底、上飄旗形）。每個買入訊號卡片底部若出現紫色區塊，代表有型態加持。
            </div>
          </div>

          {/* Market regime */}
          <div style={S.section}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent-amber)', marginBottom: 8 }}>🌐 大盤環境過濾</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {[
                { env: 'TWII > 60日均線', label: '正常市場', threshold: '滿足 2/4 即推薦', sub: '1 項符合 → 觀察清單', active: regime !== 'CAUTIOUS' },
                { env: 'TWII < 60日均線', label: '保守模式', threshold: '需滿足 3/4 才推薦', sub: '1-2 項符合 → 觀察清單', active: regime === 'CAUTIOUS' },
              ].map(r => (
                <div key={r.label} style={{ padding: '10px 14px', borderRadius: 8, background: r.active ? 'rgba(0,212,255,0.06)' : 'var(--bg-panel-2)', border: r.active ? '1px solid rgba(0,212,255,0.3)' : '1px solid transparent' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{r.env}</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: r.active ? 'var(--accent-blue)' : 'var(--text-secondary)' }}>{r.label}</div>
                  <div style={{ fontSize: 11, color: r.active ? 'var(--positive)' : 'var(--text-muted)', marginTop: 4, fontWeight: r.active ? 600 : 400 }}>{r.threshold}</div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{r.sub}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="divider" />

          {/* Exit conditions */}
          <div style={S.section}>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--negative)', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>🔴</span> 退場條件（任一觸發）
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {[
                { cond: 'Alpha 排名連續 2 天掉出 Top 50', action: '🔴 退場' },
                { cond: '觸及 Trailing Stop', action: '🔴 退場' },
                { cond: '外資連續 3 天淨賣出', action: '⚠️ 減碼觀察' },
                { cond: '大盤跌破 60MA', action: '⚠️ 全倉減碼至 50%' },
              ].map(r => (
                <div key={r.cond} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '6px 12px', background: 'rgba(255,71,87,0.04)', borderRadius: 6 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{r.cond}</span>
                  <span style={{ color: 'var(--negative)', fontWeight: 600, flexShrink: 0, marginLeft: 12 }}>{r.action}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Trailing Stop */}
          <div style={S.section}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent-amber)', marginBottom: 8 }}>📉 Trailing Stop 機制</div>
            <table className="data-table" style={{ width: '100%' }}>
              <thead><tr><th style={S.th}>持倉報酬</th><th style={S.th}>止損線位置</th></tr></thead>
              <tbody>
                {[['< +5%', '固定 -5%（成本價）'], ['≥ +5%', '成本 +2%（鎖利）'], ['≥ +10%', '成本 +6%'], ['≥ +15%', '成本 +10%']].map(([ret, stop]) => (
                  <tr key={ret}><td style={S.td}>{ret}</td><td style={{ ...S.td, color: 'var(--accent-amber)', fontFamily: 'var(--font-mono)' }}>{stop}</td></tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Position sizing */}
          <div style={S.section}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent-blue)', marginBottom: 8 }}>💰 部位管理</div>
            <table className="data-table" style={{ width: '100%' }}>
              <thead><tr><th style={S.th}>Sharpe Score</th><th style={S.th}>建議配置</th></tr></thead>
              <tbody>
                {[['> 3（高）', '15-20% 資金'], ['1-3（中）', '5-10% 資金'], ['< 1（低）', '不買']].map(([s, alloc]) => (
                  <tr key={s}><td style={S.td}>{s}</td><td style={{ ...S.td, fontWeight: 600 }}>{alloc}</td></tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: 11, color: 'var(--accent-amber)', marginTop: 8, padding: '6px 10px', background: 'rgba(255,165,0,0.06)', borderRadius: 6 }}>
              ⚠️ 同一產業（sector）不超過 30% 總部位，避免集中風險
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


// ── Signal Card (Buy) ────────────────────────────────────────────────────────

function SignalCard({ signal, onClick }) {
  const isOOD = signal.alpha_20d > 1.0;
  const borderColor = isOOD ? 'rgba(245,158,11,0.35)' : 'rgba(0,255,136,0.25)';
  const bgColor     = isOOD ? 'rgba(245,158,11,0.03)' : 'rgba(0,255,136,0.03)';
  const hoverBorder = isOOD ? 'rgba(245,158,11,0.65)' : 'rgba(0,255,136,0.5)';
  const hoverShadow = isOOD ? '0 0 16px rgba(245,158,11,0.2)' : 'var(--shadow-glow-green)';

  return (
    <div className="panel animate-fade-up" onClick={onClick}
      style={{ borderColor, background: bgColor, cursor: 'pointer', transition: 'all 0.2s' }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = hoverBorder; e.currentTarget.style.boxShadow = hoverShadow; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = borderColor; e.currentTarget.style.boxShadow = 'none'; }}
    >
      <div className="panel-body" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
              {signal.name && signal.name !== signal.ticker ? <>{signal.name} <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 400 }}>{signal.ticker}</span></> : signal.ticker}
              <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--positive)', background: 'rgba(0,255,136,0.1)', padding: '2px 8px', borderRadius: 99, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>買入推薦</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              {signal.confidence} · 評分 <span style={{ color: 'var(--positive)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{signal.score ?? '—'}</span>/150 · 符合 {signal.conditions_met}/{signal.conditions_total} 條件
            </div>
          </div>
          {signal.alpha_20d != null && (
            <div style={{ textAlign: 'right', flexShrink: 0 }}>
              {isOOD && (
                <div style={{ background: '#f59e0b', color: '#1a1a1a', fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 99, marginBottom: 6, display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                  ⚠️ 異常Alpha
                </div>
              )}
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
                Alpha 20d
                {isOOD && <MetricTooltip metricKey="alpha_ood" />}
              </div>
              <div className="mono" style={{ fontSize: 20, fontWeight: 700, color: isOOD ? '#f59e0b' : (signal.alpha_20d >= 0 ? 'var(--positive)' : 'var(--negative)') }}>
                {signal.alpha_20d >= 0 ? '+' : ''}{(signal.alpha_20d * 100).toFixed(1)}%
              </div>
            </div>
          )}
        </div>
        {/* Conditions grid */}
        {signal.rank_stability && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px', padding: '10px 14px', background: 'var(--bg-panel-2)', borderRadius: 'var(--radius-sm)', marginBottom: 12 }}>
            {[
              { met: signal.rank_stability?.met, label: signal.rank_stability?.detail },
              { met: signal.high_confidence?.met, label: signal.high_confidence?.detail },
              { met: signal.relative_low?.met, label: signal.relative_low?.detail },
              { met: signal.institutional_buy?.met, label: signal.institutional_buy?.detail?.includes('無此股') ? '— 無資料' : signal.institutional_buy?.detail },
            ].map((c, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: c.label === '— 無資料' ? 'var(--text-muted)' : c.met ? 'var(--positive)' : 'var(--text-muted)' }}>
                <span style={{ fontSize: 13 }}>{c.label === '— 無資料' ? '➖' : c.met ? '✅' : '❌'}</span>
                <span>{c.label}</span>
              </div>
            ))}
          </div>
        )}
        {/* Pattern bonus row */}
        {signal.pattern && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px', background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.25)', borderRadius: 'var(--radius-sm)', marginBottom: 12 }}>
            <span style={{ fontSize: 14 }}>🔮</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: '#a78bfa' }}>{signal.pattern.pattern_name}</span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 6 }}>型態分 {signal.pattern.pattern_score}</span>
              {signal.pattern.dual_confirm && (
                <span style={{ marginLeft: 6, fontSize: 10, fontWeight: 700, color: '#f59e0b', background: 'rgba(245,158,11,0.15)', padding: '1px 6px', borderRadius: 99 }}>雙確認</span>
              )}
              {signal.pattern.failure_stop != null && (
                <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 10 }}>
                  失敗止損 <span style={{ color: 'var(--negative)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{signal.pattern.failure_stop}</span>
                </span>
              )}
            </div>
            <span style={{ fontSize: 12, fontWeight: 700, color: '#a78bfa', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>+{signal.pattern.pattern_bonus}分</span>
          </div>
        )}

        {/* Bottom metrics */}
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {signal.sharpe != null && <div style={{ fontSize: 12 }}><span style={{ color: 'var(--text-muted)' }}>Sharpe </span><span className="mono" style={{ color: signal.sharpe > 3 ? 'var(--positive)' : 'var(--text-secondary)', fontWeight: 600 }}>{signal.sharpe.toFixed(1)}</span></div>}
          {signal.uncertainty != null && <div style={{ fontSize: 12 }}><span style={{ color: 'var(--text-muted)' }}>不確定度 </span><span className="mono" style={{ color: signal.uncertainty < 0.02 ? 'var(--positive)' : 'var(--accent-amber)', fontWeight: 600 }}>±{(signal.uncertainty * 100).toFixed(1)}%</span></div>}
          {signal.suggested_weight != null && <div style={{ fontSize: 12 }}><span style={{ color: 'var(--text-muted)' }}>Kelly </span><span className="mono" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{(signal.suggested_weight * 100).toFixed(1)}%</span></div>}
        </div>
      </div>
    </div>
  );
}


// ── Watch List Table ─────────────────────────────────────────────────────────

function WatchListTable({ watchList, onSelectStock }) {
  if (!watchList?.length) return null;
  const fmtInst = (d) => d?.includes('無此股') ? '— 無資料' : d;
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>👀</span> 觀察清單</div>
        <span className="badge badge-neutral" style={{ fontSize: 10 }}>評分未達門檻 · {watchList.length} 檔</span>
      </div>
      <div className="panel-body-flush" style={{ overflowX: 'auto' }}>
        <table className="data-table" style={{ minWidth: 760 }}>
          <thead>
            <tr>
              <th style={{ minWidth: 110 }}>股票</th>
              <th style={{ textAlign: 'center', minWidth: 70 }}>評分</th>
              <th>排名穩定性</th>
              <th>模型信心</th>
              <th>相對低點</th>
              <th>機構買賣超</th>
              <th>型態</th>
              <th style={{ textAlign: 'right' }}>Alpha 20d</th>
            </tr>
          </thead>
          <tbody>
            {watchList.map((s, i) => (
              <tr key={s.ticker} className="animate-fade-up" style={{ animationDelay: `${i * 0.03}s`, cursor: 'pointer' }}
                onClick={() => onSelectStock({ stock_id: s.ticker, name: s.name || s.ticker, sector: s.sector || '—', alpha_5d: 0, alpha_20d: s.alpha_20d, alpha_60d: 0, uncertainty: s.uncertainty, vol_ratio: 1, signal: 'HOLD', suggested_weight: s.suggested_weight, confidence: s.confidence, rank: '-' })}>
                <td>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>{s.name && s.name !== s.ticker ? s.name : s.ticker}</div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{s.ticker}</div>
                </td>
                <td style={{ textAlign: 'center' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 14, color: (s.score ?? 0) >= 45 ? 'var(--accent-amber)' : 'var(--text-secondary)' }}>{s.score ?? '—'}</div>
                  <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>/ 150</div>
                </td>
                <td><span style={{ fontSize: 12, color: s.rank_stability?.met ? 'var(--positive)' : 'var(--text-muted)' }}>{s.rank_stability?.met ? '✅' : '❌'} {s.rank_stability?.detail}</span></td>
                <td><span style={{ fontSize: 12, color: s.high_confidence?.met ? 'var(--positive)' : 'var(--text-muted)' }}>{s.high_confidence?.met ? '✅' : '❌'} {s.high_confidence?.detail}</span></td>
                <td><span style={{ fontSize: 12, color: s.relative_low?.met ? 'var(--positive)' : 'var(--text-muted)' }}>{s.relative_low?.met ? '✅' : '❌'} {s.relative_low?.detail}</span></td>
                <td><span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{fmtInst(s.institutional_buy?.detail) === '— 無資料' ? '➖ 無資料' : (s.institutional_buy?.met ? '✅' : '❌') + ' ' + s.institutional_buy?.detail}</span></td>
                <td>
                  {s.pattern
                    ? <span style={{ fontSize: 11, color: '#a78bfa', fontWeight: 600 }}>🔮 {s.pattern.pattern_name} <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>+{s.pattern.pattern_bonus}</span></span>
                    : <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>—</span>
                  }
                </td>
                <td style={{ textAlign: 'right' }}>
                  <span className="mono" style={{ fontSize: 13, fontWeight: 600, color: s.alpha_20d >= 0 ? 'var(--positive)' : 'var(--negative)' }}>
                    {s.alpha_20d >= 0 ? '+' : ''}{(s.alpha_20d * 100).toFixed(1)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


// ── Main Page ────────────────────────────────────────────────────────────────

export default function TradingSignals() {
  const { data, loading, error, refetch } = useApi(fetchScannerSignals);
  const [selectedStock, setSelectedStock] = useState(null);
  const [showRules, setShowRules] = useState(false);

  const buySignals = data?.buy_signals || [];
  const exitSignals = data?.exit_signals || [];
  const watchList = data?.watch_list || [];

  const regimeColor = data?.market_regime === 'CAUTIOUS' ? 'var(--accent-amber)' : 'var(--positive)';
  const regimeLabel = data?.market_regime === 'CAUTIOUS' ? '保守模式' : '正常市場';
  const regimeIcon = data?.market_regime === 'CAUTIOUS' ? '🟡' : '🟢';

  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header"><div className="page-title">交易訊號掃描</div></div>
      <ApiError message={error} onRetry={refetch} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {selectedStock && <StockModal stock={selectedStock} onClose={() => setSelectedStock(null)} />}
      {showRules && <EntryRulesModal regime={data?.market_regime} onClose={() => setShowRules(false)} />}

      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-title">🎯 交易訊號掃描</div>
          <div className="page-subtitle">MarketMamba V6.1 Signal Scanner · {data?.date || '—'}</div>
        </div>
        <button className="btn btn-primary" onClick={refetch}>🔄 重新掃描</button>
      </div>

      {/* Market Regime Strip */}
      <div className="grid-4">
        {loading ? Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />) : <>
          <div className="stat-card" style={{ borderColor: regimeColor, background: `color-mix(in srgb, ${regimeColor} 5%, var(--bg-panel))` }}>
            <div className="label">大盤環境</div>
            <div className="value" style={{ color: regimeColor }}>{regimeIcon} {regimeLabel}</div>
            <div className="sub">TWII vs MA60: {data?.twii_vs_ma60 || '—'}</div>
          </div>

          <div className="stat-card" style={{ cursor: 'pointer', transition: 'border-color 0.2s' }}
            onClick={() => setShowRules(true)}
            onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accent-blue)'}
            onMouseLeave={e => e.currentTarget.style.borderColor = ''}>
            <div className="label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>入場門檻 <span style={{ fontSize: 10, color: 'var(--accent-blue)' }}>📋 點擊查看規則</span></div>
            <div className="value mono">{data?.entry_threshold || '—'}</div>
            <div className="sub">滿足條件數 / 總條件</div>
          </div>

          <div className="stat-card" style={{ borderColor: buySignals.length > 0 ? 'rgba(0,255,136,0.3)' : 'var(--border)' }}>
            <div className="label">🔥 買入推薦</div>
            <div className="value mono text-positive">{buySignals.length} 檔</div>
            <div className="sub">滿足入場條件</div>
          </div>

          <div className="stat-card" style={{ borderColor: exitSignals.length > 0 ? 'rgba(255,71,87,0.3)' : 'var(--border)' }}>
            <div className="label">⚠️ 退場警告</div>
            <div className="value mono" style={{ color: exitSignals.length > 0 ? 'var(--negative)' : 'var(--text-secondary)' }}>{exitSignals.length} 檔</div>
            <div className="sub">觸發退場條件</div>
          </div>
        </>}
      </div>

      {/* Buy Signals */}
      {!loading && (
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--positive)', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 18 }}>🔥</span> 買入推薦
            <span className="badge badge-positive" style={{ fontSize: 10 }}>{buySignals.length} 檔</span>
          </div>
          {buySignals.length === 0 ? (
            <div className="panel"><div className="panel-body" style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>目前沒有股票達到入場條件 — 等待更好的機會 🧘</div></div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: 12 }}>
              {buySignals.map((s, i) => (
                <div key={s.ticker} style={{ animationDelay: `${i * 0.05}s` }}>
                  <SignalCard signal={s} onClick={() => setSelectedStock({ stock_id: s.ticker, name: s.name || s.ticker, sector: s.sector || '—', alpha_5d: 0, alpha_20d: s.alpha_20d, alpha_60d: 0, uncertainty: s.uncertainty, vol_ratio: 1, signal: 'BUY', suggested_weight: s.suggested_weight, confidence: s.confidence, rank: i + 1 })} />
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Watch List Table */}
      {!loading && <WatchListTable watchList={watchList} onSelectStock={setSelectedStock} />}

      {/* How it works */}
      {!loading && (
        <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.15)', background: 'rgba(0,212,255,0.02)' }}>
          <div className="panel-body" style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
            <span style={{ fontSize: 20 }}>💡</span>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
              <strong style={{ color: 'var(--accent-blue)' }}>運作原理：</strong>
              掃描模型 Top 50 股票，評估 4 個維度的入場條件（排名穩定性、模型信心、相對低點、機構資金方向）。
              {data?.market_regime === 'CAUTIOUS' ? '目前大盤低於 60 日均線，進入保守模式，需滿足 3/4 條件才推薦。' : '正常市場環境下，滿足 2/4 條件即推薦買入。'}
              點擊上方 <strong>「入場門檻」</strong> 查看完整規則。
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
