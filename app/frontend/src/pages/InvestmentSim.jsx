import React, { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, BarChart, Bar, Cell,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchSimBacktest } from '../api/sim';
import { fetchRebalanceHistory } from '../api/signals';
import { SkeletonBlock, SkeletonCard, ApiError } from '../components/SkeletonLoader';


// ── Tooltip ───────────────────────────────────────────────────────────────────

function EquityTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const pt = payload[0]?.payload;
  return (
    <div className="glass" style={{ padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ color: 'var(--accent-blue)' }}>
        資產淨值：NT${pt?.equity?.toLocaleString()}
      </div>
      <div style={{ color: pt?.pnl >= 0 ? 'var(--positive)' : 'var(--negative)', marginTop: 2 }}>
        累積損益：{pt?.pnl >= 0 ? '+' : ''}NT${pt?.pnl?.toLocaleString()}
      </div>
    </div>
  );
}

function ReturnTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const val = payload[0]?.value ?? 0;
  return (
    <div className="glass" style={{ padding: '8px 12px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
      <div style={{ color: val >= 0 ? 'var(--positive)' : 'var(--negative)', fontFamily: 'var(--font-mono)' }}>
        {val >= 0 ? '+' : ''}{val.toFixed(3)}%
      </div>
    </div>
  );
}


// ── Stat Card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, valueColor }) {
  return (
    <div className="stat-card">
      <div className="label">{label}</div>
      <div className="value mono" style={{ color: valueColor || 'var(--text-primary)' }}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}


// ── Rebalance History ─────────────────────────────────────────────────────────

const CONF_COLOR = { '高信心': '#00ff88', '中信心': '#ffa500', '低信心': '#ff4757' };

function RebalanceHistory() {
  const { data, loading } = useApi(fetchRebalanceHistory);
  const history = data?.history ?? [];
  const [activeIdx, setActiveIdx] = useState(0);

  if (loading) return (
    <div className="panel">
      <div className="panel-header"><div className="panel-title"><span>📅</span> 調倉紀錄</div></div>
      <div className="panel-body"><SkeletonBlock height={200} /></div>
    </div>
  );

  if (!history.length) return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>📅</span> 調倉紀錄</div>
        <span className="badge badge-neutral" style={{ fontSize: 10 }}>尚無歷史資料</span>
      </div>
      <div className="panel-body" style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, padding: '24px 0' }}>
        每次執行 run_daily_inference.py 後自動新增一筆記錄
      </div>
    </div>
  );

  const current  = history[activeIdx];
  const previous = history[activeIdx + 1];
  const prevTickers = new Set((previous?.portfolio ?? []).map(p => p.ticker));
  const currTickers = new Set(current.portfolio.map(p => p.ticker));
  const newIn  = current.portfolio.filter(p => !prevTickers.has(p.ticker)).map(p => p.ticker);
  const newOut = previous ? [...prevTickers].filter(t => !currTickers.has(t)) : [];

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>📅</span> 調倉紀錄</div>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>共 {history.length} 次</span>
      </div>

      <div style={{ display: 'flex', gap: 6, padding: '0 16px 12px', flexWrap: 'wrap', borderBottom: '1px solid var(--border)' }}>
        {history.map((h, i) => (
          <button key={h.date} onClick={() => setActiveIdx(i)}
            className={`btn ${i === activeIdx ? 'btn-primary' : 'btn-ghost'}`}
            style={{ padding: '4px 12px', fontSize: 11 }}>
            {h.date}
          </button>
        ))}
      </div>

      <div className="panel-body">
        <div style={{ display: 'flex', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            可投資股數：<strong style={{ color: 'var(--text-primary)' }}>{current.total_investable}</strong>
          </span>
          {newIn.length > 0 && <span style={{ fontSize: 12, color: '#00ff88' }}>🟢 新進：{newIn.join('、')}</span>}
          {newOut.length > 0 && <span style={{ fontSize: 12, color: '#ff4757' }}>🔴 移出：{newOut.join('、')}</span>}
          {!previous && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>（最早記錄，無法比較變動）</span>}
        </div>

        <table className="data-table">
          <thead>
            <tr><th>#</th><th>股票代號</th><th>比重</th><th>Alpha 20d</th><th>Sharpe</th><th>信心</th></tr>
          </thead>
          <tbody>
            {current.portfolio.map((p, i) => {
              const isNew = newIn.includes(p.ticker);
              return (
                <tr key={p.ticker} style={isNew ? { background: 'rgba(0,255,136,0.04)' } : undefined}>
                  <td className="mono" style={{ color: 'var(--text-muted)', fontSize: 11 }}>{i + 1}</td>
                  <td>
                    <span style={{ fontWeight: 600, fontSize: 12 }}>{p.ticker}</span>
                    {isNew && <span style={{ marginLeft: 6, fontSize: 10, color: '#00ff88' }}>NEW</span>}
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>{p.weight ? `${(p.weight * 100).toFixed(1)}%` : '—'}</td>
                  <td className={`mono ${p.alpha >= 0 ? 'text-positive' : 'text-negative'}`} style={{ fontSize: 11 }}>
                    {p.alpha >= 0 ? '+' : ''}{(p.alpha * 100).toFixed(2)}%
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>{p.sharpe?.toFixed(2)}</td>
                  <td><span style={{ fontSize: 10, color: CONF_COLOR[p.confidence] ?? 'var(--text-muted)' }}>{p.confidence || '—'}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}


// ── Main Page ─────────────────────────────────────────────────────────────────

export default function InvestmentSim() {
  const { data, loading, error, refetch } = useApi(fetchSimBacktest);

  const curve    = data?.equity_curve ?? [];
  const hasData  = curve.length >= 2;

  // Trim view to last N points
  const [viewDays, setViewDays] = useState(0);   // 0 = all
  const displayCurve = viewDays > 0 ? curve.slice(-viewDays - 1) : curve;

  const totalReturnPct  = data?.total_return_pct  ?? 0;
  const maxDrawdownPct  = data?.max_drawdown_pct  ?? 0;
  const sharpe          = data?.sharpe            ?? 0;
  const winDays         = data?.win_days          ?? 0;
  const loseDays        = data?.lose_days         ?? 0;
  const tradingDays     = data?.trading_days      ?? 0;
  const avgDeployedPct  = data?.avg_deployed_pct  ?? 0;
  const bestDay         = data?.best_day          ?? { date: '', pct: 0 };
  const worstDay        = data?.worst_day         ?? { date: '', pct: 0 };

  const winRate = tradingDays > 0
    ? Math.round((winDays / tradingDays) * 100)
    : 0;

  const returnColor = totalReturnPct >= 0 ? 'var(--positive)' : 'var(--negative)';
  const ddColor     = maxDrawdownPct < -5 ? 'var(--negative)' : maxDrawdownPct < -2 ? 'var(--accent-amber)' : 'var(--text-secondary)';

  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header"><div className="page-title">🤖 投資模擬機器人</div></div>
      <ApiError message={error} onRetry={refetch} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* ── Header ── */}
      <div className="page-header">
        <div>
          <div className="page-title">🤖 投資模擬機器人</div>
          <div className="page-subtitle">
            追蹤假設每日跟隨模型 Top-50 持倉的實際損益 · 基於真實收盤價計算
            {data?.period_start && ` · ${data.period_start} → ${data.period_end}`}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select
            value={viewDays}
            onChange={e => setViewDays(Number(e.target.value))}
            style={{
              background: 'var(--bg-panel-2)', border: '1px solid var(--border)',
              borderRadius: 6, padding: '5px 10px', fontSize: 12,
              color: 'var(--text-primary)', cursor: 'pointer',
            }}
          >
            <option value={0}>全部天數</option>
            <option value={20}>最近 20 天</option>
            <option value={40}>最近 40 天</option>
          </select>
          <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={refetch}>
            🔄 刷新
          </button>
        </div>
      </div>

      {/* ── Summary Cards ── */}
      {loading ? (
        <div className="grid-4">{Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)}</div>
      ) : !hasData ? (
        <div className="panel">
          <div className="panel-body" style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-muted)' }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>📊</div>
            <div style={{ fontSize: 14, marginBottom: 8 }}>尚無回測資料</div>
            <div style={{ fontSize: 12 }}>執行每日推論後會自動生成 · 需要至少 2 個交易日的歷史紀錄</div>
          </div>
        </div>
      ) : (
        <div className="grid-4">
          <StatCard
            label="累積報酬"
            value={`${totalReturnPct >= 0 ? '+' : ''}${totalReturnPct.toFixed(2)}%`}
            sub={`NT$${Math.round(data.initial_capital * totalReturnPct / 100).toLocaleString()} 損益`}
            valueColor={returnColor}
          />
          <StatCard
            label="最大回撤"
            value={`${maxDrawdownPct.toFixed(2)}%`}
            sub="持倉期間最深跌幅"
            valueColor={ddColor}
          />
          <StatCard
            label="年化夏普"
            value={sharpe.toFixed(2)}
            sub={`勝率 ${winRate}%（${winDays}勝 ${loseDays}負）`}
            valueColor={sharpe >= 1 ? 'var(--positive)' : sharpe >= 0.5 ? 'var(--accent-amber)' : 'var(--text-secondary)'}
          />
          <StatCard
            label="平均部署比例"
            value={`${avgDeployedPct.toFixed(1)}%`}
            sub={`共 ${tradingDays} 個交易日`}
            valueColor="var(--accent-blue)"
          />
        </div>
      )}

      {/* ── Equity Curve ── */}
      {hasData && (
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>💹</span> 資產淨值曲線（NT$）</div>
            <div style={{ display: 'flex', gap: 8, fontSize: 11, color: 'var(--text-muted)' }}>
              <span style={{ color: 'var(--positive)' }}>最佳日 {bestDay.date} +{bestDay.pct?.toFixed(2)}%</span>
              <span style={{ color: 'var(--negative)' }}>最差日 {worstDay.date} {worstDay.pct?.toFixed(2)}%</span>
            </div>
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={220} /> : (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={displayCurve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'var(--text-muted)' }}
                    tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                    tickFormatter={v => `${(v / 10000).toFixed(0)}萬`}
                    domain={['auto', 'auto']} />
                  <Tooltip content={<EquityTooltip />} />
                  <ReferenceLine y={data.initial_capital}
                    stroke="var(--border)" strokeDasharray="4 4" strokeWidth={1} />
                  <Line type="monotone" dataKey="equity" name="資產淨值"
                    stroke="var(--accent-blue)" strokeWidth={2} dot={false}
                    activeDot={{ r: 4, fill: 'var(--accent-blue)' }} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      {/* ── Daily Return Bars ── */}
      {hasData && (
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>📊</span> 每日報酬率</div>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>基於實際收盤價計算</span>
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={120} /> : (
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={displayCurve.slice(1)} barCategoryGap="20%">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'var(--text-muted)' }}
                    tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 9, fill: 'var(--text-muted)' }}
                    tickFormatter={v => `${v.toFixed(1)}%`} />
                  <Tooltip content={<ReturnTooltip />} />
                  <ReferenceLine y={0} stroke="var(--border)" />
                  <Bar dataKey="daily_return_pct" name="日報酬率" radius={[2, 2, 0, 0]}>
                    {displayCurve.slice(1).map((pt, i) => (
                      <Cell key={i}
                        fill={pt.daily_return_pct >= 0 ? 'var(--positive)' : 'var(--negative)'}
                        opacity={0.8}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      {/* ── Methodology Note ── */}
      <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.15)', background: 'rgba(0,212,255,0.02)' }}>
        <div className="panel-body" style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          <span style={{ fontSize: 20 }}>💡</span>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            <strong style={{ color: 'var(--accent-blue)' }}>計算方式：</strong>
            假設投資人每日收盤後取得模型推論結果（Top-50 Kelly 加權持倉），
            以當日收盤價建倉，隔日收盤結算損益。
            歷史資料來自 <code style={{ color: 'var(--accent-amber)' }}>prices_raw.parquet</code>（真實收盤價），
            損益計算為 <code style={{ color: 'var(--accent-amber)' }}>Σ(weight × 收盤日報酬)</code>。
            剩餘比例為現金（不計利息）。
            <strong style={{ color: 'var(--accent-amber)', marginLeft: 4 }}>
              ⚠️ 不含交易成本，僅供模型績效參考，不構成投資建議。
            </strong>
          </div>
        </div>
      </div>

      {/* ── Rebalance History ── */}
      <RebalanceHistory />

    </div>
  );
}
