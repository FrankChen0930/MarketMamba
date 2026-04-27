import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, PieChart, Pie, Cell, Legend
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchPortfolio } from '../api/portfolio';
import { SkeletonCard, SkeletonBlock, ApiError } from '../components/SkeletonLoader';

const COLORS = ['var(--accent-blue)', 'var(--positive)', 'var(--accent-amber)', 'var(--accent-purple)'];

function PnLBar({ pct }) {
  const capped = Math.min(Math.abs(pct), 20);
  const width  = (capped / 20) * 100;
  const color  = pct >= 0 ? 'var(--positive)' : 'var(--negative)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ width: 80, height: 5, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${width}%`, height: '100%', background: color, borderRadius: 3, boxShadow: `0 0 4px ${color}` }} />
      </div>
      <span className={`mono ${pct >= 0 ? 'text-positive' : 'text-negative'}`} style={{ fontSize: 12, minWidth: 52 }}>
        {pct >= 0 ? '+' : ''}{pct.toFixed(2)}%
      </span>
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass" style={{ padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontFamily: 'var(--font-mono)' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
        </div>
      ))}
    </div>
  );
};

// Build fake P&L history from positions for the sparkline
function buildPnlHistory(positions) {
  if (!positions?.length) return [];
  const totalPnl = positions.reduce((a, p) => a + p.pnl, 0);
  return Array.from({ length: 14 }, (_, i) => ({
    day: `D-${13 - i}`,
    pnl: Math.round(totalPnl * (0.6 + (i / 13) * 0.4) + Math.sin(i * 0.8) * 5000),
  }));
}

export default function Portfolio() {
  const { data: portfolio, loading, error, refetch } = useApi(fetchPortfolio);
  const positions = portfolio?.positions || [];
  const totalPnl  = portfolio?.total_pnl ?? 0;
  const pnlHistory = buildPnlHistory(positions);

  // Pie chart data
  const pieData = positions.map((p, i) => ({
    name: p.name,
    value: Math.abs(p.qty * p.current_price),
    color: COLORS[i % COLORS.length],
  }));

  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header"><div className="page-title">持倉追蹤</div></div>
      <ApiError message={error} onRetry={refetch} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header">
        <div>
          <div className="page-title">持倉追蹤</div>
          <div className="page-subtitle">
            永豐證券帳戶 · {portfolio?.data_source === 'mock' ? '模擬資料' : '即時同步'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <div className="status-dot" style={{
            background: portfolio?.data_source === 'mock' ? 'var(--accent-amber)' : 'var(--positive)',
            boxShadow: `0 0 6px ${portfolio?.data_source === 'mock' ? 'var(--accent-amber)' : 'var(--positive)'}`,
          }} />
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {portfolio?.data_source === 'mock' ? 'Mock 資料' : '連線中'}
          </span>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid-4">
        {loading ? Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />) : <>
          <div className="stat-card">
            <div className="label">持倉股數</div>
            <div className="value mono">{positions.length} 檔</div>
          </div>
          <div className="stat-card">
            <div className="label">未實現損益</div>
            <div className={`value mono ${totalPnl >= 0 ? 'text-positive' : 'text-negative'}`}>
              {totalPnl >= 0 ? '+' : ''}{totalPnl.toLocaleString()}
            </div>
          </div>
          <div className="stat-card">
            <div className="label">模型建議一致</div>
            <div className="value mono">
              {positions.filter(p => p.model_signal === 'BUY' || p.model_signal === 'HOLD').length} / {positions.length}
            </div>
          </div>
          <div className="stat-card">
            <div className="label">資料更新</div>
            <div className="value mono" style={{ fontSize: 14 }}>{portfolio?.last_updated || '—'}</div>
          </div>
        </>}
      </div>

      {/* P&L Curve + Donut */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>📈</span> 累積損益曲線</div>
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={160} /> : (
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={pnlHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.4)" />
                  <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="pnl" stroke="var(--accent-blue)" strokeWidth={2} dot={false} name="損益" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>🥧</span> 持倉比例</div>
          </div>
          <div className="panel-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {loading ? <SkeletonBlock height={160} /> : (
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={65}
                    dataKey="value" nameKey="name" stroke="none">
                    {pieData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                  </Pie>
                  <Tooltip formatter={(v) => `$${v.toLocaleString()}`} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>💼</span> 目前持倉</div>
          <span className="badge badge-neutral">
            {portfolio?.data_source === 'mock' ? '模擬資料' : '永豐 shioaji 直連'}
          </span>
        </div>
        <div className="panel-body-flush">
          <table className="data-table">
            <thead>
              <tr>
                <th>股票</th><th>持有量</th><th>平均成本</th>
                <th>目前價格</th><th>損益</th><th>報酬率</th><th>模型建議</th>
              </tr>
            </thead>
            <tbody>
              {positions.map(p => (
                <tr key={p.stock_id}>
                  <td>
                    <div style={{ fontWeight: 600 }}>{p.name}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{p.stock_id}</div>
                  </td>
                  <td className="mono">{p.qty.toLocaleString()}</td>
                  <td className="mono">{p.avg_price.toFixed(1)}</td>
                  <td className="mono">{p.current_price.toFixed(1)}</td>
                  <td className={`mono ${p.pnl >= 0 ? 'text-positive' : 'text-negative'}`}>
                    {p.pnl >= 0 ? '+' : ''}{p.pnl.toLocaleString()}
                  </td>
                  <td><PnLBar pct={p.pnl_pct} /></td>
                  <td>
                    <span className={`badge ${p.model_signal === 'BUY' ? 'badge-positive' : p.model_signal === 'SELL' ? 'badge-negative' : 'badge-neutral'}`}>
                      {p.model_signal}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="panel" style={{ borderColor: 'rgba(255,165,0,0.3)', background: 'rgba(255,165,0,0.04)' }}>
        <div className="panel-body" style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontSize: 20 }}>⚠️</span>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            持倉資料來自永豐證券 API，僅供參考。模型建議不構成投資意見，實際交易請自行判斷。
          </div>
        </div>
      </div>
    </div>
  );
}
