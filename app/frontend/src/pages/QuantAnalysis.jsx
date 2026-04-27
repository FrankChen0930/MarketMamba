import React from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, CartesianGrid, Legend
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchPerformance } from '../api/performance';
import { SkeletonBlock, SkeletonCard, ApiError } from '../components/SkeletonLoader';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass" style={{ padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontFamily: 'var(--font-mono)' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
        </div>
      ))}
    </div>
  );
};

function MetricChip({ label, value, positive }) {
  return (
    <div className="stat-card" style={{ padding: '12px 16px' }}>
      <div className="label">{label}</div>
      <div className={`value mono ${positive !== undefined ? (positive ? 'text-positive' : 'text-negative') : ''}`}
        style={{ fontSize: 18 }}>
        {value}
      </div>
    </div>
  );
}

// Monthly return heatmap (12 months × years from cumret data)
function MonthlyReturnHeatmap({ cumret }) {
  if (!cumret?.length) return null;
  const months = ['01','02','03','04','05','06','07','08','09','10','11','12'];
  const byMonth = {};
  cumret.forEach((pt, i) => {
    const ret = i === 0 ? 0 : (pt.model - cumret[i - 1].model) / cumret[i - 1].model;
    byMonth[pt.month] = ret;
  });

  const entries = Object.entries(byMonth);
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.01);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 3 }}>
      {months.map(m => {
        const key = Object.keys(byMonth).find(k => k.endsWith(`-${m}`));
        const val = key ? byMonth[key] : null;
        const opacity = val !== null ? Math.min(Math.abs(val) / maxAbs, 1) : 0;
        const color = val === null ? 'var(--bg-hover)' : val >= 0
          ? `rgba(0,255,136,${0.1 + opacity * 0.7})`
          : `rgba(255,71,87,${0.1 + opacity * 0.7})`;
        return (
          <div key={m} title={key ? `${key}: ${(val * 100).toFixed(2)}%` : m}
            style={{
              height: 28, borderRadius: 4, background: color,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 9, color: 'var(--text-muted)', cursor: 'default',
            }}>
            {m}
          </div>
        );
      })}
    </div>
  );
}

export default function QuantAnalysis() {
  const { data: perf, loading, error, refetch } = useApi(fetchPerformance);

  const wfFolds  = perf?.wf_folds  || [];
  const icHist   = perf?.ic_history || [];
  const cumret   = perf?.cumret     || [];

  const avgIC     = wfFolds.length ? (wfFolds.reduce((a, f) => a + f.ic,    0) / wfFolds.length).toFixed(4) : '—';
  const avgICIR   = wfFolds.length ? (wfFolds.reduce((a, f) => a + f.icir,  0) / wfFolds.length).toFixed(2) : '—';
  const avgSharpe = wfFolds.length ? (wfFolds.reduce((a, f) => a + f.sharpe,0) / wfFolds.length).toFixed(2) : '—';

  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header"><div className="page-title">量化分析</div></div>
      <ApiError message={error} onRetry={refetch} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header">
        <div>
          <div className="page-title">量化分析</div>
          <div className="page-subtitle">純模型數據 · 不含 LLM 消息面整合</div>
        </div>
        {perf?.training_status === 'training' && (
          <span className="badge badge-blue">⏳ 訓練中 — Walk-Forward 為模擬資料</span>
        )}
      </div>

      {/* KPIs */}
      <div className="grid-4">
        {loading ? Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />) : <>
          <MetricChip label="平均 IC (Walk-Forward)" value={`+${avgIC}`} positive={true} />
          <MetricChip label="平均 ICIR" value={avgICIR} positive={parseFloat(avgICIR) > 0.5} />
          <MetricChip label="平均年化 Sharpe" value={avgSharpe} positive={parseFloat(avgSharpe) > 1} />
          <MetricChip label="Walk-Forward Folds" value={`${wfFolds.length} 期`} />
        </>}
      </div>

      {/* IC Training Curve */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>📈</span> 訓練 IC 曲線</div>
          <span className="badge badge-blue">
            {perf?.training_status === 'training' ? '目前訓練中' : `Best Epoch ${perf?.best_epoch}`}
          </span>
        </div>
        <div className="panel-body">
          {loading ? <SkeletonBlock height={200} /> : (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={icHist}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
                <ReferenceLine y={0.05} stroke="rgba(0,255,136,0.4)" strokeDasharray="5 5"
                  label={{ value: 'IC=0.05', fill: 'var(--positive)', fontSize: 10 }} />
                <Line type="monotone" dataKey="train_loss" stroke="var(--accent-red)"   strokeWidth={1.5} dot={false} name="Train Loss" />
                <Line type="monotone" dataKey="val_loss"   stroke="var(--accent-amber)" strokeWidth={1.5} dot={false} name="Val Loss" />
                <Line type="monotone" dataKey="val_ic"     stroke="var(--accent-blue)"  strokeWidth={2}   dot={false} name="Val IC" />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Walk-Forward + Cumulative Return */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>🔄</span> Walk-Forward IC 分佈</div>
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={200} /> : (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={wfFolds}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="fold" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={[0, 0.1]} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={0.05} stroke="rgba(0,255,136,0.4)" strokeDasharray="4 4" />
                  <Bar dataKey="ic" name="IC" fill="var(--accent-blue)" radius={[3, 3, 0, 0]} fillOpacity={0.8} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>💹</span> 累積報酬 vs 大盤</div>
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={200} /> : (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={cumret}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="month" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} interval={5} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-secondary)' }} />
                  <Line type="monotone" dataKey="model"     stroke="var(--accent-blue)"  strokeWidth={2} dot={false} name="Model" />
                  <Line type="monotone" dataKey="benchmark" stroke="var(--text-muted)"   strokeWidth={1} dot={false} name="TAIEX" strokeDasharray="4 4" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Monthly Heatmap */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>🗓</span> 月度報酬分佈</div>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>綠 = 正報酬 / 紅 = 負報酬</span>
        </div>
        <div className="panel-body">
          {loading ? <SkeletonBlock height={60} /> : <MonthlyReturnHeatmap cumret={cumret} />}
        </div>
      </div>

      {/* Walk-Forward Table */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>📋</span> Walk-Forward 詳細結果</div>
        </div>
        <div className="panel-body-flush">
          <table className="data-table">
            <thead>
              <tr>
                <th>期數</th><th>時間段</th><th>IC</th><th>ICIR</th>
                <th>年化 Sharpe</th><th>超額報酬</th><th>評級</th>
              </tr>
            </thead>
            <tbody>
              {wfFolds.map(f => (
                <tr key={f.fold}>
                  <td className="text-muted">{f.fold}</td>
                  <td>{f.period}</td>
                  <td className={f.ic >= 0.05 ? 'text-positive' : 'text-secondary'}>+{f.ic.toFixed(3)}</td>
                  <td className={f.icir >= 0.5 ? 'text-positive' : 'text-secondary'}>{f.icir.toFixed(2)}</td>
                  <td className={f.sharpe >= 1 ? 'text-positive' : 'text-secondary'}>{f.sharpe.toFixed(2)}</td>
                  <td className="text-positive">+{(f.ret * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`badge ${f.ic >= 0.05 ? 'badge-positive' : 'badge-neutral'}`}>
                      {f.ic >= 0.05 ? '✓ 達標' : '觀察中'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
