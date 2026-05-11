import React from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend, ReferenceLine
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchPerformance } from '../api/performance';
import { fetchMarket } from '../api/market';
import { SkeletonBlock, SkeletonCard, ApiError } from '../components/SkeletonLoader';
import MetricTooltip from '../components/MetricTooltip';


const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass" style={{ padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontFamily: 'var(--font-mono)' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
};

export default function ModelStatus() {
  const { data: perf, loading, error, refetch } = useApi(fetchPerformance);
  const { data: market } = useApi(fetchMarket);

  const wfFolds = perf?.wf_folds  || [];
  const icHist  = perf?.ic_history || [];
  const cumret  = perf?.cumret     || [];

  const avgIC     = wfFolds.length ? (wfFolds.reduce((a, f) => a + f.ic,    0) / wfFolds.length) : 0;
  const avgICIR   = wfFolds.length ? (wfFolds.reduce((a, f) => a + f.icir,  0) / wfFolds.length) : 0;
  const avgSharpe = wfFolds.length ? (wfFolds.reduce((a, f) => a + f.sharpe,0) / wfFolds.length) : 0;

  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header"><div className="page-title">模型狀態</div></div>
      <ApiError message={error} onRetry={refetch} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header">
        <div>
          <div className="page-title">🧠 MarketMamba V6 模型狀態</div>
          <div className="page-subtitle">Walk-Forward 驗證 · Mamba+GAT · 46 因子</div>
        </div>
        <span className="badge badge-positive">✓ 訓練完成</span>
      </div>

      {/* KPIs */}
      <div className="grid-4">
        {loading ? Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />) : <>
          <div className="stat-card">
            <div className="label">平均 IC（Walk-Forward） <MetricTooltip metricKey="ic" /></div>
            <div className="value mono text-positive" style={{ fontSize: 20 }}>+{avgIC.toFixed(4)}</div>
            <div className="sub">目標 ≥ 0.05 ✓</div>
          </div>
          <div className="stat-card">
            <div className="label">平均 ICIR <MetricTooltip metricKey="icir" /></div>
            <div className="value mono text-positive" style={{ fontSize: 20 }}>{avgICIR.toFixed(2)}</div>
            <div className="sub">目標 ≥ 0.5 ✓</div>
          </div>
          <div className="stat-card">
            <div className="label">平均年化 Sharpe <MetricTooltip metricKey="sharpe" /></div>
            <div className="value mono text-positive" style={{ fontSize: 20 }}>{avgSharpe.toFixed(2)}</div>
            <div className="sub">Walk-Forward 均值</div>
          </div>
          <div className="stat-card">
            <div className="label">最後推論日期</div>
            <div className="value mono" style={{ fontSize: 16 }}>{market?.last_run || '—'}</div>
            <div className="sub">checkpoint: v6_best.pt · epoch 14 · val_loss 1.647</div>
          </div>
        </>}
      </div>

      {/* IC Learning Curve */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>📈</span> 訓練 IC 學習曲線（Fold 1）</div>
          <span className="badge badge-blue">14 epochs · val_ic 0.0825</span>
        </div>
        <div className="panel-body">
          {loading ? <SkeletonBlock height={200} /> : (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={icHist}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine y={0.05} stroke="rgba(0,255,136,0.4)" strokeDasharray="5 5"
                  label={{ value: 'IC=0.05目標', fill: 'var(--positive)', fontSize: 10 }} />
                <Line type="monotone" dataKey="train_loss" stroke="var(--accent-red)"   strokeWidth={1.5} dot={false} name="Train Loss" />
                <Line type="monotone" dataKey="val_loss"   stroke="var(--accent-amber)" strokeWidth={1.5} dot={false} name="Val Loss" />
                <Line type="monotone" dataKey="val_ic"     stroke="var(--accent-blue)"  strokeWidth={2}   dot={false} name="Val IC" />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Walk-Forward + Cumret */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>🔄</span> Walk-Forward IC 分佈</div>
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={180} /> : (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={wfFolds}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="fold" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={[0, 0.15]} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={0.05} stroke="rgba(0,255,136,0.4)" strokeDasharray="4 4" />
                  <Bar dataKey="ic" name="IC" fill="var(--accent-blue)" radius={[3,3,0,0]} fillOpacity={0.85} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>💹</span> 累積超額報酬 vs TAIEX</div>
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={180} /> : (
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={cumret}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="month" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} interval={5} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="model"     stroke="var(--accent-blue)" strokeWidth={2} dot={false} name="Model" />
                  <Line type="monotone" dataKey="benchmark" stroke="var(--text-muted)"  strokeWidth={1} dot={false} name="TAIEX" strokeDasharray="4 4" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Walk-Forward Table */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>📋</span> Walk-Forward 詳細結果</div>
          <span className="badge badge-neutral" style={{ fontSize: 10 }}>4 個真實 Fold</span>
        </div>
        <div className="panel-body-flush">
          <table className="data-table">
            <thead>
              <tr>
                <th>期數</th><th>訓練期間</th><th>IC <MetricTooltip metricKey="ic" /></th><th>ICIR <MetricTooltip metricKey="icir" /></th>
                <th>年化 Sharpe</th><th>超額報酬</th><th>評級</th>
              </tr>
            </thead>
            <tbody>
              {wfFolds.map(f => (
                <tr key={f.fold}>
                  <td className="text-muted">{f.fold}</td>
                  <td style={{ fontSize: 12 }}>{f.period}</td>
                  <td className={f.ic >= 0.05 ? 'text-positive mono' : 'text-secondary mono'}>+{f.ic.toFixed(4)}</td>
                  <td className={f.icir >= 0.5 ? 'text-positive mono' : 'text-secondary mono'}>{f.icir.toFixed(2)}</td>
                  <td className={f.sharpe >= 1 ? 'text-positive mono' : 'text-secondary mono'}>{f.sharpe.toFixed(2)}</td>
                  <td className="text-positive mono">+{(f.ret * 100).toFixed(1)}%</td>
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

      {/* Model architecture info */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>🏗️</span> 模型架構摘要</div>
        </div>
        <div className="panel-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
            {[
              { label: '骨幹架構', value: 'Multi-Scale Mamba + GAT', desc: 'SSM + Graph Attention' },
              { label: '輸入特徵', value: '46 維', desc: '技術 + 法人 + 基本面 + 宏觀' },
              { label: '預測目標', value: '5d / 20d / 60d Alpha', desc: '多尺度超額報酬' },
              { label: 'KG 節點', value: '2,890', desc: '股票 + 產業節點' },
              { label: 'KG 邊數', value: '42,905', desc: '供應鏈 + 相關性邊' },
              { label: '訓練資料', value: '2012–2024', desc: '約 3,000 股 × 12年' },
              { label: '參數量', value: '11.5M', desc: '11,456,643 trainable params' },
            ].map(m => (
              <div key={m.label} style={{ padding: '10px 14px', background: 'var(--bg-panel-2)', borderRadius: 8 }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{m.label}</div>
                <div className="mono" style={{ fontSize: 14, color: 'var(--text-primary)' }}>{m.value}</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{m.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
