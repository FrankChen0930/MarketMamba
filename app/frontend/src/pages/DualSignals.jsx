import React, { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { fetchDualSignals } from '../api/dual';
import { SkeletonBlock, ApiError } from '../components/SkeletonLoader';

const TH  = { fontSize: 12, color: 'var(--text-muted)', padding: '8px 12px', borderBottom: '1px solid var(--border)', textAlign: 'right' };
const THL = { ...TH, textAlign: 'left' };
const TD  = { fontSize: 13, padding: '8px 12px', borderBottom: '1px solid rgba(48,54,61,0.4)', textAlign: 'right', fontFamily: 'var(--font-mono)' };
const TDL = { ...TD, textAlign: 'left', fontFamily: 'inherit' };

const fmt = (v, d = 4) => (v == null ? '—' : Number(v).toFixed(d));
const cls = (v) => (v > 0 ? 'text-positive' : v < 0 ? 'text-negative' : '');

function Table({ rows, cols }) {
  return (
    <table className="data-table" style={{ width: '100%' }}>
      <thead>
        <tr>
          <th style={TH}>#</th>
          <th style={THL}>股票</th>
          {cols.map((c) => <th key={c.key} style={TH}>{c.label}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.stock_id}>
            <td style={{ ...TD, color: 'var(--text-muted)' }}>{r.rank}</td>
            <td style={TDL}>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>{r.stock_id}</span>
              <span style={{ marginLeft: 8, color: 'var(--text-secondary)' }}>{r.name}</span>
            </td>
            {cols.map((c) => (
              <td key={c.key} style={TD} className={c.color ? cls(r[c.key]) : ''}>{fmt(r[c.key], c.d)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const SHORT_COLS = [
  { key: 'Score_5d',  label: 'Score 5d',  color: true,  d: 4 },
  { key: 'Score_10d', label: 'Score 10d', color: true,  d: 4 },
  { key: 'Unc_5d',    label: '不確定性',  color: false, d: 4 },
  { key: 'SQ_5d',     label: 'SQ',        color: true,  d: 2 },
];
const TREND_COLS = [
  { key: 'Score_20d', label: 'Score 20d', color: true,  d: 4 },
  { key: 'Score_60d', label: 'Score 60d', color: true,  d: 4 },
  { key: 'Unc_20d',   label: '不確定性',  color: false, d: 4 },
  { key: 'SQ_20d',    label: 'SQ',        color: true,  d: 2 },
];

export default function DualSignals() {
  const { data, loading, error, refetch } = useApi(fetchDualSignals);
  const [tab, setTab] = useState('short');

  if (loading) return <div className="panel"><div className="panel-body"><SkeletonBlock /></div></div>;
  if (error)   return <ApiError message={error} onRetry={refetch} />;

  const short = data?.short || [];
  const trend = data?.trend || [];
  const rows  = tab === 'short' ? short : trend;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="panel">
        <div className="panel-header" style={{ justifyContent: 'space-between' }}>
          <div style={{ fontSize: 16, fontWeight: 700 }}>🔀 雙模型訊號</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{data?.date || '—'}</div>
        </div>
        <div className="panel-body">
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14, padding: '10px 12px', background: 'rgba(88,166,255,0.06)', borderRadius: 6, lineHeight: 1.7 }}>
            ⚠️ 這裡的分數是<b>橫斷面排名訊號（rank-score）</b>，<b>不是預測報酬率</b>。分數越高＝模型預測當日相對排名越前面；
            <b> SQ = Score / 不確定性</b>（風險調整後排名），表格依此排序。短線模型主攻 5d/10d、趨勢模型主攻 20d/60d，
            與線上 V6.1 並行、互不影響。
          </div>

          <div style={{ display: 'flex', gap: 4, marginBottom: 14, borderBottom: '1px solid var(--border)' }}>
            {[['short', `短線 5d/10d · ${short.length}`], ['trend', `趨勢 20d/60d · ${trend.length}`]].map(([k, label]) => (
              <button
                key={k}
                onClick={() => setTab(k)}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer', padding: '8px 16px',
                  fontSize: 13, fontWeight: 600,
                  color: tab === k ? 'var(--accent-blue)' : 'var(--text-muted)',
                  borderBottom: tab === k ? '2px solid var(--accent-blue)' : '2px solid transparent',
                  marginBottom: -1,
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {rows.length === 0 ? (
            <div style={{ padding: 28, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
              {data?.note || '尚無資料'}
            </div>
          ) : (
            <Table rows={rows} cols={tab === 'short' ? SHORT_COLS : TREND_COLS} />
          )}
        </div>
      </div>
    </div>
  );
}
