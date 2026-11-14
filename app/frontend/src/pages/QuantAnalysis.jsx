import React, { useState, useEffect } from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell, Legend,
  LineChart, Line,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchSignals } from '../api/signals';
import { fetchQuant, fetchPatterns } from '../api/quant';
import { SkeletonBlock, SkeletonCard, SkeletonTable, ApiError } from '../components/SkeletonLoader';
import MetricTooltip from '../components/MetricTooltip';


// ── Tiny helpers ──────────────────────────────────────────────────────────────

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass" style={{ padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
        </div>
      ))}
    </div>
  );
};

function KpiCard({ label, value, desc, valueColor, loading }) {
  if (loading) return <SkeletonCard />;
  return (
    <div className="stat-card">
      <div className="label">{label}</div>
      <div className="value mono" style={{ fontSize: 18, color: valueColor || 'var(--text-primary)' }}>{value ?? '—'}</div>
      {desc && <div className="sub">{desc}</div>}
    </div>
  );
}

// Pattern type icon map
const PAT_ICON = { w_bottom: 'W', spring_w: '⚡', hs_bottom: '⛰️', triangle: '△' };
const PAT_ICONS_COLOR = 'var(--positive)';  // all bullish

// Score bar
function ScoreBar({ score }) {
  const color = score >= 85 ? 'var(--positive)' : score >= 70 ? 'var(--accent-amber)' : 'var(--text-muted)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ width: 48, height: 5, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${score}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <span className="mono" style={{ fontSize: 11, color }}>{score}</span>
    </div>
  );
}


// ── Main Component ────────────────────────────────────────────────────────────

export default function QuantAnalysis() {
  const [tab, setTab] = useState('tech');

  // Signals (for model-alpha tab)
  const { data: signalData }  = useApi(fetchSignals);
  // Quant market data (tech / chip / breadth tabs)
  const { data: quant, loading: quantLoading, error: quantError } = useApi(fetchQuant);

  // Pattern data: lazy-load only when tab is active
  const [patternData, setPatternData]       = useState(null);
  const [patternLoading, setPatternLoading] = useState(false);
  const [patternError, setPatternError]     = useState(null);

  useEffect(() => {
    if (tab === 'pattern' && !patternData && !patternLoading) {
      setPatternLoading(true);
      fetchPatterns()
        .then(d => { setPatternData(d); setPatternLoading(false); })
        .catch(e => { setPatternError(e.message); setPatternLoading(false); });
    }
  }, [tab]);

  // ── Derived data ────────────────────────────────────────────────────────────
  const signals    = signalData?.signals || [];
  const tech       = quant?.taiex_technicals || {};
  const techTable  = tech.table          || [];
  const techRadar  = tech.tech_radar     || [];
  const riskMetrics = tech.risk_metrics  || [];

  const instSummary  = quant?.institutional_summary || {};
  const breadthHist  = quant?.breadth_history       || [];
  const marginSummary = quant?.margin_summary       || {};
  const sectorFlow    = quant?.sector_flow          || [];

  const lastBreadth   = breadthHist.at(-1) || {};
  const maxFlow       = Math.max(...sectorFlow.map(s => Math.abs(s.net_5d_bn || 0)), 1);

  // Sector alpha from model signals
  const sectorAlpha = Object.values(
    signals.reduce((acc, s) => {
      if (!acc[s.sector]) acc[s.sector] = { sector: s.sector, count: 0, totalAlpha: 0 };
      acc[s.sector].count++;
      acc[s.sector].totalAlpha += (s.alpha_20d || 0);
      return acc;
    }, {})
  )
    .map(g => ({ sector: g.sector, avgAlpha: g.totalAlpha / g.count * 100 }))
    .sort((a, b) => b.avgAlpha - a.avgAlpha)
    .slice(0, 8);

  const TABS = [
    { id: 'tech',    label: '技術指標' },
    { id: 'chip',    label: '籌碼面'   },
    { id: 'breadth', label: '市場廣度' },
    { id: 'model',   label: '模型 Alpha' },
    { id: 'pattern', label: '傳統型態學' },
  ];

  const dataDate = quant?.date || signalData?.date || '—';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* ── Page Header ── */}
      <div className="page-header">
        <div>
          <div className="page-title">量化分析儀表板</div>
          <div className="page-subtitle">
            技術面 · 籌碼面 · 市場廣度 · Alpha 因子 · 型態掃描 · 資料日期：{dataDate}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {TABS.map(t => (
            <button key={t.id}
              className={`btn ${tab === t.id ? 'btn-primary' : 'btn-ghost'}`}
              style={{ padding: '5px 14px', fontSize: 12 }}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── KPI Row (live data) ── */}
      <div className="grid-4">
        <KpiCard
          loading={quantLoading}
          label={<>加權指數 RSI(14) <MetricTooltip metricKey="rsi" /></>}
          value={tech.rsi_14 != null ? String(tech.rsi_14) : '—'}
          desc={tech.rsi_14 > 70 ? '⚠️ 超買' : tech.rsi_14 < 30 ? '🔥 超賣' : '中性區間 (30–70)'}
          valueColor={tech.rsi_14 > 70 ? 'var(--negative)' : tech.rsi_14 < 30 ? 'var(--positive)' : 'var(--accent-amber)'}
        />
        <KpiCard
          loading={quantLoading}
          label={<>外資近5日淨買 <MetricTooltip metricKey="foreign_net" /></>}
          value={instSummary.foreign_net_5d_bn != null
            ? `${instSummary.foreign_net_5d_bn >= 0 ? '+' : ''}${instSummary.foreign_net_5d_bn?.toFixed(1)}億`
            : '—'}
          desc={instSummary.foreign_net_5d_bn >= 0 ? '外資買超' : '外資賣超'}
          valueColor={instSummary.foreign_net_5d_bn >= 0 ? 'var(--positive)' : 'var(--negative)'}
        />
        <KpiCard
          loading={quantLoading}
          label={<>漲跌比 (最新) <MetricTooltip metricKey="ad_ratio" /></>}
          value={lastBreadth.adv
            ? `${lastBreadth.adv}:${lastBreadth.dec}`
            : '—'}
          desc={lastBreadth.ratio != null ? `A/D Ratio ${lastBreadth.ratio?.toFixed(2)}` : ''}
          valueColor={(lastBreadth.ratio || 0) >= 1 ? 'var(--positive)' : 'var(--negative)'}
        />
        <KpiCard
          loading={quantLoading}
          label={<>融資餘額 <MetricTooltip metricKey="margin_balance" /></>}
          value={marginSummary.margin_balance_bn != null
            ? `${marginSummary.margin_balance_bn?.toFixed(0)}億`
            : '—'}
          desc={marginSummary.margin_ratio != null
            ? `資券比 ${marginSummary.margin_ratio}x`
            : '信用交易'}
          valueColor="var(--accent-amber)"
        />
      </div>


      {/* ══════════════════════════════════════════════════════════════════════
          技術指標 Tab
      ══════════════════════════════════════════════════════════════════════ */}
      {tab === 'tech' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

          {/* Tech table */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>📊</span> 大盤技術指標</div>
              <span className="badge badge-neutral" style={{ fontSize: 10 }}>TAIEX ^TWII</span>
            </div>
            <div className="panel-body-flush">
              {quantLoading ? <SkeletonTable rows={8} cols={3} /> : quantError ? (
                <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>
                  資料暫不可用：{quantError}
                </div>
              ) : (
                <table className="data-table">
                  <thead><tr><th>指標</th><th>數值</th><th>狀態</th></tr></thead>
                  <tbody>
                    {techTable.map(r => (
                      <tr key={r.label}>
                        <td style={{ color: 'var(--text-secondary)' }}>{r.label}</td>
                        <td className="mono" style={{ color: 'var(--text-primary)' }}>{r.value}</td>
                        <td><span style={{ fontSize: 11, color: r.color }}>{r.status}</span></td>
                      </tr>
                    ))}
                    {techTable.length === 0 && (
                      <tr><td colSpan={3} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 20 }}>
                        暫無資料（每日推論後生成）
                      </td></tr>
                    )}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* Radar */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🕸️</span> 技術面評分雷達</div>
            </div>
            <div className="panel-body" style={{ display: 'flex', justifyContent: 'center' }}>
              {quantLoading ? <SkeletonBlock height={250} /> : (
                <ResponsiveContainer width="100%" height={250}>
                  <RadarChart data={techRadar.length > 0 ? techRadar : [
                    { subject: 'RSI 動能', A: 50 }, { subject: 'MACD 趨勢', A: 50 },
                    { subject: 'KD 隨機', A: 50 },  { subject: 'Bollinger', A: 50 },
                    { subject: 'MA 排列', A: 50 },  { subject: 'ATR 波動', A: 50 },
                  ]}>
                    <PolarGrid stroke="var(--border)" />
                    <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                    <Radar name="評分" dataKey="A" stroke="var(--accent-blue)"
                      fill="var(--accent-blue)" fillOpacity={0.15} />
                  </RadarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Risk metrics */}
          <div className="panel" style={{ gridColumn: '1 / -1' }}>
            <div className="panel-header">
              <div className="panel-title"><span>⚠️</span> 風險與槓桿指標</div>
            </div>
            <div className="panel-body">
              {quantLoading ? <SkeletonBlock height={80} /> : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                  {(riskMetrics.length > 0 ? riskMetrics : [
                    { label: '10日實現波動率', value: '—', desc: '年化' },
                    { label: '台股 Beta (vs SPX)', value: '—', desc: '近60日' },
                    { label: 'MA 20', value: '—', desc: '20日均線' },
                    { label: 'MA 60', value: '—', desc: '60日均線' },
                  ]).map(r => (
                    <div key={r.label} style={{
                      padding: '10px 14px', background: 'var(--bg-panel-2)', borderRadius: 8
                    }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{r.label}</div>
                      <div className="mono" style={{ fontSize: 16, color: 'var(--text-primary)' }}>{r.value}</div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{r.desc}</div>
                    </div>
                  ))}
                  {/* Margin from live summary */}
                  {marginSummary.margin_balance_bn != null && (
                    <>
                      <div style={{ padding: '10px 14px', background: 'var(--bg-panel-2)', borderRadius: 8 }}>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                          融資餘額 <MetricTooltip metricKey="margin_balance" />
                        </div>
                        <div className="mono" style={{ fontSize: 16, color: 'var(--accent-amber)' }}>
                          {marginSummary.margin_balance_bn?.toFixed(0)}億
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{quant?.date}</div>
                      </div>
                      <div style={{ padding: '10px 14px', background: 'var(--bg-panel-2)', borderRadius: 8 }}>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                          融券餘額 <MetricTooltip metricKey="short_balance" />
                        </div>
                        <div className="mono" style={{ fontSize: 16, color: 'var(--text-muted)' }}>
                          {marginSummary.short_balance_bn?.toFixed(0)}億
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>資券比 {marginSummary.margin_ratio}x</div>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}


      {/* ══════════════════════════════════════════════════════════════════════
          籌碼面 Tab
      ══════════════════════════════════════════════════════════════════════ */}
      {tab === 'chip' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

          {/* 三大法人 */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🏦</span> 三大法人近5日</div>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>億台幣</span>
            </div>
            <div className="panel-body-flush">
              {quantLoading ? <SkeletonTable rows={4} cols={2} /> : (
                <table className="data-table">
                  <thead><tr><th>法人</th><th>近5日淨值</th></tr></thead>
                  <tbody>
                    {[
                      { name: '外資',   net: instSummary.foreign_net_5d_bn         },
                      { name: '投信',   net: instSummary.investment_trust_net_5d_bn },
                      { name: '自營商', net: instSummary.dealer_net_5d_bn           },
                    ].map(r => (
                      <tr key={r.name}>
                        <td style={{ fontWeight: 600 }}>{r.name}</td>
                        <td className={`mono ${(r.net || 0) >= 0 ? 'text-positive' : 'text-negative'}`}>
                          {r.net != null ? `${r.net >= 0 ? '+' : ''}${r.net.toFixed(1)}億` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {/* 5-day history */}
              {!quantLoading && instSummary.history?.length > 0 && (
                <div style={{ padding: '8px 16px' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6 }}>外資每日淨買（億）</div>
                  <div style={{ display: 'flex', gap: 4 }}>
                    {instSummary.history.map(h => {
                      const v = h.foreign_net;
                      const color = v >= 0 ? 'var(--positive)' : 'var(--negative)';
                      return (
                        <div key={h.date} style={{ flex: 1, textAlign: 'center' }}>
                          <div className="mono" style={{ fontSize: 10, color }}>{v >= 0 ? '+' : ''}{v?.toFixed(0)}</div>
                          <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>{h.date_label}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 產業資金輪動 */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🔄</span> 產業資金輪動</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>外資近5日（億）</span>
            </div>
            <div className="panel-body">
              {quantLoading ? <SkeletonBlock height={160} /> : sectorFlow.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>暫無資料</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {sectorFlow.map(s => {
                    const pct = Math.abs(s.net_5d_bn) / maxFlow * 100;
                    const color = s.net_5d_bn >= 0 ? 'var(--positive)' : 'var(--negative)';
                    return (
                      <div key={s.sector}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                          <span style={{ color: 'var(--text-secondary)' }}>{s.sector}</span>
                          <span className="mono" style={{ color }}>
                            {s.net_5d_bn >= 0 ? '+' : ''}{s.net_5d_bn?.toFixed(1)}億
                          </span>
                        </div>
                        <div style={{ height: 5, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* 融資融券 */}
          <div className="panel" style={{ gridColumn: '1 / -1' }}>
            <div className="panel-header">
              <div className="panel-title"><span>📉</span> 融資融券概況</div>
              <span className="badge badge-neutral" style={{ fontSize: 10 }}>信用交易</span>
            </div>
            <div className="panel-body">
              {quantLoading ? <SkeletonBlock height={60} /> : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, textAlign: 'center' }}>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                      融資餘額 <MetricTooltip metricKey="margin_balance" />
                    </div>
                    <div className="mono" style={{ fontSize: 20, color: 'var(--accent-amber)' }}>
                      {marginSummary.margin_balance_bn != null ? `${marginSummary.margin_balance_bn?.toFixed(0)}億` : '—'}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>{quant?.date}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                      融券餘額 <MetricTooltip metricKey="short_balance" />
                    </div>
                    <div className="mono" style={{ fontSize: 20, color: 'var(--text-muted)' }}>
                      {marginSummary.short_balance_bn != null ? `${marginSummary.short_balance_bn?.toFixed(0)}億` : '—'}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>{quant?.date}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                      資券比 <MetricTooltip metricKey="margin_ratio" />
                    </div>
                    <div className="mono" style={{
                      fontSize: 20,
                      color: (marginSummary.margin_ratio || 0) > 10 ? 'var(--accent-amber)' : 'var(--positive)',
                    }}>
                      {marginSummary.margin_ratio != null ? `${marginSummary.margin_ratio}x` : '—'}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                      {(marginSummary.margin_ratio || 0) > 12 ? '偏高，留意軋空' : '正常範圍'}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}


      {/* ══════════════════════════════════════════════════════════════════════
          市場廣度 Tab
      ══════════════════════════════════════════════════════════════════════ */}
      {tab === 'breadth' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>📶</span> 漲跌家數（近5日）</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>市場廣度</span>
            </div>
            <div className="panel-body">
              {quantLoading ? <SkeletonBlock height={200} /> : breadthHist.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>暫無資料</div>
              ) : (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={breadthHist} barCategoryGap="30%">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                    <XAxis dataKey="date_label" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="adv" name="上漲" fill="var(--positive)" fillOpacity={0.8} radius={[3,3,0,0]} />
                    <Bar dataKey="dec" name="下跌" fill="var(--negative)" fillOpacity={0.8} radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>📈</span> 漲跌比（A/D Ratio）趨勢</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>&gt;1 = 多方市場廣度</span>
            </div>
            <div className="panel-body">
              {quantLoading ? <SkeletonBlock height={160} /> : (
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={breadthHist}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                    <XAxis dataKey="date_label" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={[0.5, 'auto']} />
                    <Tooltip content={<CustomTooltip />} />
                    <Line type="monotone" dataKey="ratio" name="A/D Ratio"
                      stroke="var(--accent-blue)" strokeWidth={2} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Breadth KPI cards from live data */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
            {quantLoading ? [1,2].map(i => <SkeletonCard key={i} />) : <>
              <div className="stat-card">
                <div className="label">今日上漲</div>
                <div className="value mono text-positive">{lastBreadth.adv ?? '—'} 家</div>
                <div className="sub">{lastBreadth.date_label || lastBreadth.date}</div>
              </div>
              <div className="stat-card">
                <div className="label">今日下跌</div>
                <div className="value mono text-negative">{lastBreadth.dec ?? '—'} 家</div>
                <div className="sub">A/D Ratio {lastBreadth.ratio?.toFixed(2) ?? '—'}</div>
              </div>
            </>}
          </div>
        </div>
      )}


      {/* ══════════════════════════════════════════════════════════════════════
          模型 Alpha Tab
      ══════════════════════════════════════════════════════════════════════ */}
      {tab === 'model' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <span>🎯</span> 產業平均 Alpha <MetricTooltip metricKey="sector_alpha" />（模型輸出）
              </div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                基於 {signals.length} 檔股票
              </span>
            </div>
            <div className="panel-body">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={sectorAlpha} layout="vertical">
                  <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                    tickFormatter={v => `${v.toFixed(1)}%`} />
                  <YAxis type="category" dataKey="sector"
                    tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} width={80} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="avgAlpha" name="平均Alpha%" radius={[0,3,3,0]}>
                    {sectorAlpha.map((entry, i) => (
                      <Cell key={i} fill={entry.avgAlpha >= 0 ? 'var(--positive)' : 'var(--negative)'}
                        fillOpacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="panel">
              <div className="panel-header">
                <div className="panel-title"><span>📋</span> 信心分佈 <MetricTooltip metricKey="confidence_dist" /></div>
              </div>
              <div className="panel-body">
                {['高信心', '中信心', '低信心'].map(conf => {
                  const cnt = signals.filter(s => s.confidence === conf).length;
                  const pct = signals.length ? (cnt / signals.length * 100) : 0;
                  const color = conf === '高信心' ? 'var(--positive)' : conf === '中信心' ? 'var(--accent-amber)' : 'var(--negative)';
                  return (
                    <div key={conf} style={{ marginBottom: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 5 }}>
                        <span style={{ color }}>{conf}</span>
                        <span className="mono" style={{ color: 'var(--text-muted)' }}>{cnt} 檔 ({pct.toFixed(1)}%)</span>
                      </div>
                      <div style={{ height: 6, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="panel">
              <div className="panel-header">
                <div className="panel-title"><span>⚖️</span> Alpha 分佈統計</div>
              </div>
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {(() => {
                  const alphas = signals.map(s => s.alpha_20d || 0);
                  const mean = alphas.length ? alphas.reduce((a, b) => a + b, 0) / alphas.length : 0;
                  const pos = alphas.filter(a => a > 0).length;
                  const neg = alphas.filter(a => a < 0).length;
                  const maxA = alphas.length ? Math.max(...alphas) : 0;
                  return [
                    { label: '多空比', value: `${pos} : ${neg}`, color: 'var(--positive)' },
                    { label: '平均 Alpha', value: `${mean >= 0 ? '+' : ''}${(mean * 100).toFixed(2)}%`, color: mean >= 0 ? 'var(--positive)' : 'var(--negative)' },
                    { label: '最強 Alpha', value: `+${(maxA * 100).toFixed(2)}%`, color: 'var(--accent-amber)' },
                    { label: '可投資股數', value: `${pos} / ${alphas.length}`, color: 'var(--text-primary)' },
                  ].map(m => (
                    <div key={m.label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span style={{ color: 'var(--text-muted)' }}>{m.label}</span>
                      <span className="mono" style={{ color: m.color }}>{m.value}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>
        </div>
      )}


      {/* ══════════════════════════════════════════════════════════════════════
          傳統型態學 Tab  ← NOW LIVE DATA
      ══════════════════════════════════════════════════════════════════════ */}
      {tab === 'pattern' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Summary KPI row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
            {patternLoading ? [1,2,3,4].map(i => <SkeletonCard key={i} />) : <>
              <div className="stat-card">
                <div className="label">掃描股票數</div>
                <div className="value mono">{patternData?.total_scanned ?? '—'}</div>
                <div className="sub">{patternData?.date || '每日推論後更新'}</div>
              </div>
              <div className="stat-card">
                <div className="label">型態匹配</div>
                <div className="value mono" style={{ color: (patternData?.patterns_found || 0) > 0 ? 'var(--positive)' : 'var(--text-muted)' }}>
                  {patternData?.patterns_found ?? '—'} 個
                </div>
                <div className="sub">Score ≥ 60</div>
              </div>
              <div className="stat-card" style={{ borderColor: 'rgba(0,255,136,0.2)', background: 'rgba(0,255,136,0.03)' }}>
                <div className="label">雙重確認</div>
                <div className="value mono text-positive">{patternData?.dual_confirm_count ?? '—'}</div>
                <div className="sub">型態 + Alpha 雙確認</div>
              </div>
              <div className="stat-card">
                <div className="label">掃描耗時</div>
                <div className="value mono">{patternData?.elapsed_seconds != null ? `${patternData.elapsed_seconds}s` : '—'}</div>
                <div className="sub">全市場掃描</div>
              </div>
            </>}
          </div>

          {/* Error or empty state */}
          {patternError && (
            <div style={{
              padding: '14px 18px', borderRadius: 10,
              background: 'rgba(255,71,87,0.06)', border: '1px solid rgba(255,71,87,0.25)',
            }}>
              <div style={{ fontSize: 13, color: 'var(--negative)', fontWeight: 600 }}>
                型態資料載入失敗
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                {patternError}（每日推論完成後自動生成）
              </div>
            </div>
          )}

          {!patternLoading && !patternError && patternData && patternData.patterns_found === 0 && (
            <div style={{
              padding: '14px 18px', borderRadius: 10,
              background: 'rgba(255,165,0,0.06)', border: '1px solid rgba(255,165,0,0.25)',
              display: 'flex', alignItems: 'center', gap: 14,
            }}>
              <span style={{ fontSize: 24 }}>📊</span>
              <div>
                <div style={{ fontSize: 13, color: 'var(--accent-amber)', fontWeight: 600 }}>
                  今日無符合條件的型態信號
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 3 }}>
                  掃描了 {patternData.total_scanned} 支股票，目前無 Score ≥ 60 的多方型態。
                  市場劇烈波動期間型態容易被破壞，建議等待趨勢穩定。
                </div>
              </div>
            </div>
          )}

          {/* Pattern Signals Table */}
          {!patternLoading && !patternError && (patternData?.signals?.length || 0) > 0 && (
            <div className="panel">
              <div className="panel-header">
                <div className="panel-title">
                  <span>📐</span> 型態匹配股票
                  <span className="badge badge-neutral" style={{ marginLeft: 8, fontSize: 10 }}>
                    共 {patternData.signals.length} 個信號
                  </span>
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    雙確認優先排序
                  </span>
                  <span className="badge" style={{
                    background: 'rgba(0,255,136,0.12)', color: 'var(--positive)',
                    border: '1px solid rgba(0,255,136,0.3)', fontSize: 10,
                  }}>
                    ✓ = Alpha ≤ 200 + 型態 ≥ 60
                  </span>
                </div>
              </div>
              <div className="panel-body-flush">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>股票</th>
                      <th>型態</th>
                      <th className="col-hide-mobile">時間框架</th>
                      <th>分數</th>
                      <th className="col-hide-mobile">關鍵價 / 目標 / 停損</th>
                      <th className="col-hide-mobile">風報比</th>
                      <th>Alpha 排名</th>
                      <th>確認</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patternData.signals.map((sig, idx) => (
                      <tr key={`${sig.stock_id}_${sig.pattern_id}_${sig.timeframe}`}
                        className="animate-fade-up"
                        style={{ animationDelay: `${idx * 0.025}s` }}>
                        <td>
                          <div style={{ fontWeight: 600, fontSize: 13 }}>{sig.name || sig.stock_id}</div>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{sig.stock_id}</div>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{sig.sector}</div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{
                              width: 24, height: 24, borderRadius: 4,
                              background: 'rgba(0,255,136,0.1)',
                              display: 'flex', alignItems: 'center', justifyContent: 'center',
                              fontSize: 12, fontWeight: 700, color: 'var(--positive)',
                            }}>
                              {PAT_ICON[sig.pattern_id] || '?'}
                            </span>
                            <span style={{ fontSize: 12, color: 'var(--text-primary)' }}>{sig.pattern_name}</span>
                          </div>
                        </td>
                        <td className="col-hide-mobile">
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{sig.timeframe_label}</span>
                        </td>
                        <td><ScoreBar score={sig.score} /></td>
                        <td className="col-hide-mobile">
                          <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', lineHeight: 1.7 }}>
                            <span style={{ color: 'var(--text-secondary)' }}>關</span> {sig.key_price}<br/>
                            <span style={{ color: 'var(--positive)' }}>目</span> {sig.target_price}<br/>
                            <span style={{ color: 'var(--negative)' }}>損</span> {sig.stop_loss}
                          </div>
                        </td>
                        <td className="col-hide-mobile">
                          <span className="mono" style={{ fontSize: 12, color: 'var(--accent-amber)' }}>
                            {sig.risk_reward}
                          </span>
                        </td>
                        <td>
                          <div style={{ fontSize: 12 }}>
                            <span className="mono" style={{ color: sig.alpha_rank <= 50 ? 'var(--positive)' : 'var(--text-secondary)' }}>
                              #{sig.alpha_rank}
                            </span>
                          </div>
                          <div style={{ fontSize: 10, color: sig.alpha_20d >= 0 ? 'var(--positive)' : 'var(--negative)' }}>
                            {sig.alpha_20d >= 0 ? '+' : ''}{(sig.alpha_20d * 100).toFixed(2)}%
                          </div>
                        </td>
                        <td>
                          {sig.dual_confirm ? (
                            <span style={{
                              fontSize: 10, padding: '2px 8px', borderRadius: 10,
                              background: 'rgba(0,255,136,0.12)',
                              color: 'var(--positive)',
                              border: '1px solid rgba(0,255,136,0.3)',
                            }}>✓ 雙確認</span>
                          ) : (
                            <span style={{
                              fontSize: 10, padding: '2px 8px', borderRadius: 10,
                              background: 'rgba(255,255,255,0.04)',
                              color: 'var(--text-muted)',
                              border: '1px solid var(--border)',
                            }}>型態</span>
                          )}
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
                            {sig.confidence}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ padding: '8px 16px', fontSize: 11, color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
                掃描範圍：W底 / 彈簧型W底 / 頭肩底 / 三角收斂 × 4 時間框架（短線/中線/長線/季線）·
                資料截止 {patternData.date}
              </div>
            </div>
          )}

          {/* Loading skeleton */}
          {patternLoading && (
            <div className="panel">
              <div className="panel-header">
                <div className="panel-title"><span>📐</span> 型態匹配股票</div>
              </div>
              <div className="panel-body"><SkeletonTable rows={8} cols={6} /></div>
            </div>
          )}

          {/* Pattern info */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>ℹ️</span> 掃描說明</div>
            </div>
            <div className="panel-body">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                <div>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>支援型態（多方）</div>
                  <div>• <b>W底</b>：兩低點差 &lt; 5%，頸線 &gt; 3%</div>
                  <div>• <b>彈簧型W底</b>：第二低 0.5%–2% 低於第一，快速反彈</div>
                  <div>• <b>頭肩底</b>：三底，中間最低，兩肩差 &lt; 10%</div>
                  <div>• <b>三角收斂</b>：低高收斂，至少 4 觸點</div>
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>評分系統（100分）</div>
                  <div>• 型態強度：40 分（形成品質）</div>
                  <div>• 成交量：30 分（量能確認）</div>
                  <div>• 位置：20 分（近頸線或突破）</div>
                  <div>• RSI 動能：10 分</div>
                  <div>• Alpha 加成：Top 200 +10 / Top 300 +5</div>
                  <div style={{ marginTop: 8 }}>
                    <b>雙重確認</b>：Score ≥ 60 且 Alpha 排名 ≤ 200
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
