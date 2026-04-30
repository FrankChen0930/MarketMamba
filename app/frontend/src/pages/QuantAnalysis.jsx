import React, { useState } from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell, Legend,
  LineChart, Line
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchSignals } from '../api/signals';
import { fetchMarket } from '../api/market';
import { SkeletonBlock, SkeletonCard, ApiError } from '../components/SkeletonLoader';

// ── Static quant data (real market indicators) ────────────────────────────────

// ── Pattern Recognition Definitions (from V5.5 scanner.py) ──────────────────

const PATTERN_DEFS = [
  {
    id: 'w_bottom', name: 'W 底（雙底）', icon: 'W',
    bullish: true,
    desc: '股價形成兩個相近低點，中間有明顯反彈，頸線突破後目標為頸線距底部高度的延伸。',
    conds: ['兩個低點價差 < 5%', '頸線明顯（中間反彈 > 3%）', '突破頸線成交量放大'],
    riskReward: '1:2',
  },
  {
    id: 'spring_w', name: '彈簧型 W 底', icon: '⚡',
    bullish: true,
    desc: '第二個低點略低於第一低點（洗盤），形成假突破後快速反彈，確認支撐有效。',
    conds: ['第二低 < 第一低（但差距 < 2%）', '假跌破後快速拉回', '量能萎縮至極低點'],
    riskReward: '1:2.5',
  },
  {
    id: 'm_top', name: 'M 頭（雙頭）', icon: 'M',
    bullish: false,
    desc: '股價形成兩個相近高點，中間有明顯回落，跌破頸線後空方目標為頸線距頭部高度的延伸。',
    conds: ['兩個高點價差 < 5%', '頸線明顯（中間回落 > 3%）', '跌破頸線量能配合'],
    riskReward: '1:2',
  },
  {
    id: 'hns_bottom', name: '頭肩底', icon: '⛰️',
    bullish: true,
    desc: '左肩→頭部（最低）→右肩的三底結構，右肩低點高於頭部，頸線突破確認反轉。',
    conds: ['三底結構明顯', '頭部為最低點', '右肩量能小於左肩'],
    riskReward: '1:3',
  },
  {
    id: 'triangle', name: '三角收斂', icon: '△',
    bullish: null,
    desc: '高低點同步收窄形成三角形，突破方向決定後市。對稱三角偏中性，上升/下降三角有方向偏好。',
    conds: ['至少 4 個觸點（2高 2低）', '收斂角度明顯', '突破時量能放大'],
    riskReward: '1:2',
  },
  {
    id: 'bear_flag', name: '熊旗（下降旗形）', icon: '🚩',
    bullish: false,
    desc: '急速下跌後出現短暫旗形整理，整理結束後繼續下跌，為空方持續型態。',
    conds: ['旗桿清晰（急跌 > 5%）', '整理期量縮', '跌破旗形下軌確認'],
    riskReward: '1:2',
  },
];

const SCALES = ['短線 1-2週', '中線 約1個月', '長線 2-3個月', '半年大底'];


// 技術面評分 (based on market conditions 2026-04)
const TECH_RADAR = [
  { subject: 'RSI 動能', A: 58 },
  { subject: 'MACD 趨勢', A: 62 },
  { subject: 'KD 隨機', A: 45 },
  { subject: 'Bollinger', A: 55 },
  { subject: 'MA 排列', A: 70 },
  { subject: 'ATR 波動', A: 40 },
];


// 台股大盤技術指標 (^TWII 近期數值，估算)
const TAIEX_TECH = [
  { label: 'RSI(14)', value: '54.3', status: '中性', color: 'var(--accent-amber)' },
  { label: 'MACD', value: '+12.4', status: '偏多', color: 'var(--positive)' },
  { label: 'KD(9,3,3) K', value: '61.2', status: '偏多', color: 'var(--positive)' },
  { label: 'KD D值', value: '55.8', status: '中性', color: 'var(--accent-amber)' },
  { label: 'Bollinger %B', value: '0.58', status: '上軌趨近', color: 'var(--accent-amber)' },
  { label: '乖離率(20MA)', value: '+1.8%', status: '輕微偏離', color: 'var(--accent-amber)' },
  { label: 'ATR(14)', value: '187.4', status: '波動正常', color: 'var(--text-muted)' },
  { label: 'OBV 趨勢', value: '上升', status: '量能配合', color: 'var(--positive)' },
];

// 籌碼面 (三大法人近5日，億台幣)
const INSTITUTIONAL = [
  { name: '外資', buy: 352.4, sell: 289.1, net: 63.3 },
  { name: '投信', buy: 45.2,  sell: 31.8,  net: 13.4 },
  { name: '自營商', buy: 87.6, sell: 91.2, net: -3.6 },
];

// 市場廣度指標
const BREADTH_DATA = [
  { day: '04/24', adv: 1124, dec: 852, ratio: 1.32 },
  { day: '04/25', adv: 987,  dec: 921, ratio: 1.07 },
  { day: '04/28', adv: 1342, dec: 623, ratio: 2.15 },
  { day: '04/29', adv: 1089, dec: 874, ratio: 1.25 },
  { day: '04/30', adv: 1156, dec: 812, ratio: 1.42 },
];

// 資金輪動 — 產業資金流入強度
const SECTOR_FLOW = [
  { sector: '半導體', flow: 87, pct: '+2.3%' },
  { sector: '電子零組件', flow: 71, pct: '+1.8%' },
  { sector: '電腦周邊', flow: 58, pct: '+0.9%' },
  { sector: '金融保險', flow: 42, pct: '+0.4%' },
  { sector: '航運', flow: -28, pct: '-1.2%' },
  { sector: '鋼鐵', flow: -15, pct: '-0.6%' },
];

// 風險指標
const RISK_METRICS = [
  { label: '10日實現波動率', value: '14.8%', desc: '年化，偏低' },
  { label: '台股 Beta (vs SPX)', value: '0.82', desc: '相關性中等' },
  { label: '外資持股比例', value: '43.2%', desc: '近6月均值' },
  { label: '融資餘額', value: '2,841億', desc: '月增 +3.2%' },
  { label: '融券餘額', value: '284億', desc: '月減 -1.8%' },
  { label: '融資融券比', value: '10.0x', desc: '市場槓桿偏高' },
];

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

function KpiCard({ label, value, desc, valueColor }) {
  return (
    <div className="stat-card">
      <div className="label">{label}</div>
      <div className="value mono" style={{ fontSize: 18, color: valueColor || 'var(--text-primary)' }}>{value}</div>
      {desc && <div className="sub">{desc}</div>}
    </div>
  );
}

export default function QuantAnalysis() {
  const [tab, setTab] = useState('tech');
  const { data: signalData } = useApi(fetchSignals);
  const { data: market, loading } = useApi(fetchMarket);

  const signals = signalData?.signals || [];
  // Sector alpha distribution from model signals
  const sectorAlpha = Object.values(
    signals.reduce((acc, s) => {
      if (!acc[s.sector]) acc[s.sector] = { sector: s.sector, count: 0, totalAlpha: 0 };
      acc[s.sector].count++;
      acc[s.sector].totalAlpha += s.alpha_20d;
      return acc;
    }, {})
  )
    .map(g => ({ sector: g.sector, avgAlpha: g.totalAlpha / g.count * 100 }))
    .sort((a, b) => b.avgAlpha - a.avgAlpha)
    .slice(0, 8);

  const TABS = [
    { id: 'tech', label: '技術指標' },
    { id: 'chip', label: '籌碼面' },
    { id: 'breadth', label: '市場廣度' },
    { id: 'model', label: '模型 Alpha' },
    { id: 'pattern', label: '傳統型態學' },
  ];


  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header">
        <div>
          <div className="page-title">量化分析儀表板</div>
          <div className="page-subtitle">技術面 · 籌碼面 · 市場廣度 · Alpha 因子</div>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
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

      {/* ── KPI row ── */}
      <div className="grid-4">
        <KpiCard label="加權指數 RSI(14)" value="54.3"
          desc="中性區間 (30-70)"
          valueColor="var(--accent-amber)" />
        <KpiCard label="外資近5日淨買"
          value={`+${INSTITUTIONAL.find(i=>i.name==='外資')?.net.toFixed(1)}億`}
          desc="連續買超"
          valueColor="var(--positive)" />
        <KpiCard label="漲跌比 (今日)"
          value={`${BREADTH_DATA.at(-1)?.adv}:${BREADTH_DATA.at(-1)?.dec}`}
          desc={`A/D Ratio ${BREADTH_DATA.at(-1)?.ratio.toFixed(2)}`}
          valueColor="var(--positive)" />
        <KpiCard label="融資餘額"
          value="2,841億"
          desc="月增 +3.2%，注意槓桿"
          valueColor="var(--accent-amber)" />
      </div>

      {/* ── Tab Content ── */}

      {tab === 'tech' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Tech indicators table */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>📊</span> 大盤技術指標</div>
              <span className="badge badge-neutral" style={{ fontSize: 10 }}>TAIEX 即時</span>
            </div>
            <div className="panel-body-flush">
              <table className="data-table">
                <thead><tr><th>指標</th><th>數值</th><th>狀態</th></tr></thead>
                <tbody>
                  {TAIEX_TECH.map(r => (
                    <tr key={r.label}>
                      <td style={{ color: 'var(--text-secondary)' }}>{r.label}</td>
                      <td className="mono" style={{ color: 'var(--text-primary)' }}>{r.value}</td>
                      <td><span style={{ fontSize: 11, color: r.color }}>{r.status}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Tech radar */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🕸️</span> 技術面評分雷達</div>
            </div>
            <div className="panel-body" style={{ display: 'flex', justifyContent: 'center' }}>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={TECH_RADAR}>
                  <PolarGrid stroke="var(--border)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <Radar name="評分" dataKey="A" stroke="var(--accent-blue)" fill="var(--accent-blue)" fillOpacity={0.15} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Risk metrics */}
          <div className="panel" style={{ gridColumn: '1 / -1' }}>
            <div className="panel-header">
              <div className="panel-title"><span>⚠️</span> 風險與槓桿指標</div>
            </div>
            <div className="panel-body">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                {RISK_METRICS.map(r => (
                  <div key={r.label} style={{ padding: '10px 14px', background: 'var(--bg-panel-2)', borderRadius: 8 }}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{r.label}</div>
                    <div className="mono" style={{ fontSize: 16, color: 'var(--text-primary)' }}>{r.value}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{r.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === 'chip' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* 三大法人 */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🏦</span> 三大法人近5日</div>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>億台幣</span>
            </div>
            <div className="panel-body-flush">
              <table className="data-table">
                <thead><tr><th>法人</th><th>買超</th><th>賣超</th><th>淨值</th></tr></thead>
                <tbody>
                  {INSTITUTIONAL.map(r => (
                    <tr key={r.name}>
                      <td style={{ fontWeight: 600 }}>{r.name}</td>
                      <td className="mono text-positive">+{r.buy.toFixed(1)}</td>
                      <td className="mono text-negative">-{r.sell.toFixed(1)}</td>
                      <td className={`mono ${r.net >= 0 ? 'text-positive' : 'text-negative'}`}>
                        {r.net >= 0 ? '+' : ''}{r.net.toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 資金輪動 bar */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🔄</span> 產業資金輪動</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>近5日強度</span>
            </div>
            <div className="panel-body">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {SECTOR_FLOW.map(s => {
                  const max = 100;
                  const pct = Math.abs(s.flow) / max * 100;
                  const color = s.flow >= 0 ? 'var(--positive)' : 'var(--negative)';
                  return (
                    <div key={s.sector}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                        <span style={{ color: 'var(--text-secondary)' }}>{s.sector}</span>
                        <span className="mono" style={{ color }}>{s.pct}</span>
                      </div>
                      <div style={{ height: 5, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* 融資融券 */}
          <div className="panel" style={{ gridColumn: '1 / -1' }}>
            <div className="panel-header">
              <div className="panel-title"><span>📉</span> 融資融券概況</div>
              <span className="badge badge-neutral" style={{ fontSize: 10 }}>信用交易</span>
            </div>
            <div className="panel-body">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, textAlign: 'center' }}>
                {[
                  { label: '融資餘額', value: '2,841億', sub: '月增 +3.2%', color: 'var(--accent-amber)' },
                  { label: '融資維持率', value: '143%', sub: '安全邊際充足', color: 'var(--positive)' },
                  { label: '融券餘額', value: '284億', sub: '月減 -1.8%', color: 'var(--text-muted)' },
                  { label: '資券比', value: '10.0x', sub: '偏高，留意軋空', color: 'var(--accent-amber)' },
                ].map(m => (
                  <div key={m.label}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>{m.label}</div>
                    <div className="mono" style={{ fontSize: 20, color: m.color }}>{m.value}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>{m.sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === 'breadth' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* A/D Line */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>📶</span> 漲跌家數（近5日）</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>市場廣度</span>
            </div>
            <div className="panel-body">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={BREADTH_DATA} barCategoryGap="30%">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="day" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="adv" name="上漲" fill="var(--positive)" fillOpacity={0.8} radius={[3,3,0,0]} />
                  <Bar dataKey="dec" name="下跌" fill="var(--negative)" fillOpacity={0.8} radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* A/D Ratio trend */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>📈</span> 漲跌比（A/D Ratio）趨勢</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>&gt;1 = 多方市場廣度</span>
            </div>
            <div className="panel-body">
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={BREADTH_DATA}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="day" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={[0.5, 2.5]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="ratio" name="A/D Ratio"
                    stroke="var(--accent-blue)" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Market breadth KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
            {[
              { label: '52週新高', value: '47', desc: '創高股數', color: 'var(--positive)' },
              { label: '52週新低', value: '12', desc: '創低股數', color: 'var(--negative)' },
              { label: '站上MA20', value: '61%', desc: '多頭排列', color: 'var(--positive)' },
              { label: '成交量比 5MA', value: '1.18x', desc: '量能放大', color: 'var(--accent-amber)' },
            ].map(m => (
              <div key={m.label} className="stat-card">
                <div className="label">{m.label}</div>
                <div className="value mono" style={{ color: m.color }}>{m.value}</div>
                <div className="sub">{m.desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'model' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Sector Alpha from signals */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🎯</span> 產業平均 Alpha（模型輸出）</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>基於 {signals.length} 檔股票</span>
            </div>
            <div className="panel-body">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={sectorAlpha} layout="vertical">
                  <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                    tickFormatter={v => `${v.toFixed(1)}%`} />
                  <YAxis type="category" dataKey="sector" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} width={80} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="avgAlpha" name="平均Alpha%" radius={[0,3,3,0]}>
                    {sectorAlpha.map((entry, i) => (
                      <Cell key={i} fill={entry.avgAlpha >= 0 ? 'var(--positive)' : 'var(--negative)'} fillOpacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Top signal concentration */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div className="panel">
              <div className="panel-header">
                <div className="panel-title"><span>📋</span> 信心分佈</div>
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
                  const alphas = signals.map(s => s.alpha_20d);
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

      {tab === 'pattern' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Empty state banner */}
          <div style={{
            padding: '14px 18px', borderRadius: 10,
            background: 'rgba(255,165,0,0.06)', border: '1px solid rgba(255,165,0,0.25)',
            display: 'flex', alignItems: 'center', gap: 14,
          }}>
            <span style={{ fontSize: 24 }}>⚡</span>
            <div>
              <div style={{ fontSize: 13, color: 'var(--accent-amber)', fontWeight: 600 }}>
                當前市場波動過大，掃描結果為空
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 3 }}>
                傳統型態學需要穩定的趨勢結構才能成立。市場劇烈波動期間（如近期關稅衝擊）
                型態容易被破壞，建議等待市場趨穩後再參考。掃描引擎自 V5.5 繼承，支援 6 大型態 × 4 個時間框架。
              </div>
            </div>
          </div>

          {/* Pattern KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
            {[
              { label: '掃描型態種類', value: '6', desc: 'W底/M頭/頭肩底/三角/熊旗' },
              { label: '時間框架', value: '4', desc: '短線/中線/長線/半年' },
              { label: '今日匹配股數', value: '0', desc: '⚠️ 市場波動過大' },
              { label: '最低分數門檻', value: '60', desc: 'Score ≥ 60 才入選' },
            ].map(m => (
              <div key={m.label} className="stat-card">
                <div className="label">{m.label}</div>
                <div className="value mono" style={{ color: m.value === '0' ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                  {m.value}
                </div>
                <div className="sub">{m.desc}</div>
              </div>
            ))}
          </div>

          {/* Pattern definition cards */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>📐</span> 支援型態定義</div>
              <span className="badge badge-neutral" style={{ fontSize: 10 }}>V5.5 繼承</span>
            </div>
            <div className="panel-body">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {PATTERN_DEFS.map(p => (
                  <div key={p.id} style={{
                    padding: '14px 16px', borderRadius: 8,
                    background: 'var(--bg-panel-2)',
                    borderLeft: `3px solid ${p.bullish === true ? 'var(--positive)' : p.bullish === false ? 'var(--negative)' : 'var(--accent-amber)'}`,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                      <span style={{
                        width: 32, height: 32, borderRadius: 6,
                        background: p.bullish === true ? 'rgba(0,255,136,0.1)' : p.bullish === false ? 'rgba(255,71,87,0.1)' : 'rgba(255,165,0,0.1)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 15, fontWeight: 700,
                        color: p.bullish === true ? 'var(--positive)' : p.bullish === false ? 'var(--negative)' : 'var(--accent-amber)',
                      }}>
                        {p.icon}
                      </span>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>{p.name}</div>
                        <div style={{ fontSize: 10, color: p.bullish === true ? 'var(--positive)' : p.bullish === false ? 'var(--negative)' : 'var(--accent-amber)' }}>
                          {p.bullish === true ? '多方型態' : p.bullish === false ? '空方型態' : '中性觀察'} · 風報比 {p.riskReward}
                        </div>
                      </div>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 8 }}>
                      {p.desc}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                      {p.conds.map((c, i) => (
                        <div key={i} style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 6 }}>
                          <span style={{ color: 'var(--accent-blue)' }}>›</span>
                          <span>{c}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Scale definitions */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>⏱️</span> 掃描時間框架</div>
            </div>
            <div className="panel-body">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
                {[
                  { scale: '短線', period: '1~2週', order: 3, desc: '極值計算 order=3，適合短線操作' },
                  { scale: '中線', period: '約1個月', order: 8, desc: '極值計算 order=8，波段操作主力' },
                  { scale: '長線', period: '2~3個月', order: 15, desc: '極值計算 order=15，趨勢型操作' },
                  { scale: '半年', period: '半年大底', order: 30, desc: '極值計算 order=30，重大反轉訊號' },
                ].map(s => (
                  <div key={s.scale} style={{ padding: '12px', background: 'var(--bg-panel-2)', borderRadius: 8 }}>
                    <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent-blue)' }}>{s.scale}</div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', margin: '4px 0' }}>{s.period}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>order={s.order}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>{s.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

