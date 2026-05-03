import React, { useState, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, BarChart, Bar
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchSignals, fetchRebalanceHistory } from '../api/signals';
import { fetchMarket } from '../api/market';
import { SkeletonBlock } from '../components/SkeletonLoader';

// ── Simulation Engine ─────────────────────────────────────────────────────────

const INITIAL_CAPITAL = 1_000_000; // 100萬台幣

/**
 * Simulate portfolio allocation based on signals.
 * @param signals - array of signal items
 * @param useLLM  - whether to apply LLM sentiment adjustment
 * @param llmSentiment - 'positive' | 'neutral' | 'negative' (from Claude report context)
 */
function buildPortfolio(signals, useLLM, llmSentiment = 'neutral') {
  const topSignals = signals
    .filter(s => s.alpha_20d > 0 && s.signal === 'BUY')
    .slice(0, 20);

  if (!topSignals.length) return [];

  // LLM multiplier: positive report → scale up weight, negative → scale down
  const llmMult = useLLM
    ? (llmSentiment === 'positive' ? 1.15 : llmSentiment === 'negative' ? 0.80 : 1.0)
    : 1.0;

  // Cash reserve: quant bot keeps 10%, LLM bot adjusts based on sentiment
  const cashReserve = useLLM
    ? (llmSentiment === 'positive' ? 0.05 : llmSentiment === 'negative' ? 0.25 : 0.10)
    : 0.10;

  const deployable = INITIAL_CAPITAL * (1 - cashReserve);
  const totalWeight = topSignals.reduce((s, sig) => s + (sig.suggested_weight || 0.005), 0);

  return topSignals.map(sig => {
    const rawWeight = (sig.suggested_weight || 0.005) / totalWeight;
    const adjWeight = Math.min(rawWeight * llmMult, 0.15); // cap at 15% per stock
    const capital = deployable * adjWeight;
    const expReturn = sig.alpha_20d * (useLLM ? llmMult : 1);

    return {
      stock_id: sig.stock_id,
      name: sig.name,
      sector: sig.sector,
      weight: adjWeight,
      capital: Math.round(capital),
      exp_return_20d: expReturn,
      exp_pnl: Math.round(capital * expReturn),
      confidence: sig.confidence,
      sharpe: sig.uncertainty > 0 ? (sig.alpha_20d / sig.uncertainty).toFixed(1) : '—',
    };
  });
}

/**
 * Simulate daily equity curve over N trading days.
 * Uses alpha_20d as the expected 20-day return, distributes linearly.
 */
function simulateEquity(portfolio, days = 20) {
  const curve = [];
  let equity = INITIAL_CAPITAL;

  for (let d = 0; d <= days; d++) {
    // Each day adds a fraction of the expected 20-day return per holding
    const dailyReturn = portfolio.reduce((sum, pos) => {
      return sum + (pos.capital / INITIAL_CAPITAL) * (pos.exp_return_20d / 20);
    }, 0);
    equity = d === 0 ? INITIAL_CAPITAL : equity * (1 + dailyReturn + (Math.random() - 0.5) * 0.003);
    curve.push({
      day: d,
      equity: Math.round(equity),
      pnl: Math.round(equity - INITIAL_CAPITAL),
    });
  }
  return curve;
}

// ── Components ────────────────────────────────────────────────────────────────

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass" style={{ padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>Day {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: NT${p.value?.toLocaleString()}
        </div>
      ))}
    </div>
  );
};

function BotCard({ name, color, emoji, portfolio, equity }) {
  const pnl = equity.at(-1)?.pnl ?? 0;
  const pct = (pnl / INITIAL_CAPITAL * 100).toFixed(2);
  const totalPositions = portfolio.length;
  const totalDeployed = portfolio.reduce((s, p) => s + p.capital, 0);

  return (
    <div className="panel" style={{ borderColor: color, background: `color-mix(in srgb, ${color} 4%, var(--bg-panel))` }}>
      <div className="panel-header">
        <div className="panel-title">
          <span style={{ fontSize: 18 }}>{emoji}</span>
          <span style={{ marginLeft: 8, color }}>{name}</span>
        </div>
        <span className={`badge ${pnl >= 0 ? 'badge-positive' : 'badge-negative'}`}>
          {pnl >= 0 ? '+' : ''}NT${pnl.toLocaleString()}
        </span>
      </div>
      <div className="panel-body">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12, marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>預期報酬（20d）</div>
            <div className={`mono ${pnl >= 0 ? 'text-positive' : 'text-negative'}`} style={{ fontSize: 18 }}>
              {pnl >= 0 ? '+' : ''}{pct}%
            </div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>持倉數</div>
            <div className="mono" style={{ fontSize: 18 }}>{totalPositions} 檔</div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>投入資金</div>
            <div className="mono" style={{ fontSize: 18 }}>
              NT${Math.round(totalDeployed / 10000)}萬
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── RebalanceHistory ──────────────────────────────────────────────────────────

const CONF_COLOR = { '高信心': '#00ff88', '中信心': '#ffa500', '低信心': '#ff4757' };

function RebalanceHistory() {
  const { data, loading } = useApi(fetchRebalanceHistory);
  const history = data?.history ?? [];
  const [activeIdx, setActiveIdx] = useState(0);

  if (loading) {
    return (
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>📅</span> 調倉紀錄</div>
        </div>
        <div className="panel-body">
          <SkeletonBlock height={200} />
        </div>
      </div>
    );
  }

  if (!history.length) {
    return (
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
  }

  const current  = history[activeIdx];
  const previous = history[activeIdx + 1];
  const prevTickers = new Set((previous?.portfolio ?? []).map(p => p.ticker));
  const currTickers = new Set(current.portfolio.map(p => p.ticker));

  // Delta vs previous date
  const newIn  = current.portfolio.filter(p => !prevTickers.has(p.ticker)).map(p => p.ticker);
  const newOut = previous ? [...prevTickers].filter(t => !currTickers.has(t)) : [];

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>📅</span> 調倉紀錄</div>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>共 {history.length} 次調倉</span>
      </div>

      {/* Date Tabs */}
      <div style={{ display: 'flex', gap: 6, padding: '0 16px 12px', flexWrap: 'wrap', borderBottom: '1px solid var(--border)' }}>
        {history.map((h, i) => (
          <button
            key={h.date}
            onClick={() => setActiveIdx(i)}
            className={`btn ${i === activeIdx ? 'btn-primary' : 'btn-ghost'}`}
            style={{ padding: '4px 12px', fontSize: 11 }}
          >
            {h.date}
          </button>
        ))}
      </div>

      <div className="panel-body">
        {/* Meta */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            可投資股數：<strong style={{ color: 'var(--text-primary)' }}>{current.total_investable}</strong>
          </span>
          {newIn.length > 0 && (
            <span style={{ fontSize: 12, color: '#00ff88' }}>
              🟢 新進：{newIn.join('、')}
            </span>
          )}
          {newOut.length > 0 && (
            <span style={{ fontSize: 12, color: '#ff4757' }}>
              🔴 移出：{newOut.join('、')}
            </span>
          )}
          {!previous && (
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>（最早記錄，無法比較變動）</span>
          )}
        </div>

        {/* Holdings Table */}
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th><th>股票代號</th><th>比重</th>
              <th>Alpha 20d</th><th>Sharpe</th><th>信心</th>
            </tr>
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
                  <td>
                    <span style={{ fontSize: 10, color: CONF_COLOR[p.confidence] ?? 'var(--text-muted)' }}>
                      {p.confidence || '—'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}


export default function InvestmentSim() {
  const [activeBot, setActiveBot] = useState('both');
  const [simDays, setSimDays] = useState(20);
  const [seed, setSeed] = useState(42); // for re-simulate

  const { data: signalData, loading } = useApi(fetchSignals);
  const signals = signalData?.signals || [];

  // Detect Claude report sentiment from signals composition
  const llmSentiment = useMemo(() => {
    const buyCount = signals.filter(s => s.signal === 'BUY').length;
    const total = signals.length;
    if (!total) return 'neutral';
    const ratio = buyCount / total;
    return ratio > 0.55 ? 'positive' : ratio < 0.35 ? 'negative' : 'neutral';
  }, [signals]);

  // Build portfolios
  const portfolioA = useMemo(() => buildPortfolio(signals, false), [signals, seed]);
  const portfolioB = useMemo(() => buildPortfolio(signals, true, llmSentiment), [signals, llmSentiment, seed]);

  // Equity curves (seeded with key so we can re-simulate)
  const equityA = useMemo(() => simulateEquity(portfolioA, simDays), [portfolioA, simDays, seed]);
  const equityB = useMemo(() => simulateEquity(portfolioB, simDays), [portfolioB, simDays, seed]);

  // Combined chart data
  const chartData = equityA.map((pt, i) => ({
    day: pt.day,
    quant: pt.equity,
    llm: equityB[i]?.equity ?? pt.equity,
    baseline: INITIAL_CAPITAL,
  }));

  const pnlA = equityA.at(-1)?.pnl ?? 0;
  const pnlB = equityB.at(-1)?.pnl ?? 0;

  const sentimentColor = llmSentiment === 'positive' ? 'var(--positive)' : llmSentiment === 'negative' ? 'var(--negative)' : 'var(--accent-amber)';
  const sentimentLabel = llmSentiment === 'positive' ? '偏多' : llmSentiment === 'negative' ? '偏空' : '中性';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* ── Header ── */}
      <div className="page-header">
        <div>
          <div className="page-title">🤖 投資模擬機器人</div>
          <div className="page-subtitle">
            虛擬資金 NT$100萬 · 雙機器人對照 · 基於 {signalData?.date || '今日'} 推論結果
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <select
            value={simDays}
            onChange={e => setSimDays(Number(e.target.value))}
            style={{
              background: 'var(--bg-panel-2)', border: '1px solid var(--border)',
              borderRadius: 6, padding: '5px 10px', fontSize: 12, color: 'var(--text-primary)', cursor: 'pointer'
            }}
          >
            <option value={5}>5 交易日</option>
            <option value={20}>20 交易日</option>
            <option value={60}>60 交易日</option>
          </select>
          <button className="btn btn-ghost" style={{ fontSize: 12 }}
            onClick={() => setSeed(s => s + 1)}>
            🎲 重新模擬
          </button>
        </div>
      </div>

      {/* ── LLM Sentiment Banner ── */}
      <div style={{
        padding: '12px 18px', borderRadius: 10,
        background: `color-mix(in srgb, ${sentimentColor} 6%, var(--bg-panel))`,
        border: `1px solid color-mix(in srgb, ${sentimentColor} 30%, transparent)`,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 20 }}>🤖</span>
          <div>
            <div style={{ fontSize: 13, color: sentimentColor, fontWeight: 600 }}>
              AI 情緒讀取：{sentimentLabel}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
              基於模型多空比 {signals.filter(s=>s.signal==='BUY').length}/{signals.length}。
              Bot B 依此調整倉位（{llmSentiment === 'positive' ? '+15% 加碼' : llmSentiment === 'negative' ? '-20% 減碼，現金比例升至25%' : '維持標準倉位'}）
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <span className="badge badge-neutral" style={{ fontSize: 10 }}>Bot A: 純量化</span>
          <span className="badge badge-blue" style={{ fontSize: 10 }}>Bot B: 量化+LLM</span>
        </div>
      </div>

      {/* ── Bot Summary Cards ── */}
      {loading ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <SkeletonBlock height={120} /><SkeletonBlock height={120} />
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <BotCard
            name="Bot A — 純量化"
            color="var(--accent-blue)"
            emoji="📊"
            portfolio={portfolioA}
            equity={equityA}
          />
          <BotCard
            name="Bot B — 量化 + LLM"
            color="var(--accent-amber)"
            emoji="🧠"
            portfolio={portfolioB}
            equity={equityB}
          />
        </div>
      )}

      {/* ── Equity Curve ── */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>💹</span> 資產淨值模擬曲線（NT$）</div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[['both','雙機器人'], ['quant','純量化'], ['llm','量化+LLM']].map(([id, label]) => (
              <button key={id}
                className={`btn ${activeBot === id ? 'btn-primary' : 'btn-ghost'}`}
                style={{ padding: '4px 12px', fontSize: 11 }}
                onClick={() => setActiveBot(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="panel-body">
          {loading ? <SkeletonBlock height={220} /> : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                  tickFormatter={v => `D${v}`} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                  tickFormatter={v => `${(v/10000).toFixed(0)}萬`} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="baseline" name="基準（100萬）"
                  stroke="var(--border)" strokeWidth={1} dot={false} strokeDasharray="4 4" />
                {(activeBot === 'both' || activeBot === 'quant') && (
                  <Line type="monotone" dataKey="quant" name="Bot A 純量化"
                    stroke="var(--accent-blue)" strokeWidth={2} dot={false} />
                )}
                {(activeBot === 'both' || activeBot === 'llm') && (
                  <Line type="monotone" dataKey="llm" name="Bot B 量化+LLM"
                    stroke="var(--accent-amber)" strokeWidth={2} dot={false} />
                )}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* ── Holdings Comparison ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* Bot A Holdings */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title" style={{ color: 'var(--accent-blue)' }}>
              📊 Bot A 持倉明細（純量化）
            </div>
            <span className="badge badge-neutral" style={{ fontSize: 10 }}>
              現金保留 10%
            </span>
          </div>
          <div className="panel-body-flush">
            {loading ? <SkeletonBlock height={200} /> : (
              <table className="data-table">
                <thead>
                  <tr><th>股票</th><th>產業</th><th>比重</th><th>投入</th><th>預期損益</th></tr>
                </thead>
                <tbody>
                  {portfolioA.slice(0, 10).map((p, i) => (
                    <tr key={p.stock_id} style={{ animationDelay: `${i * 0.03}s` }} className="animate-fade-up">
                      <td>
                        <div style={{ fontWeight: 600, fontSize: 12 }}>{p.name}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{p.stock_id}</div>
                      </td>
                      <td><span className="badge badge-neutral" style={{ fontSize: 10 }}>{p.sector}</span></td>
                      <td className="mono" style={{ fontSize: 11 }}>{(p.weight*100).toFixed(1)}%</td>
                      <td className="mono" style={{ fontSize: 11 }}>{Math.round(p.capital/10000)}萬</td>
                      <td className={`mono ${p.exp_pnl >= 0 ? 'text-positive' : 'text-negative'}`} style={{ fontSize: 11 }}>
                        {p.exp_pnl >= 0 ? '+' : ''}NT${p.exp_pnl.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Bot B Holdings */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title" style={{ color: 'var(--accent-amber)' }}>
              🧠 Bot B 持倉明細（量化+LLM）
            </div>
            <span className="badge badge-neutral" style={{ fontSize: 10 }}>
              LLM 情緒：{sentimentLabel}
              {llmSentiment === 'positive' ? ' · 加碼 +15%' : llmSentiment === 'negative' ? ' · 減碼 -20%' : ''}
            </span>
          </div>
          <div className="panel-body-flush">
            {loading ? <SkeletonBlock height={200} /> : (
              <table className="data-table">
                <thead>
                  <tr><th>股票</th><th>產業</th><th>比重</th><th>投入</th><th>預期損益</th></tr>
                </thead>
                <tbody>
                  {portfolioB.slice(0, 10).map((p, i) => (
                    <tr key={p.stock_id} style={{ animationDelay: `${i * 0.03}s` }} className="animate-fade-up">
                      <td>
                        <div style={{ fontWeight: 600, fontSize: 12 }}>{p.name}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{p.stock_id}</div>
                      </td>
                      <td><span className="badge badge-neutral" style={{ fontSize: 10 }}>{p.sector}</span></td>
                      <td className="mono" style={{ fontSize: 11 }}>{(p.weight*100).toFixed(1)}%</td>
                      <td className="mono" style={{ fontSize: 11 }}>{Math.round(p.capital/10000)}萬</td>
                      <td className={`mono ${p.exp_pnl >= 0 ? 'text-positive' : 'text-negative'}`} style={{ fontSize: 11 }}>
                        {p.exp_pnl >= 0 ? '+' : ''}NT${p.exp_pnl.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      {/* ── Summary Comparison ── */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>⚖️</span> 機器人績效對比</div>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>預期 {simDays} 交易日後</span>
        </div>
        <div className="panel-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, textAlign: 'center' }}>
            {[
              { label: 'Bot A 預期淨值', value: `NT$${(INITIAL_CAPITAL + pnlA).toLocaleString()}`, sub: `${pnlA >= 0 ? '+' : ''}NT$${pnlA.toLocaleString()}`, color: 'var(--accent-blue)' },
              { label: 'Bot B 預期淨值', value: `NT$${(INITIAL_CAPITAL + pnlB).toLocaleString()}`, sub: `${pnlB >= 0 ? '+' : ''}NT$${pnlB.toLocaleString()}`, color: 'var(--accent-amber)' },
              { label: 'LLM 效果', value: `${((pnlB - pnlA) / INITIAL_CAPITAL * 100).toFixed(2)}%`, sub: pnlB > pnlA ? '🧠 LLM 有貢獻' : '📊 純量化更好', color: pnlB > pnlA ? 'var(--positive)' : 'var(--negative)' },
              { label: '模擬資金', value: 'NT$100萬', sub: '虛擬帳戶', color: 'var(--text-muted)' },
            ].map(m => (
              <div key={m.label}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>{m.label}</div>
                <div className="mono" style={{ fontSize: 18, color: m.color }}>{m.value}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{m.sub}</div>
              </div>
            ))}
          </div>
          <div className="divider" />
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
            ⚠️ 以上為基於模型 Alpha 預測的模擬結果，不代表真實報酬。模擬每日隨機性已加入。
            每次重新跑推論後更新持倉。
          </div>
        </div>
      </div>

      {/* ── Rebalance History ── */}
      <RebalanceHistory />

    </div>
  );
}

