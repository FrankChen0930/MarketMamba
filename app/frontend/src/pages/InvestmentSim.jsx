import React, { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, BarChart, Bar, Cell, Legend,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchSimBacktest, fetchIcAnalysis } from '../api/sim';
import { SkeletonBlock, SkeletonCard, ApiError } from '../components/SkeletonLoader';


// ── Helpers ───────────────────────────────────────────────────────────────────

const pct    = (v, dp = 2) => `${v >= 0 ? '+' : ''}${v?.toFixed(dp)}%`;
const money  = (v) => `NT$${Math.abs(Math.round(v)).toLocaleString()}`;
const pos    = (v, color = true) => color ? (v >= 0 ? 'var(--positive)' : 'var(--negative)') : 'inherit';

const PNL_POS  = 'var(--positive)';
const PNL_NEG  = 'var(--negative)';
const CLR_BLUE = 'var(--accent-blue)';
const CLR_AMB  = 'var(--accent-amber)';

// ── Tooltips ──────────────────────────────────────────────────────────────────

function EquityTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const pt = payload[0]?.payload;
  const portLine  = payload.find(p => p.dataKey === 'port_idx');
  const benchLine = payload.find(p => p.dataKey === 'bench_idx');
  return (
    <div className="glass" style={{ padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {portLine && (
        <div style={{ color: CLR_BLUE }}>
          機器人：{portLine.value?.toFixed(2)} ({pt?.daily_return_pct >= 0 ? '+' : ''}{pt?.daily_return_pct?.toFixed(2)}%)
        </div>
      )}
      {benchLine && (
        <div style={{ color: CLR_AMB, marginTop: 2 }}>
          TWII：{benchLine.value?.toFixed(2)}
        </div>
      )}
      {pt?.n_positions !== undefined && (
        <div style={{ color: 'var(--text-muted)', fontSize: 10, marginTop: 4 }}>
          {pt.n_positions} 檔持倉
        </div>
      )}
    </div>
  );
}

function IcTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const v = payload[0]?.value;
  return (
    <div className="glass" style={{ padding: '8px 12px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
      <div style={{ color: v >= 0 ? PNL_POS : PNL_NEG, fontFamily: 'var(--font-mono)' }}>
        IC = {v >= 0 ? '+' : ''}{v?.toFixed(4)}
      </div>
    </div>
  );
}

// ── Stat Card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, color }) {
  return (
    <div className="stat-card">
      <div className="label">{label}</div>
      <div className="value mono" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

// ── Holdings Panel ────────────────────────────────────────────────────────────

function HoldingsPanel({ holdings, loading }) {
  if (loading) return (
    <div className="panel">
      <div className="panel-header"><div className="panel-title"><span>💼</span> 目前持倉</div></div>
      <div className="panel-body"><SkeletonBlock height={160} /></div>
    </div>
  );

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>💼</span> 目前持倉</div>
        <span className="badge badge-neutral">{holdings.length} 檔</span>
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        {!holdings.length ? (
          <div style={{ padding: '24px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            目前無持倉 · 等待 BUY 訊號進場
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>代號</th><th>入場日</th><th>入場價</th><th>現價</th>
                <th>市值</th><th>損益</th><th>持有</th><th>SQ排名</th>
              </tr>
            </thead>
            <tbody>
              {holdings.map(h => (
                <tr key={h.ticker}>
                  <td>
                    <span style={{ fontWeight: 600, fontSize: 12 }}>{h.ticker}</span>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{h.weight_pct?.toFixed(1)}%</div>
                  </td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{h.entry_date?.slice(5)}</td>
                  <td className="mono" style={{ fontSize: 11 }}>{h.entry_price?.toFixed(2)}</td>
                  <td className="mono" style={{ fontSize: 11, color: h.current_price > h.entry_price ? PNL_POS : PNL_NEG }}>
                    {h.current_price?.toFixed(2)}
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {(h.market_value / 10000).toFixed(1)}萬
                  </td>
                  <td>
                    <div className="mono" style={{ fontSize: 11, color: pos(h.pnl) }}>
                      {h.pnl >= 0 ? '+' : ''}{(h.pnl / 1000).toFixed(1)}K
                    </div>
                    <div className="mono" style={{ fontSize: 10, color: pos(h.pnl_pct) }}>
                      {pct(h.pnl_pct)}
                    </div>
                  </td>
                  <td className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {h.days_held}天
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    <span style={{ color: h.rank <= 10 ? PNL_POS : CLR_BLUE }}>#{h.rank}</span>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{h.signal_quality?.toFixed(2)}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ── Watchlist Panel ───────────────────────────────────────────────────────────

function WatchlistPanel({ watchlist, loading }) {
  if (loading) return (
    <div className="panel">
      <div className="panel-header"><div className="panel-title"><span>👀</span> 觀察清單</div></div>
      <div className="panel-body"><SkeletonBlock height={120} /></div>
    </div>
  );

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>👀</span> 觀察清單</div>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          排名第 21–35，候選入場
        </span>
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        {!watchlist.length ? (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            目前無候選股票
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr><th>代號</th><th>排名</th><th>SQ</th><th>收盤價</th><th>觀察天數</th></tr>
            </thead>
            <tbody>
              {watchlist.map(w => (
                <tr key={w.ticker}>
                  <td style={{ fontWeight: 600, fontSize: 12 }}>{w.ticker}</td>
                  <td className="mono" style={{ fontSize: 11, color: CLR_AMB }}>#{w.rank}</td>
                  <td className="mono" style={{ fontSize: 11 }}>{w.signal_quality?.toFixed(2)}</td>
                  <td className="mono" style={{ fontSize: 11 }}>{w.close?.toFixed(2)}</td>
                  <td className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{w.days_in_watch}天</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ── IC Panel ──────────────────────────────────────────────────────────────────

function IcPanel() {
  const { data, loading } = useApi(fetchIcAnalysis);
  const s5 = data?.horizon_summary?.['5d'] ?? {};
  const rolling = data?.rolling_ic_5d ?? [];
  const series  = data?.ic_series_5d  ?? [];

  const icirColor = !s5.icir ? 'var(--text-muted)'
    : s5.icir >= 0.4 ? PNL_POS
    : s5.icir >= 0.2 ? CLR_AMB
    : PNL_NEG;

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>🧮</span> 模型預測力驗證（IC / ICIR）</div>
        {data?.period_start && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {data.period_start} → {data.period_end}
          </span>
        )}
      </div>
      <div className="panel-body">
        {loading ? <SkeletonBlock height={180} /> : !s5.n_days ? (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, padding: '24px 0' }}>
            歸檔資料不足，需要 ≥5 個交易日後才能計算 IC
          </div>
        ) : (
          <>
            {/* IC Stats row */}
            <div style={{ display: 'flex', gap: 24, marginBottom: 16, flexWrap: 'wrap' }}>
              {[
                { label: 'IC 均值 (5d)', value: s5.mean_ic?.toFixed(4), color: pos(s5.mean_ic) },
                { label: 'ICIR',         value: s5.icir?.toFixed(3),    color: icirColor },
                { label: 't-統計量',     value: s5.t_stat?.toFixed(2),  color: Math.abs(s5.t_stat) >= 2 ? PNL_POS : 'var(--text-muted)' },
                { label: 'IC > 0 比例',  value: `${s5.ic_gt0_pct?.toFixed(0)}%`, color: s5.ic_gt0_pct >= 55 ? PNL_POS : CLR_AMB },
                { label: '有效天數',     value: `${s5.n_days} 天`,      color: 'var(--text-secondary)' },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ minWidth: 90 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
                  <div className="mono" style={{ fontSize: 14, fontWeight: 600, color }}>{value ?? '—'}</div>
                </div>
              ))}
            </div>

            {/* IC interpretation */}
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 12, padding: '8px 12px', background: 'rgba(255,255,255,0.03)', borderRadius: 6 }}>
              {s5.icir >= 0.4
                ? `✅ ICIR=${s5.icir?.toFixed(2)} 表示模型預測力良好（業界標準 ≥0.4）`
                : s5.icir >= 0.2
                ? `⚠️ ICIR=${s5.icir?.toFixed(2)} 尚可，但低於理想門檻 0.4`
                : `❌ ICIR=${s5.icir?.toFixed(2)} 偏低，模型預測穩定性不足`}
              {s5.t_stat && ` · t=${s5.t_stat?.toFixed(2)}${Math.abs(s5.t_stat) >= 2 ? '（顯著）' : '（不顯著）'}`}
            </div>

            {/* Rolling IC chart */}
            {rolling.length >= 3 && (
              <ResponsiveContainer width="100%" height={120}>
                <LineChart data={rolling}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'var(--text-muted)' }}
                    tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 9, fill: 'var(--text-muted)' }}
                    tickFormatter={v => v.toFixed(2)} domain={[-0.15, 0.15]} />
                  <Tooltip content={<IcTooltip />} />
                  <ReferenceLine y={0} stroke="var(--border)" />
                  <Line type="monotone" dataKey="rolling_ic" name="滾動IC(5d)"
                    stroke={CLR_BLUE} strokeWidth={2} dot={false}
                    activeDot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            )}

            {/* Daily IC bars */}
            {series.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>逐日 IC（5天期）</div>
                <ResponsiveContainer width="100%" height={80}>
                  <BarChart data={series} barCategoryGap="15%">
                    <XAxis dataKey="pred_date" tick={false} />
                    <YAxis hide domain={[-0.15, 0.15]} />
                    <Tooltip content={<IcTooltip />} />
                    <ReferenceLine y={0} stroke="var(--border)" />
                    <Bar dataKey="ic" radius={[2, 2, 0, 0]}>
                      {series.map((pt, i) => (
                        <Cell key={i} fill={pt.ic >= 0 ? PNL_POS : PNL_NEG} opacity={0.75} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Transactions Panel ────────────────────────────────────────────────────────

function TransactionsPanel({ transactions, loading }) {
  const [show, setShow] = useState(15);
  const txs = [...(transactions ?? [])].reverse().slice(0, show);

  if (loading) return (
    <div className="panel">
      <div className="panel-header"><div className="panel-title"><span>📋</span> 交易紀錄</div></div>
      <div className="panel-body"><SkeletonBlock height={160} /></div>
    </div>
  );

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>📋</span> 交易紀錄</div>
        <span className="badge badge-neutral">{transactions?.length ?? 0} 筆</span>
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        {!txs.length ? (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            尚無交易紀錄
          </div>
        ) : (
          <>
            <table className="data-table">
              <thead>
                <tr><th>日期</th><th>動作</th><th>代號</th><th>價格</th><th>股數</th><th>損益</th><th>原因</th></tr>
              </thead>
              <tbody>
                {txs.map((t, i) => (
                  <tr key={i}>
                    <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{t.date?.slice(5)}</td>
                    <td>
                      <span style={{
                        fontSize: 10, fontWeight: 600,
                        color: t.action === 'BUY' ? PNL_POS : PNL_NEG,
                        background: t.action === 'BUY' ? 'rgba(0,255,136,0.1)' : 'rgba(255,71,87,0.1)',
                        borderRadius: 4, padding: '1px 6px',
                      }}>
                        {t.action === 'BUY' ? '買進' : '賣出'}
                      </span>
                    </td>
                    <td style={{ fontWeight: 600, fontSize: 12 }}>{t.ticker}</td>
                    <td className="mono" style={{ fontSize: 11 }}>{t.price?.toFixed(2)}</td>
                    <td className="mono" style={{ fontSize: 11 }}>{t.shares?.toLocaleString()}</td>
                    <td>
                      {t.action === 'SELL' ? (
                        <span className="mono" style={{ fontSize: 11, color: pos(t.pnl) }}>
                          {t.pnl >= 0 ? '+' : ''}{(t.pnl / 1000).toFixed(1)}K
                        </span>
                      ) : (
                        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>—</span>
                      )}
                    </td>
                    <td style={{ fontSize: 10, color: 'var(--text-muted)', maxWidth: 140 }}>{t.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {(transactions?.length ?? 0) > show && (
              <div style={{ textAlign: 'center', padding: '10px 0' }}>
                <button className="btn btn-ghost" style={{ fontSize: 11 }}
                  onClick={() => setShow(s => s + 15)}>
                  顯示更多
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}


// ── Main Page ─────────────────────────────────────────────────────────────────

export default function InvestmentSim() {
  const { data, loading, error, refetch } = useApi(fetchSimBacktest);
  const [viewDays, setViewDays] = useState(0);

  const s        = data?.summary         ?? {};
  const curve    = data?.equity_curve    ?? [];
  const bench    = data?.benchmark_curve ?? [];
  const holdings = data?.current_holdings ?? [];
  const watchlist= data?.watchlist        ?? [];
  const txs      = data?.transactions     ?? [];
  const hasData  = curve.length >= 2;

  // Normalise equity to 100-based index for chart overlay with TWII
  const initialEquity = curve[0]?.equity ?? 1;
  const benchMap      = Object.fromEntries(bench.map(b => [b.date, b.level]));
  const displayCurve  = (viewDays > 0 ? curve.slice(-viewDays) : curve).map(pt => ({
    ...pt,
    port_idx:  parseFloat((pt.equity / initialEquity * 100).toFixed(3)),
    bench_idx: benchMap[pt.date] ?? null,
  }));

  const retColor = pos(s.total_return_pct);
  const ddColor  = (s.max_drawdown_pct ?? 0) < -5 ? PNL_NEG : (s.max_drawdown_pct ?? 0) < -2 ? CLR_AMB : 'var(--text-secondary)';

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
            訊號驅動持倉模擬 · Top-20 進場 / Top-35 退場 · 含交易成本
            {data?.period_start && ` · ${data.period_start} → ${data.period_end}`}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select value={viewDays} onChange={e => setViewDays(Number(e.target.value))}
            style={{ background: 'var(--bg-panel-2)', border: '1px solid var(--border)', borderRadius: 6, padding: '5px 10px', fontSize: 12, color: 'var(--text-primary)' }}>
            <option value={0}>全部</option>
            <option value={20}>近 20 天</option>
            <option value={40}>近 40 天</option>
          </select>
          <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={refetch}>
            🔄 刷新
          </button>
        </div>
      </div>

      {/* ── Summary Cards ── */}
      {loading ? (
        <div className="grid-4">{[0,1,2,3].map(i => <SkeletonCard key={i} />)}</div>
      ) : !hasData ? (
        <div className="panel">
          <div className="panel-body" style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-muted)' }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>📊</div>
            <div style={{ fontSize: 14, marginBottom: 8 }}>尚無回測資料</div>
            <div style={{ fontSize: 12 }}>執行每日推論後自動生成 · 需要至少 2 個交易日</div>
          </div>
        </div>
      ) : (
        <div className="grid-4">
          <StatCard
            label="累積報酬"
            value={pct(s.total_return_pct)}
            sub={s.benchmark_return_pct != null
              ? `大盤 ${pct(s.benchmark_return_pct)} / 超額 ${pct(s.excess_return_pct)}`
              : `${money(s.portfolio_value - (data?.initial_capital ?? 0))} 損益`}
            color={retColor}
          />
          <StatCard
            label="最大回撤"
            value={`${s.max_drawdown_pct?.toFixed(2)}%`}
            sub={`夏普 ${s.sharpe?.toFixed(2)} · 勝率 ${s.win_rate_pct?.toFixed(0)}%`}
            color={ddColor}
          />
          <StatCard
            label="目前持倉"
            value={`${s.current_positions} 檔`}
            sub={`已部署 ${s.deployed_pct?.toFixed(1)}% · 現金 ${((s.cash ?? 0) / 10000).toFixed(1)}萬`}
            color={CLR_BLUE}
          />
          <StatCard
            label="交易紀錄"
            value={`${s.total_trades ?? 0} 筆`}
            sub={`平均持有 ${s.avg_hold_days?.toFixed(1)} 天`}
            color="var(--text-secondary)"
          />
        </div>
      )}

      {/* ── Equity Curve vs TWII ── */}
      {hasData && (
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>💹</span> 資產曲線 vs TWII（基準值 = 100）</div>
            <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
              <span style={{ color: CLR_BLUE }}>● 機器人</span>
              {bench.length > 0 && <span style={{ color: CLR_AMB }}>● 大盤 TWII</span>}
            </div>
          </div>
          <div className="panel-body">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={displayCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'var(--text-muted)' }}
                  tickFormatter={v => v.slice(5)} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                  tickFormatter={v => v.toFixed(1)} domain={['auto', 'auto']} />
                <Tooltip content={<EquityTooltip />} />
                <ReferenceLine y={100} stroke="var(--border)" strokeDasharray="4 4" strokeWidth={1} />
                <Line type="monotone" dataKey="port_idx" name="機器人"
                  stroke={CLR_BLUE} strokeWidth={2} dot={false}
                  activeDot={{ r: 4, fill: CLR_BLUE }} />
                {bench.length > 0 && (
                  <Line type="monotone" dataKey="bench_idx" name="TWII"
                    stroke={CLR_AMB} strokeWidth={1.5} dot={false} strokeDasharray="4 2"
                    activeDot={{ r: 3, fill: CLR_AMB }} connectNulls />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Holdings + Watchlist ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <HoldingsPanel holdings={holdings} loading={loading} />
        <WatchlistPanel watchlist={watchlist} loading={loading} />
      </div>

      {/* ── IC Analysis ── */}
      <IcPanel />

      {/* ── Transactions ── */}
      <TransactionsPanel transactions={txs} loading={loading} />

      {/* ── Methodology ── */}
      <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.15)', background: 'rgba(0,212,255,0.02)' }}>
        <div className="panel-body" style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          <span style={{ fontSize: 18 }}>💡</span>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            <strong style={{ color: CLR_BLUE }}>模擬邏輯：</strong>
            Signal_Quality 排名進入 Top-20 時以當日收盤價建倉，跌出 Top-35 或超過 30 天時賣出。
            含交易成本（買入 0.15%，賣出 0.45% = 手續費 + 證交稅），最多持有 15 檔，帳戶保留 5% 現金。
            <br />
            <strong style={{ color: CLR_AMB }}>IC 分析：</strong>
            預測 Alpha_5d 與實際 5 天後報酬的 Spearman 相關係數。ICIR &gt; 0.4 為業界標準良好門檻。
            <strong style={{ color: PNL_NEG, marginLeft: 4 }}>⚠️ 僅供模型驗證參考，不構成投資建議。</strong>
          </div>
        </div>
      </div>

    </div>
  );
}
