import React, { useState, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, BarChart, Bar, Cell, Legend,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchSimBacktest, fetchIcAnalysis, fetchScannerBacktest } from '../api/sim';
import { fetchDualIc } from '../api/dual';
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

// ── Strategy Guide ────────────────────────────────────────────────────────────

function StrategyGuide() {
  const [open, setOpen] = useState(false);

  const Rule = ({ icon, title, items }) => (
    <div style={{
      flex: '1 1 180px',
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '14px 16px',
    }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 10 }}>
        <span style={{ marginRight: 6 }}>{icon}</span>{title}
      </div>
      <ul style={{ margin: 0, padding: 0, listStyle: 'none' }}>
        {items.map((item, i) => (
          <li key={i} style={{
            fontSize: 12, color: 'var(--text-secondary)',
            lineHeight: 1.8, display: 'flex', gap: 6, alignItems: 'flex-start',
          }}>
            <span style={{ color: 'var(--text-muted)', flexShrink: 0, marginTop: 1 }}>›</span>
            <span dangerouslySetInnerHTML={{ __html: item }} />
          </li>
        ))}
      </ul>
    </div>
  );

  const Concept = ({ term, color, children }) => (
    <div style={{
      background: 'rgba(255,255,255,0.02)',
      border: `1px solid ${color}33`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 6,
      padding: '12px 16px',
      flex: '1 1 280px',
    }}>
      <div style={{ fontSize: 12, fontWeight: 600, color, marginBottom: 6 }}>{term}</div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>{children}</div>
    </div>
  );

  return (
    <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.15)' }}>
      <div
        className="panel-header"
        onClick={() => setOpen(o => !o)}
        style={{ cursor: 'pointer', userSelect: 'none' }}
      >
        <div className="panel-title"><span>⚙️</span> 投資策略說明</div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            訊號驅動 · 含交易成本 · 量化選股驗證
          </span>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>▼</span>
        </div>
      </div>

      {open && (
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* Rule blocks */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <Rule icon="🟢" title="進場條件" items={[
              `Signal_Quality 排名進入全市場 <strong style="color:var(--positive)">Top-20</strong>`,
              `以當日收盤價建倉（信號日即執行）`,
              `最多同時持有 <strong>15 檔</strong>，有空位才建新倉`,
              `單次建倉至少佔淨值 <strong>2%</strong> 以上才執行`,
            ]} />
            <Rule icon="🔴" title="退場條件（任一觸發）" items={[
              `排名跌出 <strong style="color:var(--negative)">Top-35</strong>（訊號弱化）`,
              `Signal_Quality &lt; 0.5（淨 Alpha 轉負）`,
              `持有超過 <strong>30 個交易日</strong>（時間停損）`,
              `退場以當日收盤價計算損益`,
            ]} />
            <Rule icon="⚖️" title="倉位管理" items={[
              `按 <strong>Kelly 加權比例</strong>（Signal_Quality 比例分配）`,
              `單股上限 <strong>12%</strong>，防止過度集中`,
              `帳戶保留至少 <strong>5% 現金</strong>緩衝`,
              `新倉依可用資金均分，不強制賣現有持股`,
            ]} />
            <Rule icon="💸" title="交易成本" items={[
              `買進：手續費 <strong>0.15%</strong>`,
              `賣出：手續費 0.15% + 證交稅 0.30% = <strong>0.45%</strong>`,
              `一來一回 round-trip 合計約 <strong>0.6%</strong>`,
              `每次換倉都完整計算，不低估摩擦成本`,
            ]} />
          </div>

          {/* Flow diagram */}
          <div style={{
            background: 'rgba(255,255,255,0.02)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: '14px 20px',
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 10 }}>整體流程</div>
            <div style={{ display: 'flex', gap: 0, alignItems: 'center', flexWrap: 'wrap', rowGap: 8 }}>
              {[
                { label: 'Mamba+GAT 推論', sub: '每日 17:00', color: CLR_BLUE },
                { arrow: true },
                { label: 'Signal_Quality 排名', sub: 'Alpha / 不確定度', color: CLR_BLUE },
                { arrow: true },
                { label: '進退場判斷', sub: 'Top-20 / Top-35', color: CLR_AMB },
                { arrow: true },
                { label: '倉位計算', sub: 'Kelly 加權', color: CLR_AMB },
                { arrow: true },
                { label: '執行交易', sub: '含成本紀錄', color: PNL_POS },
                { arrow: true },
                { label: '績效追蹤', sub: 'vs TWII / IC', color: PNL_POS },
              ].map((step, i) =>
                step.arrow ? (
                  <span key={i} style={{ color: 'var(--text-muted)', fontSize: 14, padding: '0 6px' }}>→</span>
                ) : (
                  <div key={i} style={{
                    background: `${step.color}15`,
                    border: `1px solid ${step.color}40`,
                    borderRadius: 6,
                    padding: '6px 12px',
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: step.color }}>{step.label}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{step.sub}</div>
                  </div>
                )
              )}
            </div>
          </div>

          {/* Concept explanations */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <Concept term="Signal_Quality 是什麼？" color={CLR_BLUE}>
              模型對每支股票計算的「風險調整後超額報酬信心度」。
              公式：<code style={{ color: CLR_AMB, fontSize: 11 }}>Signal_Quality = Net_Alpha_20d / (Uncertainty + ε)</code>，
              其中 Net_Alpha 是扣除滑點後的預期 20 天超額報酬，Uncertainty 由 MC-Dropout（30 次採樣）估算。
              數值越高代表「預期報酬強且模型信心高」，是排名的核心依據。
            </Concept>
            <Concept term="IC / ICIR 是什麼？" color={CLR_AMB}>
              IC（Information Coefficient）= 預測 Alpha 排名與實際報酬排名的 Spearman 相關係數，
              衡量「模型預測方向是否準確」。ICIR = IC 均值 / IC 標準差，衡量「預測是否穩定」。
              業界標準：IC &gt; 0.02 為有效，ICIR &gt; 0.4 為良好。
              IC &gt; 0 的天數比例（目前 72.7%）代表模型大多數時候方向是對的。
            </Concept>
            <Concept term="觀察清單的意義" color="var(--text-secondary)">
              排名介於第 21–35 名之間、尚未建倉的股票。這些股票距離進場門檻（Top-20）只差一步，
              若隔天排名上升便可能觸發買進訊號。觀察清單讓你預判「明天可能進場的候選股」，
              也可以搭配自己的基本面判斷進行手動參考。
            </Concept>
          </div>

          <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: '8px 12px', background: 'rgba(255,71,87,0.05)', borderRadius: 6, border: '1px solid rgba(255,71,87,0.15)' }}>
            ⚠️ 本模擬以歸檔 df_kelly.csv 回測，假設信號日即可以收盤價成交，不考慮市場衝擊與流動性限制。
            所有數據僅供量化模型驗證使用，不構成任何投資建議。
          </div>

        </div>
      )}
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


// ── Scanner Robot ─────────────────────────────────────────────────────────────

const TRAILING_TIERS = [
  { threshold: 0.15, stopPct:  10, label: '峰值≥+15%' },
  { threshold: 0.10, stopPct:   6, label: '峰值≥+10%' },
  { threshold: 0.05, stopPct:   2, label: '峰值≥+5%'  },
  { threshold: -99,  stopPct:  -5, label: '預設'       },
];

function ScannerHoldingRow({ h }) {
  const conds = [
    { met: h.rank_stability_met,    label: '排名穩定' },
    { met: h.high_confidence_met,   label: '高信心'   },
    { met: h.relative_low_met,      label: '相對低點' },
    { met: h.institutional_buy_met, label: '機構買入' },
  ];
  const stopColor = h.trailing_stop_price >= h.current_price
    ? 'var(--negative)'
    : h.trailing_stop_pct >= 0
    ? 'var(--positive)'
    : 'var(--accent-amber)';

  return (
    <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)' }}>
      {/* Top row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <div>
          <span style={{ fontWeight: 700, fontSize: 14 }}>{h.ticker}</span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>
            入場 {h.entry_date?.slice(5)} @ {h.entry_price?.toFixed(2)}
          </span>
          <span style={{ marginLeft: 8, fontSize: 10, background: 'rgba(0,212,255,0.12)', color: 'var(--accent-blue)', borderRadius: 99, padding: '1px 7px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {h.entry_score}分 · {h.conditions_met}/4
          </span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className="mono" style={{ fontSize: 14, fontWeight: 700, color: h.pnl_pct >= 0 ? 'var(--positive)' : 'var(--negative)' }}>
            {h.pnl_pct >= 0 ? '+' : ''}{h.pnl_pct?.toFixed(2)}%
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            現價 {h.current_price?.toFixed(2)} · {h.days_held}天
          </div>
        </div>
      </div>
      {/* Conditions */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
        {conds.map((c, i) => (
          <span key={i} style={{ fontSize: 11, color: c.met ? 'var(--positive)' : 'var(--text-muted)' }}>
            {c.met ? '✅' : '❌'} {c.label}
          </span>
        ))}
      </div>
      {/* Trailing stop */}
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
        止損：
        <span style={{ color: stopColor, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
          {h.trailing_stop_price?.toFixed(2)}
        </span>
        <span style={{ marginLeft: 4 }}>
          ({h.trailing_stop_pct >= 0 ? '+' : ''}{h.trailing_stop_pct}% 鎖利)
        </span>
        {h.peak_pnl_pct > 0 && (
          <span style={{ marginLeft: 8, color: 'var(--positive)' }}>
            峰值 +{h.peak_pnl_pct?.toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}

function ScannerHoldingsPanel({ holdings, loading }) {
  if (loading) return (
    <div className="panel">
      <div className="panel-header"><div className="panel-title"><span>💼</span> Scanner 持倉</div></div>
      <div className="panel-body"><SkeletonBlock height={160} /></div>
    </div>
  );
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>💼</span> Scanner 持倉</div>
        <span className="badge badge-neutral">{holdings.length} 檔</span>
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        {!holdings.length ? (
          <div style={{ padding: '24px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            目前無持倉 · 等待 Scanner BUY 訊號進場
          </div>
        ) : (
          holdings.map(h => <ScannerHoldingRow key={h.ticker} h={h} />)
        )}
      </div>
    </div>
  );
}

function ScannerWatchlistPanel({ watchlist, loading }) {
  if (loading) return null;
  if (!watchlist?.length) return null;
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>👀</span> 觀察清單</div>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Scanner watch_list · 差 1 個條件</span>
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        <table className="data-table">
          <thead><tr><th>代號</th><th style={{ textAlign: 'right' }}>觀察天數</th></tr></thead>
          <tbody>
            {watchlist.map(w => (
              <tr key={w.ticker}>
                <td style={{ fontWeight: 600, fontSize: 13 }}>{w.ticker}</td>
                <td className="mono" style={{ textAlign: 'right', fontSize: 12, color: 'var(--accent-amber)' }}>{w.days_in_watch} 天</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ScannerStrategyGuide() {
  const [open, setOpen] = useState(false);
  const S = { row: { display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '6px 12px', borderRadius: 6, marginBottom: 4 } };

  return (
    <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.15)' }}>
      <div className="panel-header" onClick={() => setOpen(o => !o)} style={{ cursor: 'pointer', userSelect: 'none' }}>
        <div className="panel-title"><span>⚙️</span> Scanner 策略說明</div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>4條件加權入場 · Trailing Stop退場</span>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>▼</span>
        </div>
      </div>

      {open && (
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {/* Entry conditions */}
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--positive)', marginBottom: 10 }}>🟢 入場條件（加權評分，正常市場 ≥55分）</div>
            <table className="data-table" style={{ width: '100%' }}>
              <thead><tr><th>條件</th><th>邏輯</th><th style={{ textAlign: 'right' }}>配分</th></tr></thead>
              <tbody>
                {[
                  ['排名穩定性', 'Top 10 連續 ≥2 天 或 Top 50 連續 ≥3 天', 30],
                  ['高信心', 'Uncertainty < 當日 Q30 分位數', 25],
                  ['機構淨買入', '外資/投信連續 ≥2 天淨買', 25],
                  ['相對低點', 'RSI < 40 或 價格 < MA20', 20],
                ].map(([name, logic, pts]) => (
                  <tr key={name}>
                    <td style={{ fontWeight: 600 }}>{name}</td>
                    <td style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{logic}</td>
                    <td className="mono" style={{ textAlign: 'right', color: 'var(--accent-amber)', fontWeight: 700 }}>{pts}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, padding: '6px 10px', background: 'rgba(255,165,0,0.05)', borderRadius: 6 }}>
              保守模式（TWII &lt; MA60）：門檻提高至 ≥70 分
            </div>
          </div>

          {/* Exit conditions */}
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--negative)', marginBottom: 10 }}>🔴 退場條件（任一觸發）</div>
            {[
              { cond: 'Scanner EXIT 訊號（排名掉出 Top 50 連 2 天）', color: 'var(--negative)' },
              { cond: 'Scanner EXIT 訊號（外資連續 3 天淨賣出）', color: 'var(--negative)' },
              { cond: 'Trailing Stop 觸及（見下表）', color: 'var(--negative)' },
            ].map(r => (
              <div key={r.cond} style={{ ...S.row, background: 'rgba(255,71,87,0.04)' }}>
                <span style={{ color: 'var(--text-secondary)' }}>🔴 {r.cond}</span>
              </div>
            ))}
          </div>

          {/* Trailing stop tiers */}
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent-amber)', marginBottom: 8 }}>📉 Trailing Stop 機制（以峰值報酬決定止損線）</div>
            <table className="data-table" style={{ width: '100%' }}>
              <thead><tr><th>峰值報酬</th><th>止損線位置</th></tr></thead>
              <tbody>
                {[
                  ['< +5%', '固定 −5%（成本價）'],
                  ['≥ +5%', '成本 +2%（開始鎖利）'],
                  ['≥ +10%', '成本 +6%'],
                  ['≥ +15%', '成本 +10%'],
                ].map(([ret, stop]) => (
                  <tr key={ret}>
                    <td>{ret}</td>
                    <td className="mono" style={{ color: 'var(--accent-amber)', fontWeight: 600 }}>{stop}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
              ⚠ 止損線只會往上調整（鎖利），不會因為報酬下滑而向下重置。
            </div>
          </div>

          {/* Position sizing */}
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent-blue)', marginBottom: 8 }}>💰 倉位管理</div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
              按 Scanner <code style={{ color: 'var(--accent-amber)', fontSize: 11 }}>suggested_weight</code> 比例分配；單股上限 15%；最多同時持有 10 檔；維持 5% 現金緩衝。
            </div>
          </div>

          <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: '8px 12px', background: 'rgba(255,71,87,0.05)', borderRadius: 6, border: '1px solid rgba(255,71,87,0.15)' }}>
            ⚠️ 本模擬從首次推論日開始累積，假設以當日收盤價成交，不考慮市場衝擊。僅供量化策略驗證，不構成任何投資建議。
          </div>
        </div>
      )}
    </div>
  );
}

function ScannerRobotTab() {
  const { data, loading, error, refetch } = useApi(fetchScannerBacktest);
  const [viewDays, setViewDays] = useState(0);

  const s          = data?.summary        ?? {};
  const curve      = data?.equity_curve   ?? [];
  const bench      = data?.benchmark_curve ?? [];
  const holdings   = data?.current_holdings ?? [];
  const watchlist  = data?.watchlist       ?? [];
  const txs        = data?.transactions    ?? [];
  const meta       = data?.scanner_meta   ?? {};
  const hasData    = curve.length >= 2;
  const tradingDays = data?.trading_days ?? 0;

  const initialEquity = curve[0]?.equity ?? 1;
  const benchMap      = Object.fromEntries(bench.map(b => [b.date, b.level]));
  const displayCurve  = (viewDays > 0 ? curve.slice(-viewDays) : curve).map(pt => ({
    ...pt,
    port_idx:  parseFloat((pt.equity / initialEquity * 100).toFixed(3)),
    bench_idx: benchMap[pt.date] ?? null,
  }));

  const retColor  = pos(s.total_return_pct);
  const ddColor   = (s.max_drawdown_pct ?? 0) < -5 ? PNL_NEG : (s.max_drawdown_pct ?? 0) < -2 ? CLR_AMB : 'var(--text-secondary)';
  const regimeColor = meta.market_regime === 'CAUTIOUS' ? CLR_AMB : 'var(--positive)';

  // Not yet available
  if (!loading && (error || data?.error)) {
    return (
      <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.2)' }}>
        <div className="panel-body" style={{ textAlign: 'center', padding: '40px 0' }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>🌱</div>
          <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
            Scanner 機器人尚未啟動
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.8 }}>
            今日 17:00 推論完成後將自動建立機器人並開始追蹤<br />
            機器人將依 Scanner 每日 BUY/EXIT 訊號自動操作
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'flex-end' }}>
        <select value={viewDays} onChange={e => setViewDays(Number(e.target.value))}
          style={{ background: 'var(--bg-panel-2)', border: '1px solid var(--border)', borderRadius: 6, padding: '5px 10px', fontSize: 12, color: 'var(--text-primary)' }}>
          <option value={0}>全部</option>
          <option value={20}>近 20 天</option>
          <option value={40}>近 40 天</option>
        </select>
        <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={refetch}>🔄 刷新</button>
      </div>

      {/* Summary cards */}
      {loading ? (
        <div className="grid-4">{[0,1,2,3].map(i => <SkeletonCard key={i} />)}</div>
      ) : (
        <div className="grid-4">
          <StatCard label="累積報酬" value={tradingDays > 0 ? `${s.total_return_pct >= 0 ? '+' : ''}${s.total_return_pct?.toFixed(2)}%` : '—'}
            sub={s.benchmark_return_pct != null ? `大盤 ${s.benchmark_return_pct >= 0 ? '+' : ''}${s.benchmark_return_pct?.toFixed(2)}% / 超額 ${(s.excess_return_pct ?? 0) >= 0 ? '+' : ''}${s.excess_return_pct?.toFixed(2)}%` : '尚無基準資料'}
            color={tradingDays > 0 ? retColor : 'var(--text-muted)'} />
          <StatCard label="最大回撤" value={tradingDays > 0 ? `${s.max_drawdown_pct?.toFixed(2)}%` : '—'}
            sub={tradingDays > 0 ? `夏普 ${s.sharpe?.toFixed(2)} · 勝率 ${s.win_rate_pct?.toFixed(0)}%` : `已追蹤 ${tradingDays} 個交易日`}
            color={tradingDays > 0 ? ddColor : 'var(--text-muted)'} />
          <StatCard label="目前持倉" value={`${s.current_positions ?? 0} 檔`}
            sub={`已部署 ${s.deployed_pct?.toFixed(1) ?? 0}% · 現金 ${((s.cash ?? 0) / 10000).toFixed(1)}萬`}
            color={CLR_BLUE} />
          <div className="stat-card" style={{ borderColor: regimeColor }}>
            <div className="label">大盤環境</div>
            <div className="value" style={{ color: regimeColor }}>{meta.market_regime === 'CAUTIOUS' ? '🟡 保守模式' : '🟢 正常市場'}</div>
            <div className="sub">{meta.entry_threshold ?? '—'} · {meta.scan_date ?? '—'}</div>
          </div>
        </div>
      )}

      {/* Equity curve */}
      {!loading && (hasData ? (
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>💹</span> 資產曲線 vs TWII（基準值 = 100）</div>
            <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
              <span style={{ color: 'var(--positive)' }}>● Scanner 機器人</span>
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
                <Line type="monotone" dataKey="port_idx" name="Scanner機器人"
                  stroke="var(--positive)" strokeWidth={2} dot={false}
                  activeDot={{ r: 4, fill: 'var(--positive)' }} />
                {bench.length > 0 && (
                  <Line type="monotone" dataKey="bench_idx" name="TWII"
                    stroke={CLR_AMB} strokeWidth={1.5} dot={false} strokeDasharray="4 2"
                    activeDot={{ r: 3, fill: CLR_AMB }} connectNulls />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.15)' }}>
          <div className="panel-body" style={{ textAlign: 'center', padding: '28px 0' }}>
            <div style={{ fontSize: 26, marginBottom: 10 }}>🌱</div>
            <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 4 }}>機器人已啟動 · 正在累積歷史資料</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              已追蹤 {tradingDays} 個交易日 · 需要至少 2 個交易日才能繪製曲線
            </div>
          </div>
        </div>
      ))}

      {/* Holdings */}
      <ScannerHoldingsPanel holdings={holdings} loading={loading} />

      {/* Watchlist */}
      <ScannerWatchlistPanel watchlist={watchlist} loading={loading} />

      {/* Strategy guide */}
      <ScannerStrategyGuide />

      {/* Transactions */}
      <TransactionsPanel transactions={txs} loading={loading} />
    </div>
  );
}


// ── Dual Model 真實市場效益驗證（Tab）───────────────────────────────────────────

function DualHorizonBlock({ label, sub, summary, series }) {
  const n = summary?.n_days ?? 0;
  const ready = n >= 20;
  const icColor  = summary?.mean_ic == null ? 'var(--text-muted)' : pos(summary.mean_ic);
  const topColor = summary?.mean_top50_excess_pct == null ? 'var(--text-muted)' : pos(summary.mean_top50_excess_pct);

  return (
    <div style={{ padding: '12px 14px', background: 'var(--bg-panel-2)', borderRadius: 8, minWidth: 220, flex: 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 600 }}>{label}</div>
        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{sub}</span>
      </div>

      {n === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '8px 0' }}>
          尚未累積足夠時間（需等到對應天數後的價格資料才能計算）
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 8 }}>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>平均 IC</div>
              <div className="mono" style={{ fontSize: 15, fontWeight: 600, color: icColor }}>
                {summary.mean_ic >= 0 ? '+' : ''}{summary.mean_ic?.toFixed(4)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>ICIR</div>
              <div className="mono" style={{ fontSize: 15, fontWeight: 600 }}>{summary.icir?.toFixed(2)}</div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Top50 實現超額</div>
              <div className="mono" style={{ fontSize: 15, fontWeight: 600, color: topColor }}>
                {summary.mean_top50_excess_pct != null
                  ? `${summary.mean_top50_excess_pct >= 0 ? '+' : ''}${summary.mean_top50_excess_pct.toFixed(2)}%`
                  : '—'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>樣本天數</div>
              <div className="mono" style={{ fontSize: 15, fontWeight: 600 }}>{n} 天</div>
            </div>
          </div>

          {!ready && (
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', marginBottom: 8, padding: '5px 8px', background: 'rgba(255,255,255,0.03)', borderRadius: 5 }}>
              ⏳ 樣本仍少（{n} 天），建議累積 20+ 天再參考結論
            </div>
          )}

          {series?.length >= 3 && (
            <ResponsiveContainer width="100%" height={60}>
              <BarChart data={series} barCategoryGap="20%">
                <XAxis dataKey="pred_date" tick={false} />
                <YAxis hide />
                <Tooltip content={<IcTooltip />} />
                <ReferenceLine y={0} stroke="var(--border)" />
                <Bar dataKey="ic" radius={[2, 2, 0, 0]}>
                  {series.map((pt, i) => (
                    <Cell key={i} fill={pt.ic >= 0 ? PNL_POS : PNL_NEG} opacity={0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </>
      )}
    </div>
  );
}

function DualIcPanel() {
  const { data, loading, error, refetch } = useApi(fetchDualIc);
  const h = data?.horizon_summary ?? {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="panel">
        <div className="panel-body" style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          這裡驗證的是 <b>V6.2 雙模型</b>（短線 v6_short.pt + 趨勢 v6_trend.pt）用真實市場走勢反算的效益，
          回答「照這組排名選股會不會賺錢」；跟目前真倉使用的 V6.1（20d 目標）是分開的兩件事，互不影響。
          資料每日自動累積，不需要手動操作，樣本量會隨時間自然增加。
        </div>
      </div>

      {error ? (
        <ApiError message={error} onRetry={refetch} />
      ) : loading ? (
        <div className="panel"><div className="panel-body"><SkeletonBlock height={200} /></div></div>
      ) : (
        <>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>⚡</span> 短線模型（5d / 10d）</div>
              {data?.archive_count_short != null && (
                <span className="badge badge-neutral" style={{ fontSize: 10 }}>archive {data.archive_count_short} 天</span>
              )}
            </div>
            <div className="panel-body" style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <DualHorizonBlock label="5 天"  sub="Score_5d / SQ_5d"  summary={h['5d']}  series={data?.ic_series_5d} />
              <DualHorizonBlock label="10 天" sub="Score_10d / SQ_5d" summary={h['10d']} series={data?.ic_series_10d} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>📈</span> 趨勢模型（20d / 60d）</div>
              {data?.archive_count_trend != null && (
                <span className="badge badge-neutral" style={{ fontSize: 10 }}>archive {data.archive_count_trend} 天</span>
              )}
            </div>
            <div className="panel-body" style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <DualHorizonBlock label="20 天" sub="Score_20d / SQ_20d" summary={h['20d']} series={data?.ic_series_20d} />
              <DualHorizonBlock label="60 天" sub="Score_60d / SQ_20d" summary={h['60d']} series={data?.ic_series_60d} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function InvestmentSim() {
  const [activeTab, setActiveTab] = useState('alpha');
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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* ── Header ── */}
      <div className="page-header">
        <div>
          <div className="page-title">🤖 投資模擬機器人</div>
          <div className="page-subtitle">量化訊號驅動持倉模擬 · 含交易成本</div>
        </div>
      </div>

      {/* ── Tab Switcher ── */}
      <div style={{ display: 'flex', gap: 6, background: 'var(--bg-panel-2)', padding: '5px', borderRadius: 10, width: 'fit-content', border: '1px solid var(--border)' }}>
        {[
          { key: 'alpha',   icon: '🤖', label: 'Alpha 機器人',   sub: 'SQ 排名驅動' },
          { key: 'scanner', icon: '🎯', label: 'Scanner 機器人', sub: '4條件訊號驅動' },
          { key: 'dual',    icon: '🔬', label: '雙模型驗證',     sub: '真實市場效益追蹤' },
        ].map(tab => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key)}
            style={{
              padding: '8px 20px', borderRadius: 7, border: 'none', cursor: 'pointer',
              background: activeTab === tab.key ? 'var(--bg-panel)' : 'transparent',
              color: activeTab === tab.key ? 'var(--accent-blue)' : 'var(--text-muted)',
              fontWeight: activeTab === tab.key ? 600 : 400,
              boxShadow: activeTab === tab.key ? 'var(--shadow)' : 'none',
              transition: 'all 0.15s', textAlign: 'left',
            }}>
            <div style={{ fontSize: 13 }}>{tab.icon} {tab.label}</div>
            <div style={{ fontSize: 10, color: activeTab === tab.key ? 'var(--text-secondary)' : 'var(--text-muted)', fontWeight: 400, marginTop: 1 }}>{tab.sub}</div>
          </button>
        ))}
      </div>

      {/* ── Scanner Tab ── */}
      {activeTab === 'scanner' && <ScannerRobotTab />}

      {/* ── Dual Model Validation Tab ── */}
      {activeTab === 'dual' && <DualIcPanel />}

      {/* ── Alpha Tab (existing content below) ── */}
      {activeTab === 'alpha' && (<>

      {/* ── Alpha Header controls ── */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'flex-end' }}>
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

      {/* ── Summary Cards ── */}
      {error ? (
        <ApiError message={error} onRetry={refetch} />
      ) : loading ? (
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

      {/* ── Strategy Guide ── */}
      <StrategyGuide />

      </>)}
    </div>
  );
}
