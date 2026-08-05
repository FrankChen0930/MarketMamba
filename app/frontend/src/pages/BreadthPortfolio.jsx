import React, { useEffect, useState } from 'react';
import { fetchV62Portfolio, fetchV62Arms } from '../api/v62';

/**
 * V6.2 持股組合（20 日）
 *
 * ⚠️ 命名與文案的設計理由（不要簡化）
 * ----------------------------------
 * 這個模型**每 20 個交易日才換一次股**。回測實測：每日再平衡 −19.9%、
 * 每 20 日 +38.1%，差別在換手成本。所以這一頁刻意**不顯示「今日選股」**——
 * 那會誘導每日換股，正好是回測證明最賠錢的行為。
 * 分頁名稱把「20 日」放進去，就是為了讓「今天的清單」永遠不會被讀成
 * 「今天要買的清單」。
 *
 * 中間 19 天只顯示三件有意義的事：現行持股、再平衡倒數、漂移度
 * （持股還有幾檔在模型當前 Top50）。
 */
export default function BreadthPortfolio() {
  const [arm, setArm] = useState(null);
  const [arms, setArms] = useState([]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchV62Arms().then((d) => { setArms(d.arms || []); if (!arm) setArm(d.default); })
      .catch(() => setArms([]));
  }, []);

  useEffect(() => {
    setLoading(true);
    fetchV62Portfolio(arm).then(setData).catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [arm]);

  if (loading) return <div style={{ padding: 24, opacity: .6 }}>載入中…</div>;
  if (!data) return <div style={{ padding: 24, opacity: .6 }}>無法取得 V6.2 資料</div>;

  const spec = data.spec || {};
  const bt = spec.backtest || {};
  const notReady = !data.date;
  const missing = Object.entries(data.data_complete || {})
    .filter(([, v]) => v === false).map(([k]) => k);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* ── 模型切換（並行跑的變體）── */}
      {arms.length > 1 && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {arms.map((a) => (
            <button key={a.arm} onClick={() => setArm(a.arm)}
              style={{
                padding: '6px 14px', borderRadius: 8, cursor: 'pointer',
                border: a.arm === arm ? '1px solid #4a9eff' : '1px solid #333',
                background: a.arm === arm ? 'rgba(74,158,255,.15)' : 'transparent',
                color: a.arm === arm ? '#4a9eff' : '#999', fontSize: 13,
              }}>{a.label}</button>
          ))}
        </div>
      )}

      {notReady ? (
        <div style={{ padding: 20, borderRadius: 10, background: 'rgba(255,193,7,.08)',
                      border: '1px solid rgba(255,193,7,.3)' }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>⏳ 尚未開始累積</div>
          <div style={{ fontSize: 13, opacity: .8 }}>{data.note}</div>
        </div>
      ) : (
        <>
          {/* ── 最重要的一塊：今天到底要不要動 ── */}
          <div style={{
            padding: 18, borderRadius: 10,
            background: data.is_rebalance_day ? 'rgba(74,158,255,.1)' : 'rgba(255,255,255,.03)',
            border: data.is_rebalance_day ? '1px solid rgba(74,158,255,.4)' : '1px solid #2a2a2a',
          }}>
            {data.is_rebalance_day ? (
              <>
                <div style={{ fontWeight: 700, fontSize: 16, color: '#4a9eff' }}>
                  ★ 今天是再平衡日（第 {data.n_rebalances} 次）
                </div>
                <div style={{ fontSize: 13, opacity: .85, marginTop: 6 }}>
                  下方持股即為本次調整後的名單。距下次再平衡 {spec.rebalance_days} 個交易日。
                </div>
              </>
            ) : (
              <>
                <div style={{ fontWeight: 700, fontSize: 16 }}>
                  今天不是再平衡日 — 不需要任何動作
                </div>
                <div style={{ fontSize: 13, opacity: .85, marginTop: 6 }}>
                  下方是<strong>現行持股</strong>，<strong>不是今天建議買進的名單</strong>。
                  距下次再平衡 <strong>{data.days_to_next}</strong> 個交易日
                  （已過 {data.days_since_rebalance} 天）。
                </div>
                <div style={{ fontSize: 12, opacity: .6, marginTop: 8 }}>
                  為什麼不顯示「今日選股」：這個模型每 {spec.rebalance_days} 個交易日換一次股。
                  同一份訊號改成每日再平衡，回測是 −19.9%（每 20 日是 +38.1%）——
                  差別全在換手成本。中間的分數變動不是交易訊號。
                </div>
              </>
            )}
          </div>

          {/* ── 資料完整性（缺漏一定要看得到）── */}
          {missing.length > 0 && (
            <div style={{ padding: 14, borderRadius: 10, background: 'rgba(255,82,82,.08)',
                          border: '1px solid rgba(255,82,82,.35)', fontSize: 13 }}>
              ⚠️ <strong>當日資料缺漏</strong>：{missing.join('、')}。
              這一天的分數可能不可靠，判讀實戰紀錄時要把它排除。
            </div>
          )}

          {/* ── 三個數字 ── */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))',
                        gap: 12 }}>
            {[['持股檔數', `${data.holdings.length} / ${spec.n}`, null],
              ['仍在模型 Top50', `${data.in_top_n ?? '—'} 檔`, '漂移度：越低代表名單越過時'],
              ['距下次再平衡', `${data.days_to_next} 個交易日`, null],
              ['已再平衡次數', `${data.n_rebalances} 次`, `上次 ${data.last_rebalance || '—'}`],
            ].map(([label, val, hint]) => (
              <div key={label} style={{ padding: 14, borderRadius: 10, background: 'rgba(255,255,255,.03)',
                                        border: '1px solid #2a2a2a' }}>
                <div style={{ fontSize: 12, opacity: .6 }}>{label}</div>
                <div style={{ fontSize: 20, fontWeight: 700, marginTop: 4 }}>{val}</div>
                {hint && <div style={{ fontSize: 11, opacity: .45, marginTop: 4 }}>{hint}</div>}
              </div>
            ))}
          </div>

          {/* ── 持股 ── */}
          <div style={{ borderRadius: 10, border: '1px solid #2a2a2a', overflow: 'hidden' }}>
            <div style={{ padding: '10px 14px', background: 'rgba(255,255,255,.03)',
                          fontSize: 13, fontWeight: 600 }}>
              現行持股（{data.date}）
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ opacity: .55, textAlign: 'left' }}>
                    <th style={{ padding: '8px 14px' }}>#</th>
                    <th style={{ padding: '8px 14px' }}>代號</th>
                    <th style={{ padding: '8px 14px' }}>名稱</th>
                    <th style={{ padding: '8px 14px' }}>權重</th>
                    <th style={{ padding: '8px 14px' }}>模型當前排名</th>
                  </tr>
                </thead>
                <tbody>
                  {data.holdings.map((h, i) => (
                    <tr key={h.stock_id} style={{ borderTop: '1px solid #222' }}>
                      <td style={{ padding: '7px 14px', opacity: .5 }}>{i + 1}</td>
                      <td style={{ padding: '7px 14px', fontFamily: 'monospace' }}>{h.stock_id}</td>
                      <td style={{ padding: '7px 14px' }}>{h.name}</td>
                      <td style={{ padding: '7px 14px' }}>{(h.weight * 100).toFixed(2)}%</td>
                      <td style={{ padding: '7px 14px',
                                   color: h.in_top_n ? '#4caf50' : '#ff9800' }}>
                        {h.current_rank ?? '—'}{h.in_top_n ? '' : '（已跌出 Top50）'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── 規格與限制（永遠顯示，包含尚未開始時）── */}
      <div style={{ padding: 16, borderRadius: 10, background: 'rgba(255,255,255,.02)',
                    border: '1px solid #2a2a2a', fontSize: 13 }}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>規格與回測（{data.label}）</div>
        <div style={{ opacity: .8, lineHeight: 1.7 }}>
          N={spec.n} 檔等權｜緩衝 k={spec.k}（跌出前 {Math.round(spec.n * spec.k)} 名才換掉）｜
          每 {spec.rebalance_days} 個交易日再平衡<br />
          回測 {bt.window}：年化 <strong>{(bt.ann_return * 100).toFixed(1)}%</strong>｜
          Sharpe {bt.sharpe}｜最大回撤 {(bt.max_drawdown * 100).toFixed(1)}%｜
          decile spread Sharpe {bt.decile_spread_sharpe}
        </div>
        <div style={{ marginTop: 10, fontSize: 12, opacity: .6 }}>
          <strong>誠實的限制</strong>：
          <ul style={{ margin: '4px 0 0 18px', padding: 0, lineHeight: 1.7 }}>
            {(spec.caveats || []).map((c) => <li key={c}>{c}</li>)}
          </ul>
        </div>
      </div>
    </div>
  );
}
