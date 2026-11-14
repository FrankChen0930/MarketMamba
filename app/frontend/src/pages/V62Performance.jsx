import React, { useEffect, useState } from 'react';
import { fetchV62Performance } from '../api/v62';

/**
 * V6.2 前瞻績效比較（真實 out-of-sample）
 *
 * ⚠️ 這一頁與「回測年化」是兩種不同的東西，設計上刻意分開講
 * ------------------------------------------------------------
 *   回測 = 582 天單一窗、單一 seed、事後算的
 *   前瞻 = 每天收盤後決定持股、沒有任何事後資訊，但**樣本很小**
 * 兩者放同一列並排是為了看差距，不是為了比大小——所以「差」那一欄
 * 一定要附上雜訊底線（±6pp），否則小樣本的隨機波動會被讀成「模型退步」。
 *
 * ⚠️ 樣本不足時**不顯示年化與 Sharpe**（不是灰掉，是整欄不給）
 * ------------------------------------------------------------
 * 把 12 天的報酬用 252/n 外推成年化，會得到 ±100% 這種數字，
 * 而它看起來跟回測值一樣「像個結論」。灰掉還是會被讀，所以直接不給，
 * 只顯示累積報酬與已累積天數。門檻由後端的 min_days_for_any_claim 決定
 * （目前 20 個交易日），**不在前端寫死**。
 */
export default function V62Performance() {
  const [d, setD] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetchV62Performance().then(setD).catch(() => setErr(true))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 24, opacity: .6 }}>載入中…</div>;
  if (err || !d) return <div style={{ padding: 24, opacity: .6 }}>無法取得前瞻績效</div>;

  const models = d.models || {};
  const names = Object.keys(models);

  if (d.not_started || names.length === 0) {
    return (
      <div style={{ padding: 20, borderRadius: 10, background: 'rgba(255,193,7,.08)',
                    border: '1px solid rgba(255,193,7,.3)' }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>⏳ 尚未開始累積前瞻紀錄</div>
        <div style={{ fontSize: 13, opacity: .8, lineHeight: 1.7 }}>
          {d.note || '第一次再平衡之後才會出現。'}<br />
          前瞻績效是<strong>用時間換來的</strong>——回測買不到，只能一天一天累積。
        </div>
      </div>
    );
  }

  const nDays = d.n_days || 0;
  const minDays = d.min_days_for_any_claim ?? 20;
  const noise = d.noise_floor_pp ?? 6;
  const enough = nDays >= minDays;

  const FAM_ORDER = ['main_5d', 'main_10d', 'ckpt', 'ablation', 'baseline'];
  const FAM_LABEL = {
    main_5d: '上線模型 · 5d 頭', main_10d: '上線模型 · 10d 頭',
    ckpt: '獨立訓練 checkpoint', ablation: 'F6 消融對照組',
    baseline: 'B 類經典模型對照組', _other: '其他',
  };
  const TIER_DOT = {
    primary: '#4caf50', equivalent: '#ffc107',
    inferior: '#ff5252', incomparable: '#9e9e9e',
  };

  // family 沒帶過來時（舊 json）不要整組消失——歸到「其他」，寧可多一列也不要靜默漏掉
  const famOf = (m) => (FAM_ORDER.includes(m.family) ? m.family : '_other');
  const groups = [...FAM_ORDER, '_other']
    .map((f) => [f, names.filter((n) => famOf(models[n]) === f)])
    .filter(([, ns]) => ns.length > 0);

  const pct = (x) => `${(x * 100).toFixed(1)}%`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* ── 樣本量：最重要的一件事，放最上面 ── */}
      <div style={{
        padding: 16, borderRadius: 10,
        background: enough ? 'rgba(255,255,255,.03)' : 'rgba(255,193,7,.08)',
        border: enough ? '1px solid #2a2a2a' : '1px solid rgba(255,193,7,.35)',
      }}>
        <div style={{ fontWeight: 700, marginBottom: 6 }}>
          {enough
            ? `📈 已累積 ${nDays} 個交易日的前瞻紀錄`
            : `⏳ 只累積了 ${nDays} 個交易日 — 還不能下任何結論`}
        </div>
        <div style={{ fontSize: 13, opacity: .82, lineHeight: 1.7 }}>
          {enough ? (
            <>
              這些是<strong>真實 out-of-sample</strong>：每一天的持股都是那天收盤後
              決定的，沒有任何事後資訊。但組合層 N=50 的雜訊底線約 <strong>±{noise}pp
              年化</strong>，而年化本身還帶著自己的 <strong>±標準誤</strong>
              （表上有標）——兩道都跨過才算差距，標 <code>*</code> 的沒跨過。
            </>
          ) : (
            <>
              門檻是 {minDays} 個交易日（一季）。在那之前<strong>不顯示年化與
              Sharpe</strong>——把 {nDays} 天的報酬外推成年化會得到荒謬的數字
              （實測 30 天可以算出「年化 −52%，誤差 ±126pp」），而它看起來會跟
              回測值一樣「像個結論」。現在這張表只用來確認<strong>管線有在跑</strong>。
            </>
          )}
        </div>
      </div>

      {/* ── 逐組合表格，按 family 分組 ── */}
      {groups.map(([fam, ns]) => (
        <div key={fam} style={{ borderRadius: 10, border: '1px solid #2a2a2a',
                                overflow: 'hidden' }}>
          <div style={{ padding: '10px 14px', background: 'rgba(255,255,255,.03)',
                        fontSize: 13, fontWeight: 600 }}>
            {FAM_LABEL[fam]}<span style={{ opacity: .5, fontWeight: 400 }}>
              （{ns.length} 個）</span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ opacity: .55, textAlign: 'left' }}>
                  <th style={{ padding: '8px 14px' }}>組合</th>
                  <th style={{ padding: '8px 14px' }}>天數</th>
                  <th style={{ padding: '8px 14px' }}>累積報酬</th>
                  {enough && <>
                    <th style={{ padding: '8px 14px' }}>年化</th>
                    <th style={{ padding: '8px 14px' }}>Sharpe</th>
                    <th style={{ padding: '8px 14px' }}>MDD</th>
                  </>}
                  <th style={{ padding: '8px 14px' }}>再平衡</th>
                  <th style={{ padding: '8px 14px' }}>回測年化</th>
                  {enough && <th style={{ padding: '8px 14px' }}>差</th>}
                  <th style={{ padding: '8px 14px' }}>走勢</th>
                </tr>
              </thead>
              <tbody>
                {ns.sort((a, b) => models[b].cum_return - models[a].cum_return)
                   .map((n) => {
                  const m = models[n];
                  const bt = m.backtest_ann;
                  const diff = bt != null ? (m.ann_return - bt) * 100 : null;
                  // 差距要同時跨過**雜訊底線**與**它自己的標準誤**才上色。
                  // 只看雜訊底線不夠：小樣本時標準誤動輒上百 pp，任何差距
                  // 都會超過 6pp 而被畫成紅綠，讀起來像「模型好/壞」。
                  //
                  // ⚠️ `ann_stderr_pp == null` ＝**算不出來**（n<2 或無波動），
                  //    必須當成「不確定性無限大」→ 一律不上色。
                  //    寫成 `?? 0` 會讓它變成「零誤差」，效果完全相反
                  //    （上線第一天實測踩到：19 個 arm 全被標成大幅落後）。
                  const se = m.ann_stderr_pp;
                  const solid = diff != null && se != null
                                && Math.abs(diff) >= noise && Math.abs(diff) >= se;
                  const dCol = !solid ? '#888' : (diff > 0 ? '#4caf50' : '#ff5252');
                  return (
                    <tr key={n} style={{ borderTop: '1px solid #222' }}>
                      <td style={{ padding: '7px 14px' }}>
                        <span style={{ color: TIER_DOT[m.tier] || '#888', fontSize: 10 }}>●</span>
                        {' '}{m.head} / {m.freq} 日
                        <span style={{ opacity: .35, fontSize: 11, marginLeft: 6,
                                       fontFamily: 'monospace' }}>{n}</span>
                      </td>
                      <td style={{ padding: '7px 14px', opacity: .6 }}>{m.n_days}</td>
                      <td style={{ padding: '7px 14px', fontWeight: 600,
                                   color: m.cum_return >= 0 ? '#4caf50' : '#ff5252' }}>
                        {m.cum_return >= 0 ? '+' : ''}{pct(m.cum_return)}
                      </td>
                      {enough && <>
                        <td style={{ padding: '7px 14px', whiteSpace: 'nowrap' }}>
                          {pct(m.ann_return)}
                          {/* 誤差棒不是裝飾——樣本小的時候它比模型之間的差距
                              大好幾倍，沒有它那個年化會被讀成精確值 */}
                          <span style={{ opacity: .5, fontSize: 11 }}>
                            {' '}±{m.ann_stderr_pp != null
                              ? `${m.ann_stderr_pp.toFixed(0)}pp` : '?'}</span>
                        </td>
                        <td style={{ padding: '7px 14px' }}>{m.ann_sharpe?.toFixed(2)}</td>
                        <td style={{ padding: '7px 14px', opacity: .7 }}>
                          {pct(m.max_drawdown)}</td>
                      </>}
                      <td style={{ padding: '7px 14px', opacity: .6 }}>
                        {m.n_rebalances} 次
                        {m.avg_turnover != null &&
                          <span style={{ opacity: .6 }}>｜換手 {pct(m.avg_turnover)}</span>}
                      </td>
                      <td style={{ padding: '7px 14px', opacity: .6 }}>
                        {bt != null ? pct(bt) : '—'}</td>
                      {enough && (
                        <td style={{ padding: '7px 14px', color: dCol,
                                     whiteSpace: 'nowrap' }}>
                          {diff != null
                            ? `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}pp`
                            : '—'}
                          {diff != null && !solid &&
                            <span style={{ opacity: .5, fontSize: 11 }}> *</span>}
                        </td>
                      )}
                      <td style={{ padding: '7px 14px' }}>
                        <Spark rets={m.daily_returns || []} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      {/* ── 誠實的限制 ── */}
      <div style={{ padding: 16, borderRadius: 10, background: 'rgba(255,255,255,.02)',
                    border: '1px solid #2a2a2a', fontSize: 12, opacity: .72,
                    lineHeight: 1.8 }}>
        <strong>判讀這張表之前</strong>
        <ul style={{ margin: '6px 0 0 18px', padding: 0 }}>
          <li>「回測年化」是 582 天單一窗、單一 seed，<strong>不是</strong>前瞻紀錄；
              兩欄並排是為了看差距，不是比大小。</li>
          <li>「差」要同時跨過 ±{noise}pp（組合層雜訊底線）<strong>與它自己的
              ±標準誤</strong>才上色；沒跨過的標 <code>*</code> 並保持灰色。
              只看前者會讓小樣本的隨機波動全部被畫成紅綠。</li>
          <li>年化的 ±標準誤 = 252 × s<sub>daily</sub> / √n。
              它通常比模型之間的差距大好幾倍——<strong>第一年之內年化排不出
              名次</strong>，這不是工具的問題，是樣本量的問題。</li>
          <li>前瞻紀錄<strong>算不出 decile spread</strong>（只有持股、沒有全市場
              逐日排序），而 Top50 年化的 run-to-run σ 是 decile 的 40 倍
              ——這張表天生比較吵。</li>
          <li><span style={{ color: TIER_DOT.incomparable }}>●</span> 灰點＝
              <strong>不可與主線並列</strong>（出自不同訓練輪），不是「比較差」。</li>
          <li>資料缺漏的日子已記在每日紀錄裡；缺漏當天的分數可能不可靠。</li>
        </ul>
      </div>
    </div>
  );
}

/** 極簡累積報酬走勢（自繪 SVG，不引外部圖表庫）。 */
function Spark({ rets }) {
  if (!rets || rets.length < 2) return <span style={{ opacity: .3 }}>—</span>;
  let v = 1;
  const eq = rets.map((r) => (v *= 1 + r));
  const lo = Math.min(...eq), hi = Math.max(...eq);
  const W = 80, H = 20, span = hi - lo || 1;
  const pts = eq.map((y, i) =>
    `${(i / (eq.length - 1)) * W},${H - ((y - lo) / span) * H}`).join(' ');
  const up = eq[eq.length - 1] >= 1;
  return (
    <svg width={W} height={H} style={{ display: 'block' }}>
      <polyline points={pts} fill="none" strokeWidth="1.5"
                stroke={up ? '#4caf50' : '#ff5252'} />
    </svg>
  );
}
