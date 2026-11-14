import React, { useEffect, useState } from 'react';
import { fetchV62Portfolio, fetchV62Arms } from '../api/v62';
import V62Performance from './V62Performance';

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
  const [famDesc, setFamDesc] = useState({});
  const [famOrder, setFamOrder] = useState([]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState('holdings');

  useEffect(() => {
    fetchV62Arms().then((d) => {
      setArms(d.arms || []);
      setFamDesc(d.family_desc || {});
      setFamOrder(d.family_order || Object.keys(d.family_desc || {}));
      if (!arm) setArm(d.default);
    }).catch(() => setArms([]));
  }, []);

  useEffect(() => {
    setLoading(true);
    fetchV62Portfolio(arm).then(setData).catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [arm]);

  // 分頁列在 loading / error 之前就算好——否則持股載入失敗時，
  // 連「前瞻績效」都點不進去（兩者是獨立的 endpoint，不該互相拖累）。
  const tabBar = (
    <div style={{ display: 'flex', gap: 4, borderBottom: '1px solid #2a2a2a' }}>
      {[['holdings', '持股組合'], ['performance', '前瞻績效比較']].map(([k, lbl]) => (
        <button key={k} onClick={() => setTab(k)} style={{
          padding: '8px 16px', border: 'none', cursor: 'pointer', fontSize: 14,
          background: 'transparent', color: tab === k ? '#4a9eff' : '#888',
          borderBottom: tab === k ? '2px solid #4a9eff' : '2px solid transparent',
          fontWeight: tab === k ? 600 : 400,
        }}>{lbl}</button>
      ))}
    </div>
  );

  if (tab === 'performance') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {tabBar}
        <V62Performance />
      </div>
    );
  }
  if (loading) {
    return <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {tabBar}<div style={{ padding: 24, opacity: .6 }}>載入中…</div></div>;
  }
  if (!data) {
    return <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {tabBar}<div style={{ padding: 24, opacity: .6 }}>無法取得 V6.2 資料</div></div>;
  }

  const spec = data.spec || {};
  const freq = spec.freq ?? 20;
  const notReady = !data.date;
  const missing = Object.entries(data.data_complete || {})
    .filter(([, v]) => v === false).map(([k]) => k);

  // tier 決定這個組合要不要下警告。分級不是憑感覺——是拿回測差距對
  // 組合層雜訊底線（N=50 約 ±6pp）量出來的，見 V6/v62_portfolio.py:PORTFOLIOS。
  const TIER_STYLE = {
    primary:      { dot: '#4caf50', bg: null, border: null },
    equivalent:   { dot: '#ffc107', bg: 'rgba(255,193,7,.07)',  border: 'rgba(255,193,7,.3)' },
    inferior:     { dot: '#ff5252', bg: 'rgba(255,82,82,.08)',  border: 'rgba(255,82,82,.35)' },
    // incomparable：不同訓練輪，**不是「比較差」**——用中性色，不可跟 inferior 同色，
    // 否則會把「不能比」誤導成「比較爛」。
    incomparable: { dot: '#9e9e9e', bg: 'rgba(158,158,158,.08)', border: 'rgba(158,158,158,.35)' },
  };
  const ts = TIER_STYLE[data.tier] || TIER_STYLE.equivalent;

  // 「每日再平衡會差多少」——從 manifest 取同一組（同 family）的最慢與最快頻率，
  // **不寫死數字**。找不到成對的就退回定性描述（寧可少講，不要講過時的數字）。
  const sameFam = arms.filter((a) => a.family === data.family
                                     && a.backtest_ann != null);
  const slowest = sameFam.reduce((m, a) => (!m || a.freq > m.freq ? a : m), null);
  const fastest = sameFam.reduce((m, a) => (!m || a.freq < m.freq ? a : m), null);
  const freqCost = (slowest && fastest && slowest.freq > fastest.freq)
    ? { slow: slowest.backtest_ann, fast: fastest.backtest_ann } : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {tabBar}

      {/* ── 組合切換：按 family 分組 ──────────────────────────────────
          ⚠️ 19 個平鋪在同一排看不出結構，而且會誤導：`v2_kg_nomacro_f03`
             （主線換 3 日再平衡）與 `old_kg_f20`（KG 壞掉的對照組）**tier 都是
             inferior**，平鋪的話長得一模一樣。family 說「這是什麼」、
             tier 說「能不能照做」，兩個維度都要看得到。            */}
      {arms.length > 1 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ fontSize: 12, opacity: .5 }}>
            並行組合（{arms.length} 個）—
            <span style={{ color: '#4caf50' }}> ● 主線</span>
            <span style={{ color: '#ffc107' }}> ● 雜訊內</span>
            <span style={{ color: '#ff5252' }}> ● 已知較差</span>
            {/* 灰點原本不在圖例裡——沒有圖例的顏色等於沒有意義 */}
            <span style={{ color: '#9e9e9e' }}> ● 不可比較</span>
          </div>
          {(famOrder.length ? famOrder : ['_all']).concat('_other')
            .map((fam) => {
              const members = fam === '_all' ? arms
                : fam === '_other'
                  // family 對不上任何一組的不可以靜默消失——那正是本專案
                  // 反覆踩到的失敗型態（整組不見、完全不報錯）。
                  ? arms.filter((a) => !famOrder.includes(a.family))
                  : arms.filter((a) => a.family === fam);
              if (!members.length) return null;
              return (
                <div key={fam}>
                  <div style={{ fontSize: 11, opacity: .42, marginBottom: 5 }}>
                    {famDesc[fam] || (fam === '_other' ? '其他' : fam)}
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {members.map((a) => {
                      const at = TIER_STYLE[a.tier] || TIER_STYLE.equivalent;
                      return (
                        <button key={a.arm} onClick={() => setArm(a.arm)}
                          title={a.note || ''}
                          style={{
                            padding: '6px 12px', borderRadius: 8, cursor: 'pointer',
                            border: a.arm === arm ? '1px solid #4a9eff' : '1px solid #333',
                            background: a.arm === arm ? 'rgba(74,158,255,.15)' : 'transparent',
                            color: a.arm === arm ? '#4a9eff' : '#999', fontSize: 13,
                            display: 'flex', alignItems: 'center', gap: 6,
                          }}>
                          <span style={{ color: at.dot, fontSize: 10 }}>●</span>
                          {a.tier === 'primary' ? '★ ' : ''}{a.label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
        </div>
      )}

      {/* ── 非主線組合的警告（使用者明確要求：要講清楚這不是最好的）── */}
      {data.tier && data.tier !== 'primary' && (
        <div style={{ padding: 14, borderRadius: 10, background: ts.bg,
                      border: `1px solid ${ts.border}`, fontSize: 13, lineHeight: 1.7 }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>
            {data.tier === 'inferior'
              ? '⛔ 這不是上線規格，而且已知比主線差'
              : data.tier === 'incomparable'
                ? '⚖️ 這不是上線規格，而且與主線不可直接比較'
                : '🔬 這不是上線規格，僅供研究參考'}
          </div>
          <div style={{ opacity: .85 }}>
            {data.tier_desc}
            {/* ⚠️ 主線回測值不可寫死——2026-08-09 回補資料重跑後從 38.0% 變成 37.3%，
                寫死的話這裡會一直顯示過時的比較基準。改由 API 帶 primary_backtest_ann。 */}
            {data.backtest_ann != null && (
              <>（回測年化 {(data.backtest_ann * 100).toFixed(1)}%
                {data.primary_backtest_ann != null &&
                  <>，主線是 {(data.primary_backtest_ann * 100).toFixed(1)}%</>}）</>
            )}
            <br />
            實際操作請看<strong>主線（★ 5d 頭 / 20 日）</strong>。
            這些並行組合的用途是<strong>累積實戰紀錄做比較</strong>，
            以及在主線不換股的那幾天提供參考視角。
          </div>
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
                  下方持股即為本次調整後的名單。距下次再平衡 {freq} 個交易日。
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
                {/* ⚠️ 這段原本寫死「從 +38.0% 掉到 −19.9%」——兩個數字都過時了
                    （2026-08-09 重跑後是 37.3% 與 25.0%），而且 −19.9% 連正負號
                    都是錯的。比較基準一律從 API 取值，不寫死在文案裡。 */}
                <div style={{ fontSize: 12, opacity: .6, marginTop: 8 }}>
                  為什麼不顯示「今日選股」：這個組合每 {freq} 個交易日換一次股。
                  {freqCost != null ? (
                    <> 同一份訊號改成<strong>每日</strong>再平衡，回測年化
                      從 {(freqCost.slow * 100).toFixed(1)}% 掉到{' '}
                      {(freqCost.fast * 100).toFixed(1)}%——差別全在換手成本。</>
                  ) : (
                    <> 同一份訊號改成每日再平衡，回測年化會明顯下降——差別全在換手成本。</>
                  )}
                  {' '}中間的分數變動不是交易訊號。
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
        <div style={{ fontWeight: 600, marginBottom: 8 }}>
          規格與回測（{data.label}）
          <span style={{ color: ts.dot, fontSize: 11, marginLeft: 8 }}>● {data.tier}</span>
        </div>
        <div style={{ opacity: .8, lineHeight: 1.7 }}>
          N={spec.n} 檔等權｜緩衝 k={spec.k}（跌出前 {Math.round(spec.n * spec.k)} 名才換掉）｜
          每 {freq} 個交易日再平衡<br />
          回測 2024-01-02 ~ 2026-06-02（582 天）：年化{' '}
          <strong>{data.backtest_ann != null
            ? `${(data.backtest_ann * 100).toFixed(1)}%` : '—'}</strong>
          {data.note_arm && <span style={{ opacity: .7 }}>｜{data.note_arm}</span>}
        </div>
        <div style={{ marginTop: 10, fontSize: 12, opacity: .6 }}>
          <strong>誠實的限制</strong>：
          <ul style={{ margin: '4px 0 0 18px', padding: 0, lineHeight: 1.7 }}>
            {(data.caveats || []).map((c) => <li key={c}>{c}</li>)}
          </ul>
        </div>
      </div>
    </div>
  );
}
