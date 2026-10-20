import { useEffect, useRef, useState } from 'react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getPortfolioHistory } from '../api/portfolioHistory';
import { createHistoryLoader } from '../api/portfolioHistoryContract.mjs';
import './V7PortfolioHistory.css';

function Exact({ children }) { return <span className="mono ph-exact">{children ?? '未提供'}</span>; }

export default function V7PortfolioHistory() {
  const [result, setResult] = useState({ state: 'loading' });
  const [selected, setSelected] = useState(null);
  const loader = useRef(null);
  useEffect(() => {
    const current = createHistoryLoader(getPortfolioHistory, next => { setResult(next); setSelected(null); });
    loader.current = current;
    current.load();
    return () => current.dispose();
  }, []);
  const history = result.state === 'ready' ? result.history : null;
  const event = history ? (history.events.find(item => item.seq === selected) ?? history.events.at(-1)) : null;
  const state = event?.state;
  const points = history?.events.map(item => ({ seq: item.seq, nav: Number(item.state.net_value), exact: item.state.net_value })) ?? [];
  const chartable = points.every(point => Number.isFinite(point.nav));
  return <main className="portfolio-history">
    <header className="page-header">
      <div><h1 className="page-title">V7 模擬組合歷史</h1><p className="page-subtitle">離線 Portfolio History · 唯讀查閱</p></div>
      <button type="button" className="btn btn-ghost" onClick={() => loader.current?.load()} disabled={result.state === 'loading'}>重新整理</button>
    </header>
    <aside className="ph-warning" role="note">
      <strong>Fixture 模擬資料 · OFFLINE_SYNTHETIC_ENGINEERING</strong>
      <p>僅供工程驗證；非真實交易、非正式投資績效、非已批准的 V7 模型輸出。真實研究資料未接入，Research admission：false；Strict Phase 0：STOP。</p>
    </aside>
    <div role="status" aria-live="polite" data-testid="history-status">{result.state === 'loading' ? '正在讀取模擬歷史…' : result.state === 'ready' ? `已讀取 ${history.source.event_count} 個模擬事件` : result.message}</div>
    {history && <>
      <section className="panel" aria-labelledby="ph-nav"><div className="panel-header"><h2 className="panel-title" id="ph-nav">NAV 歷史</h2></div><div className="panel-body">
        <p className="text-muted">圖表為近似視覺化；下方事件表保留引擎的完整十進位數值。NAV 為引擎淨值，並非報酬率。</p>
        {chartable ? <div className="ph-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={points} accessibilityLayer><XAxis dataKey="seq" /><YAxis domain={['auto', 'auto']} /><Tooltip formatter={(_, __, entry) => [entry.payload.exact, 'NAV']} labelFormatter={seq => `事件 ${seq}`} /><Line dataKey="nav" name="NAV" stroke="var(--accent, #80b8ff)" strokeWidth={2} dot isAnimationActive={false} /></LineChart></ResponsiveContainer></div> : <p>數值超出圖表顯示範圍；請查看精確事件表。</p>}
      </div></section>
      <section className="panel" aria-labelledby="ph-state"><div className="panel-header"><h2 className="panel-title" id="ph-state">事件狀態</h2><label>事件 <select aria-label="選擇歷史事件" value={event.seq} onChange={e => setSelected(Number(e.target.value))}>{history.events.map(item => <option value={item.seq} key={item.seq}>{item.seq} · {item.kind} · {item.occurred_at}</option>)}</select></label></div><div className="panel-body">
        <dl className="ph-metrics"><div><dt>NAV</dt><dd data-testid="state-nav"><Exact>{state.net_value}</Exact></dd></div><div><dt>Cash 現金</dt><dd data-testid="state-cash"><Exact>{state.cash}</Exact></dd></div><div><dt>Costs 累計成本</dt><dd data-testid="state-cost"><Exact>{state.total_cost}</Exact></dd></div><div><dt>Session count</dt><dd><Exact>{state.session_count}</Exact></dd></div></dl>
        <h3>Positions 持倉</h3><p className="text-muted">Weights 權重：未提供（未定義顯示口徑）</p>
        {Object.keys(state.positions).length ? <div className="ph-table"><table><thead><tr><th>標的</th><th>Quantity 數量</th><th>Last price 最後價格</th></tr></thead><tbody>{Object.entries(state.positions).map(([ticker, position]) => <tr key={ticker} data-position-ticker={ticker}><th scope="row">{ticker}</th><td data-field="quantity"><Exact>{position.quantity}</Exact></td><td data-field="last_price"><Exact>{position.last_price}</Exact></td></tr>)}</tbody></table></div> : <p>目前無持倉。</p>}
        <h3>Pending 待成交目標</h3><div data-testid="pending-state">{state.pending === null ? <p>無待成交目標（pending = null）。</p> : <dl className="ph-provenance"><dt>Signal ID</dt><dd><Exact>{state.pending.signal_id}</Exact></dd><dt>As of</dt><dd><Exact>{state.pending.as_of}</Exact></dd><dt>Target tickers</dt><dd>{state.pending.target_tickers.length ? state.pending.target_tickers.join('、') : '空目標'}</dd><dt>Created session</dt><dd><Exact>{state.pending.created_session}</Exact></dd></dl>}</div>
      </div></section>
      <section className="panel" aria-labelledby="ph-events"><div className="panel-header"><h2 className="panel-title" id="ph-events">Journal / Events 精確歷史</h2></div><div className="panel-body ph-table"><table><caption>每列皆為事件完成後的引擎狀態</caption><thead><tr><th>Seq</th><th>Event ID / Kind</th><th>Occurred at</th><th>NAV</th><th>Cash</th><th>Costs</th></tr></thead><tbody>{history.events.map(item => <tr key={item.seq} data-event-seq={item.seq}><th scope="row">{item.seq}</th><td><Exact>{item.event_id}</Exact><br />{item.kind}</td><td><Exact>{item.occurred_at}</Exact></td><td data-field="net_value"><Exact>{item.state.net_value}</Exact></td><td data-field="cash"><Exact>{item.state.cash}</Exact></td><td data-field="total_cost"><Exact>{item.state.total_cost}</Exact></td></tr>)}</tbody></table></div></section>
      <section className="panel" aria-labelledby="ph-source"><div className="panel-header"><h2 className="panel-title" id="ph-source">資料來源與版本</h2></div><div className="panel-body"><dl className="ph-provenance"><dt>Schema</dt><dd><Exact>{history.schema}</Exact></dd>{Object.entries(history.source).map(([key, value]) => <div className="ph-source-entry" key={key}><dt>{key}</dt><dd><Exact>{value}</Exact></dd></div>)}</dl><details><summary>引擎原始 Spec</summary><pre>{JSON.stringify(history.spec, null, 2)}</pre></details><p className="text-muted">Hash 用於完整性追溯，不構成市場事實或研究批准。UNMERGED / NOT_DEPLOYED</p></div></section>
    </>}
  </main>;
}
