import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getPortfolioHistory } from '../api/portfolioHistory';
import { createHistoryLoader } from '../api/portfolioHistoryContract.mjs';
import { describeEvent, eventName, eventTime, formatDecimal, selectedEvent } from '../components/portfolioHistoryPresentation.mjs';
import './V7PortfolioHistory.css';

function Value({ value, advanced, digits = 4, testId }) {
  return <span className="ph-number" data-testid={testId}>{advanced ? value ?? '資料不可用' : formatDecimal(value, digits)}</span>;
}
function EventDot({ cx, cy, payload, select, selected }) {
  if (!payload || !Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  return <circle cx={cx} cy={cy} r={selected === payload.seq ? 7 : 5} className="ph-chart-dot" data-chart-seq={payload.seq}
    role="button" tabIndex={0} aria-label={`選取事件 ${payload.seq}：${eventName(payload.kind)}，${eventTime(payload.occurred_at)}`}
    aria-pressed={selected === payload.seq} onClick={e => { e.stopPropagation(); select(payload.seq); }}
    onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(payload.seq); } }} />;
}

function EventTimeline({ events, selected, onSelect }) {
  const list = useRef(null);
  const current = useRef(null);
  useLayoutEffect(() => {
    const container = list.current;
    const item = current.current;
    if (!container || !item) return;
    const keepVisible = () => {
      const bounds = container.getBoundingClientRect();
      const target = item.getBoundingClientRect();
      const top = bounds.top + container.clientTop;
      const bottom = top + container.clientHeight;
      // Scroll only this list; preserve page position and the user's focus.
      if (target.top < top) container.scrollTop += target.top - top;
      else if (target.bottom > bottom) container.scrollTop += target.bottom - bottom;
    };
    keepVisible();
    const observer = new ResizeObserver(keepVisible);
    observer.observe(container);
    // Fonts can resize earlier rows while the scroll container stays fixed-height.
    for (const row of container.children) observer.observe(row);
    return () => observer.disconnect();
  }, [events, selected]);
  return <section className="panel ph-timeline-panel" aria-labelledby="ph-timeline">
    <div className="panel-header"><h2 className="panel-title" id="ph-timeline">事件時間線</h2></div>
    <ol className="ph-timeline" ref={list}>{events.map(item => <li key={item.seq}>
      <button type="button" ref={selected === item.seq ? current : null} data-timeline-seq={item.seq}
        aria-pressed={selected === item.seq} aria-current={selected === item.seq ? 'step' : undefined} onClick={() => onSelect(item.seq)}>
        <span className="ph-sequence">{item.seq}</span><span><strong>{eventName(item.kind)}</strong>
          {selected === item.seq ? <span className="ph-selected-label">目前選取</span> : null}
          <time dateTime={item.occurred_at}>{eventTime(item.occurred_at)}</time></span>
      </button>
    </li>)}</ol>
  </section>;
}

export default function V7PortfolioHistory() {
  const [result, setResult] = useState({ state: 'loading' });
  const [selected, setSelected] = useState(null);
  const [advanced, setAdvanced] = useState(false);
  const loader = useRef(null);
  useEffect(() => {
    const current = createHistoryLoader(getPortfolioHistory, next => { setResult(next); setSelected(null); });
    loader.current = current;
    current.load();
    return () => current.dispose();
  }, []);
  const history = result.state === 'ready' ? result.history : null;
  const event = history ? selectedEvent(history, selected) : null;
  const state = event?.state;
  const points = history?.events.map(item => ({ seq: item.seq, nav: Number(item.state.net_value), exact: item.state.net_value, kind: item.kind, occurred_at: item.occurred_at })) ?? [];
  const chartable = points.every(point => Number.isFinite(point.nav));
  return <main className="portfolio-history" data-mode={advanced ? 'advanced' : 'general'}>
    <header className="page-header">
      <div><h1 className="page-title">模擬組合歷史</h1><p className="page-subtitle">選擇事件，查看當時的組合狀態</p></div>
      <button type="button" className="btn btn-ghost" onClick={() => loader.current?.load()} disabled={result.state === 'loading'}>重新整理</button>
    </header>
    <aside className="ph-warning" role="note">
      <strong>Synthetic Fixture · 模擬資料</strong><span>僅供工程驗證，非真實交易或正式績效</span>
      <small>research_admission=false · Strict Phase 0：STOP</small>
    </aside>
    <div className="ph-view-bar">
      <fieldset className="ph-mode"><legend className="ph-sr-only">顯示模式</legend>
        <button type="button" aria-pressed={!advanced} onClick={() => setAdvanced(false)}>一般模式</button>
        <button type="button" aria-pressed={advanced} onClick={() => setAdvanced(true)}>進階研究模式</button>
      </fieldset>
      <span role="status" aria-live="polite" data-testid="history-status">{result.state === 'loading' ? '正在讀取模擬歷史…' : result.state === 'ready' ? `${history.source.event_count} 個模擬事件 · 唯讀` : result.message}</span>
    </div>
    {history ? <>
      <section className="ph-overview" aria-labelledby="ph-current">
        <div className="ph-current-heading"><div><p className="ph-context">{event.seq === history.events.length ? '最新事件' : '歷史事件'} {event.seq} / {history.events.length}</p><h2 id="ph-current" data-testid="selected-event-title">{eventName(event.kind)}</h2><time dateTime={event.occurred_at}>{eventTime(event.occurred_at)}</time></div>
          <label className="ph-event-select">查看事件<select aria-label="選擇歷史事件" value={event.seq} onChange={e => setSelected(Number(e.target.value))}>{history.events.map(item => <option value={item.seq} key={item.seq}>{item.seq} · {eventName(item.kind)} · {eventTime(item.occurred_at)}</option>)}</select></label>
        </div>
        <p className="ph-explanation" data-testid="event-explanation" aria-live="polite">{describeEvent(event, history.events[event.seq - 2])}</p>
        <dl className="ph-metrics">
          <div><dt>NAV 淨值</dt><dd><Value value={state.net_value} advanced={advanced} testId="state-nav" /></dd><small>引擎淨值，非報酬率</small></div>
          <div><dt>Cash 現金</dt><dd><Value value={state.cash} advanced={advanced} testId="state-cash" /></dd><small>沿用引擎原始單位</small></div>
          <div><dt>Costs 累計成本</dt><dd><Value value={state.total_cost} advanced={advanced} testId="state-cost" /></dd><small>截至所選事件</small></div>
          <div><dt>Pending 待成交</dt><dd className="ph-pending-summary" data-testid="pending-summary">{state.pending === null ? '無待成交' : state.pending === undefined ? '資料不可用' : '有待成交目標'}</dd><small>{Object.keys(state.positions).length} 項持倉 · {state.session_count} 個市場時段</small></div>
        </dl>
        <p className="ph-format-note">{advanced ? '完整 Decimal 原始值；不經顯示四捨五入。' : '顯示精度：數值 4 位、持倉數量 6 位小數；僅四捨五入顯示。完整值見進階研究模式，狀態比較使用完整精度。'}</p>
      </section>
      <div className="ph-history-grid">
        <section className="panel ph-nav-panel" aria-labelledby="ph-nav"><div className="panel-header"><h2 className="panel-title" id="ph-nav">NAV 事件歷程</h2></div><div className="panel-body">
          <p className="text-muted">點選圖上的事件或右側時間線，查看該事件完成後的狀態。圖表為近似視覺化。</p>
          {chartable ? <div className="ph-chart"><ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 600, height: 260 }}><LineChart data={points} margin={{ top: 20, right: 20, bottom: 10, left: 0 }} accessibilityLayer onClick={chart => { if (chart?.activeTooltipIndex != null) { const point = points[Number(chart.activeTooltipIndex)]; if (point) setSelected(point.seq); } }}>
            <XAxis dataKey="seq" tickFormatter={seq => `${seq} · ${points[seq - 1]?.occurred_at.slice(5, 10)}`} minTickGap={22} />
            <YAxis domain={['auto', 'auto']} tickFormatter={value => value.toFixed(4)} width={70} />
            <Tooltip formatter={(_, __, entry) => [advanced ? entry.payload.exact : formatDecimal(entry.payload.exact), 'NAV']} labelFormatter={seq => { const point = points[Number(seq) - 1]; return point ? `${eventName(point.kind)} · ${eventTime(point.occurred_at)}` : ''; }} />
            <Line dataKey="nav" name="NAV" stroke="var(--accent-blue)" strokeWidth={2} dot={<EventDot select={setSelected} selected={event.seq} />} activeDot={false} isAnimationActive={false} />
          </LineChart></ResponsiveContainer></div> : <p>數值超出圖表顯示範圍；請使用時間線與進階原始值。</p>}
        </div></section>
        <EventTimeline events={history.events} selected={event.seq} onSelect={setSelected} />
      </div>
      <div className="ph-detail-grid">
        <section className="panel" aria-labelledby="ph-positions"><div className="panel-header"><h2 className="panel-title" id="ph-positions">持倉</h2></div><div className="panel-body">
          {Object.keys(state.positions).length ? <div className="ph-table" tabIndex={0} role="region" aria-label="持倉數量與價格，可橫向捲動"><table><thead><tr><th>標的</th><th>數量</th><th>最後價格</th></tr></thead><tbody>{Object.entries(state.positions).map(([ticker, position]) => <tr key={ticker} data-position-ticker={ticker}><th scope="row">{ticker}</th><td data-field="quantity"><Value value={position.quantity} advanced={advanced} digits={6} /></td><td data-field="last_price"><Value value={position.last_price} advanced={advanced} /></td></tr>)}</tbody></table></div> : <p>此事件完成後沒有持倉。</p>}
          <p className="ph-format-note">權重口徑未定義，未新增權重計算。</p>
        </div></section>
        <section className="panel" aria-labelledby="ph-pending"><div className="panel-header"><h2 className="panel-title" id="ph-pending">待成交目標</h2></div><div className="panel-body" data-testid="pending-state">{state.pending === null ? <p>沒有待成交目標（pending = null）。</p> : state.pending === undefined ? <p>待成交資料不可用。</p> : <dl className="ph-provenance"><dt>訊號 ID</dt><dd>{state.pending.signal_id}</dd><dt>記錄時間</dt><dd>{eventTime(state.pending.as_of)}</dd><dt>目標標的</dt><dd>{state.pending.target_tickers.length ? state.pending.target_tickers.join('、') : '空目標'}</dd><dt>建立時段</dt><dd>{state.pending.created_session}</dd></dl>}</div></section>
      </div>
      {advanced ? <>
        <section className="panel" aria-labelledby="ph-events"><div className="panel-header"><h2 className="panel-title" id="ph-events">完整精度事件表</h2></div><div className="panel-body ph-table" tabIndex={0} role="region" aria-label="完整事件表，可橫向捲動"><table><thead><tr><th>Seq</th><th>Event ID / Kind</th><th>Occurred at</th><th>NAV</th><th>Cash</th><th>Costs</th></tr></thead><tbody>{history.events.map(item => <tr key={item.seq} data-event-seq={item.seq}><th scope="row">{item.seq}</th><td>{item.event_id}<br />{item.kind}</td><td>{item.occurred_at}</td><td data-field="net_value">{item.state.net_value}</td><td data-field="cash">{item.state.cash}</td><td data-field="total_cost">{item.state.total_cost}</td></tr>)}</tbody></table></div></section>
        <section className="panel" aria-labelledby="ph-raw"><div className="panel-header"><h2 className="panel-title" id="ph-raw">所選事件原始欄位</h2></div><div className="panel-body"><p>API 提供的原始事件與狀態；可選取文字複製，不含未提供的 Journal payload。</p><textarea className="ph-raw" aria-label="原始事件 JSON，可選取複製" readOnly value={JSON.stringify(event, null, 2)} data-testid="raw-event" spellCheck={false} /></div></section>
        <section className="panel" aria-labelledby="ph-source"><div className="panel-header"><h2 className="panel-title" id="ph-source">來源、版本與完整性</h2></div><div className="panel-body"><dl className="ph-provenance"><dt>Schema</dt><dd>{history.schema}</dd>{Object.entries(history.source).map(([key, value]) => <div className="ph-source-entry" key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl><details><summary>引擎原始 Spec</summary><pre>{JSON.stringify(history.spec, null, 2)}</pre></details><p className="text-muted">Hash 用於完整性追溯，不構成市場事實或研究批准。</p></div></section>
      </> : null}
      <footer className="ph-source-footer"><span>來源：{history.source.fixture_id}</span><span>OFFLINE_SYNTHETIC_ENGINEERING · UNMERGED / NOT_DEPLOYED</span>{!advanced ? <button type="button" className="btn btn-ghost" onClick={() => setAdvanced(true)}>查看原始值與來源識別</button> : null}</footer>
    </> : <div className="ph-unavailable"><h2>{result.state === 'loading' ? '準備歷史資料' : '目前沒有可顯示的組合狀態'}</h2><p>{result.state === 'loading' ? '正在讀取已驗證的模擬資料。' : '確認本機 Fixture 服務與資料後，再按重新整理。缺少資料不代表現金為零或沒有待成交。'}</p></div>}
  </main>;
}
