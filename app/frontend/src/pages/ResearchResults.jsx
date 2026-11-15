import { useEffect, useState } from 'react';
import EvidenceTierBadge from '../components/EvidenceTierBadge';
import { getResearchSummary } from '../api/research';
import { normalizeResearch, formatMetric as fmt, formatCI } from '../api/researchContract.mjs';
import './ResearchResults.css';

function Comparison({ caption, columns, rows }) {
  return <div className="research-table" tabIndex={0} role="region" aria-label={caption}>
    <table><caption>{caption}</caption><thead><tr>{columns.map(([key, label]) =>
      <th scope="col" key={key}>{label}</th>)}</tr></thead><tbody>{rows.map(row =>
      <tr key={row.key}>{columns.map(([key], i) => i === 0
        ? <th scope="row" key={key}>{row[key]}</th> : <td key={key}>{row[key]}</td>)}</tr>)}</tbody></table>
  </div>;
}

export default function ResearchResults() {
  const [data, setData] = useState(() => normalizeResearch(null));
  const [loading, setLoading] = useState(true);
  const [horizon, setHorizon] = useState(20);
  const [year, setYear] = useState('all');
  useEffect(() => {
    let active = true;
    getResearchSummary().then(result => { if (active) { setData(result); setLoading(false); } });
    return () => { active = false; };
  }, []);
  const years = [...new Set(data.portfolios.flatMap(row => Object.keys(row.years)))].sort();
  return <div className="research-results">
    <header className="page-header"><div><h1 className="page-title">模型與投組比較</h1>
      <p className="page-subtitle">離線 OOS 研究快照 · UNMERGED / NOT_DEPLOYED</p></div></header>
    <section className="panel research-caution" aria-labelledby="research-evidence">
      <div className="panel-body"><h2 id="research-evidence"><EvidenceTierBadge tier="RESEARCH_ONLY" /></h2>
        <p>未經交易驗證，不代表可成交、即時績效或投資建議。高報酬不能取代 PIT、選樣與來源驗證；SSM 未證明優於基準。</p>
        <ul>{data.caveats.map(caveat => <li key={caveat}>{caveat}</li>)}</ul>
      </div></section>
    <p role="status">{loading ? '正在讀取研究摘要…' : data.state !== 'ready' ? '研究摘要未就緒或驗證失敗；沒有沿用舊數字。' : '摘要可讀；證據等級仍為 RESEARCH_ONLY。'}</p>
    {data.state === 'ready' && <>
      <section className="panel"><div className="panel-body">
        <h2>OOS 截面排名</h2><p>Rank IC 為相關係數，ICIR 不年化；95% CI 與 t 採 Newey–West。seed 列為單次模型；GRU／selective 無 seed 列為三 seed 集成。</p>
        <label>預測 horizon <select value={horizon} onChange={e => setHorizon(Number(e.target.value))}>
          <option value={5}>5 sessions</option><option value={20}>20 sessions</option></select></label>
        <Comparison caption={`${horizon} sessions · 依模型名稱排列，不宣告 winner`} columns={[
          ['model','模型／策略'], ['ic','Rank IC'], ['icir','ICIR'], ['t','NW t'], ['ci','95% CI'], ['support','有效／全部 sessions'],
        ]} rows={data.models.filter(row => row.horizon === horizon).map(row => ({
          key: row.model, model: row.model, ic: fmt(row.ic_mean), icir: fmt(row.icir),
          t: fmt(row.nw_t), ci: formatCI(row.ic_ci95), support: `${row.days} / ${row.sessions}`,
        }))} />
      </div></section>
      <section className="panel"><div className="panel-body"><h2>投組歷史模擬</h2>
        <p>20-session horizon · Top 20 · 每 20 sessions 再平衡 · 等權。基準為相同偏差 universe 等權。Turnover 是全 session 平均；成本拖累為累計成本／初始 NAV。</p>
        <Comparison caption="全期扣成本後比較（非可成交證據）" columns={[
          ['model','模型／策略'], ['window','期間'], ['return','總報酬'], ['annual','年化'], ['benchmark','基準'],
          ['excess','超額'], ['sharpe','Sharpe-like'], ['dd','最大回撤'], ['turnover','平均換手'], ['cost','成本拖累'], ['ir','IR'],
        ]} rows={data.portfolios.map(row => ({ key:row.model, model:row.model, window:`${row.start}～${row.end}`,
          return:fmt(row.total_return,true), annual:fmt(row.annualized_return,true), benchmark:fmt(row.benchmark_return,true),
          excess:fmt(row.excess_return,true), sharpe:fmt(row.sharpe_like), dd:fmt(row.max_drawdown,true),
          turnover:fmt(row.mean_turnover,true), cost:fmt(row.cost_drag,true), ir:fmt(row.information_ratio),
        }))} />
      </div></section>
      <section className="panel"><div className="panel-body"><h2>逐年報酬</h2>
        <label>年度 <select value={year} onChange={e => setYear(e.target.value)}><option value="all">全部年度</option>
          {years.map(y => <option key={y} value={y}>{y}</option>)}</select></label>
        <p>首尾年度為部分期間；未定義的數值顯示「—」。</p>
        <Comparison caption="年度扣成本後報酬與等權基準" columns={[
          ['model','模型／策略'], ['year','年度'], ['return','報酬'], ['benchmark','基準'],
        ]} rows={data.portfolios.flatMap(row => Object.entries(row.years).filter(([y]) => year === 'all' || year === y)
          .map(([y, values]) => ({key:`${row.model}/${y}`, model:row.model, year:y,
            return:fmt(values.total_return,true), benchmark:fmt(values.benchmark_return,true)})))} />
      </div></section>
      <details className="panel"><summary>摘要與資料身份</summary><dl className="panel-body research-identity">
        <dt>摘要 SHA-256</dt><dd>{data.sha256}</dd>
        {['dataset_sha256', 'manifest_sha256', 'evaluation_sha256', 'source_commit'].map(key =>
          <div key={key}><dt>{key}</dt><dd>{data.identity[key]}</dd></div>)}
        <dt>評估口徑</dt><dd>{data.evaluation_policy}</dd>
      </dl></details>
    </>}
  </div>;
}
