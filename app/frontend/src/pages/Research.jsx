import React, { useMemo, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import data from '../research/experiments.json';

/**
 * /research — 研究紀錄（作品集頁）
 *
 * 設計原則
 *  - 資料全部在 `src/research/experiments.json`，本元件只負責渲染
 *    → 之後新增實驗只要往 JSON 加一筆，不用碰 React
 *  - 收納三層：主線分組摺疊 → 實驗列（7 欄）→ 展開卡片
 *  - 「只看被推翻的假設」篩選：這是這份紀錄最有說服力的部分，
 *    一般作品集只放贏的實驗，這裡把八次自我推翻攤開來
 */

const V_STYLE = {
  採納:     { bg: 'rgba(34,197,94,.14)',  fg: '#4ade80', label: '採納' },
  拒絕:     { bg: 'rgba(239,68,68,.14)',  fg: '#f87171', label: '拒絕' },
  無效應:   { bg: 'rgba(148,163,184,.14)', fg: '#94a3b8', label: '無效應' },
  待覆核:   { bg: 'rgba(234,179,8,.14)',  fg: '#facc15', label: '待覆核' },
  正確性修正: { bg: 'rgba(56,189,248,.14)', fg: '#38bdf8', label: '正確性修正' },
  進行中:   { bg: 'rgba(168,85,247,.14)', fg: '#c084fc', label: '進行中' },
};

function Badge({ verdict }) {
  const s = V_STYLE[verdict] || V_STYLE['無效應'];
  return (
    <span style={{
      background: s.bg, color: s.fg, borderRadius: 5, padding: '2px 8px',
      fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap',
    }}>{s.label}</span>
  );
}

function Row({ exp }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderBottom: '1px solid rgba(148,163,184,.12)' }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: '100%', background: 'none', border: 'none', cursor: 'pointer',
          display: 'grid', gridTemplateColumns: '64px 1fr auto auto 20px',
          gap: 12, alignItems: 'center', padding: '11px 4px',
          textAlign: 'left', color: 'inherit', font: 'inherit',
        }}
      >
        <code style={{ color: '#64748b', fontSize: 12 }}>{exp.id}</code>
        <span style={{ fontSize: 14 }}>
          {exp.title}
          {exp.overturned && (
            <span title="這個實驗推翻了事先提出的假設"
                  style={{ marginLeft: 8, fontSize: 11, color: '#fb923c' }}>↺ 假設被推翻</span>
          )}
        </span>
        <Badge verdict={exp.verdict} />
        <span style={{ color: '#475569', fontSize: 12 }}>{exp.date}</span>
        <span style={{ color: '#475569', fontSize: 11 }}>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div style={{
          padding: '4px 4px 18px 76px', display: 'grid', gap: 10,
          fontSize: 13.5, lineHeight: 1.65, color: '#cbd5e1',
        }}>
          <Field k="問題" v={exp.question} />
          <Field k="設計" v={exp.design} />
          <Field k="跑前定死的門檻" v={exp.threshold} accent />
          <Field k="結果" v={exp.result} />
          {exp.note && <Field k="誠實備註" v={exp.note} />}
        </div>
      )}
    </div>
  );
}

function Field({ k, v, accent }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: 12 }}>
      <span style={{
        color: accent ? '#facc15' : '#64748b', fontSize: 12,
        fontWeight: accent ? 700 : 400,
      }}>{k}</span>
      <span>{v}</span>
    </div>
  );
}

function Group({ line, experiments, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen);
  if (experiments.length === 0) return null;
  const overturned = experiments.filter((e) => e.overturned).length;
  return (
    <section style={{
      background: 'rgba(15,23,42,.5)', border: '1px solid rgba(148,163,184,.14)',
      borderRadius: 12, marginBottom: 14, overflow: 'hidden',
    }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: '100%', background: 'rgba(30,41,59,.5)', border: 'none',
          cursor: 'pointer', padding: '14px 18px', textAlign: 'left',
          color: 'inherit', font: 'inherit',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 20 }}>{line.icon}</span>
          <strong style={{ fontSize: 16 }}>{line.name}</strong>
          <span style={{ color: '#64748b', fontSize: 12 }}>{line.period}</span>
          <span style={{ marginLeft: 'auto', color: '#64748b', fontSize: 12 }}>
            {experiments.length} 個實驗
            {overturned > 0 && <span style={{ color: '#fb923c' }}>｜{overturned} 次假設被推翻</span>}
            <span style={{ marginLeft: 10 }}>{open ? '▲' : '▼'}</span>
          </span>
        </div>
        <p style={{ margin: '8px 0 0', color: '#94a3b8', fontSize: 13, lineHeight: 1.6 }}>
          {line.summary}
        </p>
      </button>
      {open && <div style={{ padding: '0 18px 6px' }}>
        {experiments.map((e) => <Row key={e.id} exp={e} />)}
      </div>}
    </section>
  );
}

export default function Research() {
  const { pathname } = useLocation();
  const [onlyOverturned, setOnlyOverturned] = useState(false);
  const [q, setQ] = useState('');

  const shown = useMemo(() => {
    const kw = q.trim().toLowerCase();
    return data.experiments.filter((e) => {
      if (onlyOverturned && !e.overturned) return false;
      if (!kw) return true;
      return [e.id, e.title, e.question, e.result, e.note]
        .filter(Boolean).join(' ').toLowerCase().includes(kw);
    });
  }, [onlyOverturned, q]);

  const total = data.experiments.length;
  const nOverturned = data.experiments.filter((e) => e.overturned).length;
  const nFix = data.experiments.filter((e) => e.verdict === '正確性修正').length;

  // /research/pipeline 等子頁 → 只渲染子頁
  if (pathname !== '/research') {
    return (
      <div style={{ padding: '18px 22px' }}>
        <SubNav />
        <Outlet />
      </div>
    );
  }

  return (
    <div style={{ padding: '18px 22px', maxWidth: 1180, margin: '0 auto' }}>
      <SubNav />

      <header style={{ marginBottom: 18 }}>
        <h1 style={{ margin: '0 0 6px', fontSize: 26 }}>研究紀錄</h1>
        <p style={{ margin: 0, color: '#94a3b8', fontSize: 14, lineHeight: 1.7 }}>
          從資料層到風控層，每一個實驗的問題、設計、
          <strong style={{ color: '#facc15' }}>跑之前就定死的判定門檻</strong>、結果與誠實備註。
          <br />
          方法紀律：一次一變因、判定規則 pre-register、
          結論與被推翻的假設一律留存。最後更新 {data.meta.updated}。
        </p>
      </header>

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
        <Stat n={total} label="實驗總數" />
        <Stat n={nOverturned} label="假設被自己推翻" color="#fb923c" />
        <Stat n={nFix} label="正確性修正" color="#38bdf8" />
        <Stat n={data.lines.length} label="主線" />
      </div>

      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 16, flexWrap: 'wrap' }}>
        <input
          value={q} onChange={(e) => setQ(e.target.value)}
          placeholder="搜尋實驗（代號 / 標題 / 結果 / 備註）"
          style={{
            flex: '1 1 260px', minWidth: 220, padding: '9px 12px',
            background: 'rgba(15,23,42,.7)', color: '#e2e8f0',
            border: '1px solid rgba(148,163,184,.22)', borderRadius: 8, fontSize: 13,
          }}
        />
        <button
          onClick={() => setOnlyOverturned((v) => !v)}
          style={{
            padding: '9px 14px', borderRadius: 8, cursor: 'pointer', fontSize: 13,
            border: `1px solid ${onlyOverturned ? '#fb923c' : 'rgba(148,163,184,.22)'}`,
            background: onlyOverturned ? 'rgba(251,146,60,.16)' : 'rgba(15,23,42,.7)',
            color: onlyOverturned ? '#fb923c' : '#94a3b8',
          }}
        >
          ↺ 只看被推翻的假設（{nOverturned}）
        </button>
      </div>

      {onlyOverturned && (
        <p style={{
          margin: '0 0 14px', padding: '10px 14px', fontSize: 13, lineHeight: 1.7,
          background: 'rgba(251,146,60,.08)', border: '1px solid rgba(251,146,60,.25)',
          borderRadius: 8, color: '#fdba74',
        }}>
          這裡列的是<strong>事先提出、後來被自己的實驗推翻</strong>的假設。
          之所以特別標出來，是因為它們證明的是方法而不是運氣——
          判定門檻在跑之前就定死，所以結果不如預期時無法事後改規則。
        </p>
      )}

      {data.lines.map((line, i) => (
        <Group
          key={line.id} line={line}
          experiments={shown.filter((e) => e.line === line.id)}
          defaultOpen={i >= data.lines.length - 2 || onlyOverturned || !!q.trim()}
        />
      ))}

      {shown.length === 0 && (
        <p style={{ color: '#64748b', textAlign: 'center', padding: 40 }}>沒有符合的實驗</p>
      )}

      <footer style={{
        marginTop: 22, paddingTop: 16, borderTop: '1px solid rgba(148,163,184,.12)',
        color: '#64748b', fontSize: 12.5, lineHeight: 1.8,
      }}>
        完整數字與推導過程在 repo 的 <code>docs/</code>：
        <code>feature-protocol-v2.md</code>（特徵層協定與 F5 全部級數）、
        <code>f6-training-log-and-readout.md</code>（F6 訓練紀錄與判讀清單）、
        <code>portfolio-construction-baseline-v1.md</code>（組合建構凍結規格）、
        <code>portfolio-lab-results-2026-08-01.md</code>（組合層與風控層結果）。
      </footer>
    </div>
  );
}

function Stat({ n, label, color }) {
  return (
    <div style={{
      background: 'rgba(15,23,42,.6)', border: '1px solid rgba(148,163,184,.14)',
      borderRadius: 10, padding: '10px 16px', minWidth: 110,
    }}>
      <div style={{ fontSize: 24, fontWeight: 700, color: color || '#e2e8f0' }}>{n}</div>
      <div style={{ fontSize: 12, color: '#64748b' }}>{label}</div>
    </div>
  );
}

function SubNav() {
  const tabs = [
    { to: '/research', label: '實驗總覽', end: true },
    { to: '/research/pipeline', label: '資料與模型管線' },
  ];
  return (
    <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
      {tabs.map((t) => (
        <NavLink
          key={t.to} to={t.to} end={t.end}
          style={({ isActive }) => ({
            padding: '7px 14px', borderRadius: 8, fontSize: 13, textDecoration: 'none',
            background: isActive ? 'rgba(56,189,248,.14)' : 'rgba(15,23,42,.6)',
            color: isActive ? '#38bdf8' : '#94a3b8',
            border: `1px solid ${isActive ? 'rgba(56,189,248,.35)' : 'rgba(148,163,184,.14)'}`,
          })}
        >{t.label}</NavLink>
      ))}
    </div>
  );
}
