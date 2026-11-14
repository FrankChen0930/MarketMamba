import React, { useEffect, useState } from 'react';
import { useApi } from '../hooks/useApi';
import { fetchMarket } from '../api/market';
import { SkeletonBlock, ApiError } from '../components/SkeletonLoader';

const API_BASE = import.meta.env.VITE_API_URL || 'https://marketmamba-api.onrender.com';

// Fetch Claude report
async function fetchReport() {
  const r = await fetch(`${API_BASE}/api/reports/latest`);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

// Parse markdown-ish report text into sections
function parseReport(text) {
  if (!text) return [];
  const lines = text.split('\n');
  const sections = [];
  let cur = null;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith('## ') || trimmed.startsWith('# ')) {
      if (cur) sections.push(cur);
      cur = { title: trimmed.replace(/^#+\s*/, ''), lines: [] };
    } else if (trimmed.startsWith('---')) {
      // divider, ignore
    } else {
      if (!cur) cur = { title: '', lines: [] };
      cur.lines.push(trimmed);
    }
  }
  if (cur) sections.push(cur);
  return sections;
}

function ReportSection({ title, lines }) {
  const emojiTitle = title;
  return (
    <div style={{ marginBottom: 20 }}>
      {title && (
        <div style={{
          fontSize: 14, fontWeight: 700, color: 'var(--text-primary)',
          marginBottom: 10, paddingBottom: 6, borderBottom: '1px solid var(--border)'
        }}>
          {emojiTitle}
        </div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {lines.map((line, i) => {
          const isItem = line.startsWith('- ') || line.startsWith('• ');
          const text = isItem ? line.replace(/^[-•]\s*/, '') : line;
          const parts = text.split(/\*\*([^*]+)\*\*/g);
          return (
            <div key={i} style={{
              fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7,
              paddingLeft: isItem ? 14 : 0,
              borderLeft: isItem ? '2px solid var(--border)' : 'none',
            }}>
              {parts.map((p, j) => j % 2 === 1
                ? <strong key={j} style={{ color: 'var(--text-primary)' }}>{p}</strong>
                : p
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Top10Table({ top10 }) {
  if (!top10?.length) return null;
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title"><span>🏆</span> 今日 AI 精選 Top 10</div>
        <span className="badge badge-positive">Claude 推薦</span>
      </div>
      <div className="panel-body-flush">
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th><th>股票</th><th>20d Alpha</th>
              <th>信號強度</th><th>建議比重</th><th>信心</th>
            </tr>
          </thead>
          <tbody>
            {top10.map((s, i) => (
              <tr key={s.Ticker} style={{ animationDelay: `${i * 0.04}s` }} className="animate-fade-up">
                <td style={{ color: 'var(--accent-amber)', fontSize: 12 }}>{i + 1}</td>
                <td>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>
                    {s.Name && s.Name !== s.Ticker ? s.Name : s.Ticker}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {s.Ticker}
                  </div>
                </td>
                <td className="mono text-positive">+{((s.Exp_Alpha_20d || 0) * 100).toFixed(2)}%</td>
                <td className="mono text-positive">{(s.Signal_Quality || 0).toFixed(2)}</td>
                <td className="mono">{s.Suggested_Weight ? `${(s.Suggested_Weight * 100).toFixed(1)}%` : '—'}</td>
                <td>
                  <span style={{
                    fontSize: 11,
                    color: s.Confidence === '高信心' ? 'var(--positive)' : 'var(--accent-amber)'
                  }}>{s.Confidence || '—'}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


export default function MarketView() {
  const { data: market, loading: mktLoading } = useApi(fetchMarket);
  const { data: report, loading: repLoading, error: repError } = useApi(fetchReport);

  const sections = parseReport(report?.summary || '');
  const loading = mktLoading || repLoading;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header">
        <div>
          <div className="page-title">AI 市場日報</div>
          <div className="page-subtitle">
            Claude 量化策略分析 · {report?.date || market?.last_run || '—'} · 僅供參考
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {report?.model && (
            <span className="badge badge-blue" style={{ fontSize: 11 }}>
              {report.model}
            </span>
          )}
          <span className="badge badge-neutral" style={{ fontSize: 11 }}>⚠️ LLM 生成，不構成投資建議</span>
        </div>
      </div>

      {/* ── Macro Snapshot ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        {mktLoading ? Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="stat-card"><SkeletonBlock height={40} /></div>
        )) : [
          {
            label: 'TAIEX',
            value: market?.taiex?.value?.toLocaleString('zh-TW', { maximumFractionDigits: 0 }) || '—',
            sub: market?.taiex ? (
              <span className={market.taiex.change >= 0 ? 'text-positive' : 'text-negative'}>
                {market.taiex.change >= 0 ? '▲' : '▼'} {Math.abs(market.taiex.change_pct).toFixed(2)}%
              </span>
            ) : null,
            accent: (market?.taiex?.change ?? 0) >= 0 ? 'var(--positive)' : 'var(--negative)',
          },
          {
            label: 'VIX 恐慌指數',
            value: market?.vix?.toFixed(2) || '—',
            sub: <span style={{ color: (market?.vix || 0) > 20 ? 'var(--negative)' : 'var(--positive)' }}>
              {(market?.vix || 0) > 30 ? '高度恐慌' : (market?.vix || 0) > 20 ? '波動偏高' : '市場平穩'}
            </span>,
          },
          {
            label: 'S&P 500 今日',
            value: market?.spx_change !== undefined ? `${market.spx_change >= 0 ? '+' : ''}${market.spx_change.toFixed(2)}%` : '—',
            sub: <span className={(market?.spx_change ?? 0) >= 0 ? 'text-positive' : 'text-negative'}>
              美股參考
            </span>,
          },
          {
            label: '黃金 今日漲跌',
            value: market?.gold_change !== undefined ? `${market.gold_change >= 0 ? '+' : ''}${market.gold_change.toFixed(2)}%` : '—',
            sub: <span style={{ color: 'var(--text-muted)' }}>USD/TWD {market?.usd_twd?.toFixed(3)}</span>,
          },
        ].map(card => (
          <div key={card.label} className="stat-card" style={card.accent ? {
            borderColor: card.accent, background: `color-mix(in srgb, ${card.accent} 5%, var(--bg-panel))`
          } : {}}>
            <div className="label">{card.label}</div>
            <div className="value mono" style={{ fontSize: 18 }}>{card.value}</div>
            {card.sub && <div className="sub">{card.sub}</div>}
          </div>
        ))}
      </div>

      {/* ── Main Content ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 16, alignItems: 'start' }}>

        {/* Claude Report */}
        <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.15)', background: 'rgba(0,212,255,0.02)' }}>
          <div className="panel-header">
            <div className="panel-title">
              <span>🤖</span> Claude AI 市場分析報告
            </div>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {report?.date || '—'}
            </span>
          </div>
          <div className="panel-body">
            {repLoading ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {Array.from({ length: 4 }).map((_, i) => <SkeletonBlock key={i} height={60} />)}
              </div>
            ) : repError || !report?.available ? (
              <div style={{
                padding: '24px', textAlign: 'center',
                color: 'var(--text-muted)', fontSize: 13
              }}>
                <div style={{ fontSize: 24, marginBottom: 12 }}>⏳</div>
                <div>今日報告生成中</div>
                <div style={{ fontSize: 11, marginTop: 8 }}>推論完成後 Claude 將自動生成分析</div>
              </div>
            ) : sections.length > 0 ? (
              <div>
                {sections.map((s, i) => (
                  <ReportSection key={i} title={s.title} lines={s.lines} />
                ))}
              </div>
            ) : (
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
                {report?.summary}
              </p>
            )}
            <div className="divider" />
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              ⚠️ 本報告由 Claude AI 自動生成，僅供學術研究，不構成投資建議
            </div>
          </div>
        </div>

        {/* Right: market data from report */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Market summary from report */}
          {report?.market_data && (
            <div className="panel">
              <div className="panel-header">
                <div className="panel-title"><span>📡</span> 推論時市場快照</div>
              </div>
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  { label: '台股漲跌', value: `${(report.market_data.twii_change >= 0 ? '+' : '')}${(report.market_data.twii_change || 0).toFixed(2)}%`, up: report.market_data.twii_change >= 0 },
                  { label: '美股 S&P 500', value: `${(report.market_data.spx_change >= 0 ? '+' : '')}${(report.market_data.spx_change || 0).toFixed(2)}%`, up: report.market_data.spx_change >= 0 },
                  { label: 'VIX', value: (report.market_data.vix || 0).toFixed(2) },
                  { label: '黃金', value: `${(report.market_data.gold_change >= 0 ? '+' : '')}${(report.market_data.gold_change || 0).toFixed(2)}%`, up: report.market_data.gold_change >= 0 },
                  { label: 'USD/TWD', value: (report.market_data.usd_twd || 0).toFixed(3) },
                ].map(r => (
                  <div key={r.label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                    <span style={{ color: 'var(--text-muted)' }}>{r.label}</span>
                    <span className={`mono ${r.up === true ? 'text-positive' : r.up === false ? 'text-negative' : ''}`}>
                      {r.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Risk reminders */}
          <div className="panel" style={{ borderColor: 'rgba(255,165,0,0.2)', background: 'rgba(255,165,0,0.03)' }}>
            <div className="panel-header">
              <div className="panel-title"><span>⚠️</span> 風險提示</div>
            </div>
            <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                'AI 報告為自動生成，可能包含錯誤',
                '所有信號僅供研究，非投資建議',
                '過去績效不代表未來表現',
                '請在完整風控框架下操作',
              ].map((tip, i) => (
                <div key={i} style={{ fontSize: 12, color: 'var(--text-muted)', display: 'flex', gap: 8 }}>
                  <span style={{ color: 'var(--accent-amber)' }}>›</span>
                  <span>{tip}</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>

      {/* Top 10 table */}
      {!repLoading && report?.top10?.length > 0 && (
        <Top10Table top10={report.top10} />
      )}
    </div>
  );
}
