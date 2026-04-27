import React from 'react';
import { useApi } from '../hooks/useApi';
import { fetchMarket } from '../api/market';
import { SkeletonBlock, ApiError } from '../components/SkeletonLoader';

// Static mock news (will be replaced by Claude API later)
const NEWS = [
  { time: '09:12', title: '台積電法說會：Q2 業績指引優於市場預期，AI 需求持續旺盛', sentiment: 'positive', impact: '高', stocks: ['2330', '3034'] },
  { time: '10:34', title: '聯準會官員暗示年內降息可能延後，科技股承壓', sentiment: 'negative', impact: '中', stocks: [] },
  { time: '11:05', title: '鴻海與 NVIDIA 合作擴大 AI 伺服器生產線', sentiment: 'positive', impact: '中', stocks: ['2317'] },
  { time: '13:22', title: '航運業運費連續三週下滑，長榮、萬海面臨壓力', sentiment: 'negative', impact: '中', stocks: ['2603', '2615'] },
  { time: '14:18', title: '台灣 3 月出口年增 28.4%，連三個月創新高', sentiment: 'positive', impact: '高', stocks: [] },
];

const SENTIMENT_SCORES = [
  { name: '市場整體', score: 64, label: '偏多' },
  { name: '半導體',   score: 81, label: '強多' },
  { name: '金融',     score: 49, label: '中性' },
  { name: '航運',     score: 24, label: '偏空' },
  { name: '傳產',     score: 38, label: '偏空' },
];

const MARKET_STATE_CONFIG = {
  completed:  { label: 'LIVE',    color: 'var(--positive)',  desc: '推論已完成，訊號有效' },
  not_ready:  { label: 'PENDING', color: 'var(--accent-amber)', desc: 'Final Training 進行中，使用模擬訊號' },
  running:    { label: 'RUNNING', color: 'var(--accent-blue)',  desc: '推論正在執行中...' },
};

function SentimentBar({ score }) {
  const color = score >= 60 ? 'var(--positive)' : score >= 40 ? 'var(--accent-amber)' : 'var(--negative)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <div style={{ flex: 1, height: 6, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${score}%`, height: '100%', background: color, borderRadius: 3, boxShadow: `0 0 6px ${color}` }} />
      </div>
      <span className="mono" style={{ fontSize: 12, minWidth: 28, color }}>{score}</span>
    </div>
  );
}

export default function MarketView() {
  const { data: market, loading, error, refetch } = useApi(fetchMarket);
  const state = MARKET_STATE_CONFIG[market?.run_status] || MARKET_STATE_CONFIG.not_ready;

  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header"><div className="page-title">AI 消息面分析</div></div>
      <ApiError message={error} onRetry={refetch} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header">
        <div>
          <div className="page-title">AI 消息面分析</div>
          <div className="page-subtitle">Claude AI 整合 · 僅供參考，不影響量化訊號</div>
        </div>
        <span className="badge badge-neutral" style={{ fontSize: 12 }}>⚠️ LLM 生成，有幻覺風險</span>
      </div>

      {/* Market State Card — 參考六維度 AI 分析頁 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        {loading ? Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="stat-card"><SkeletonBlock height={40} /></div>
        )) : <>
          <div className="stat-card" style={{ borderColor: state.color, background: `color-mix(in srgb, ${state.color} 6%, var(--bg-panel))` }}>
            <div className="label">系統狀態</div>
            <div className="value mono" style={{ color: state.color, fontSize: 18 }}>{state.label}</div>
            <div className="sub">{state.desc}</div>
          </div>
          <div className="stat-card">
            <div className="label">訓練進度</div>
            <div className="value mono" style={{ fontSize: 18 }}>
              Ep {market?.training_epoch || '?'} / 100
            </div>
            <div className="sub">Val IC: +{market?.model_ic.toFixed(4)}</div>
          </div>
          <div className="stat-card">
            <div className="label">LLM 狀態</div>
            <div className="value mono" style={{ fontSize: 18, color: 'var(--accent-amber)' }}>PENDING</div>
            <div className="sub">Claude API 待申請</div>
          </div>
        </>}
      </div>

      {/* News + Sentiment */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>

        {/* News Feed */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>📰</span> 今日市場新聞</div>
            <span className="badge badge-blue">Claude 摘要（mock）</span>
          </div>
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {NEWS.map((n, i) => (
              <div key={i} style={{
                padding: '12px 14px', background: 'var(--bg-panel-2)', borderRadius: 8,
                borderLeft: `3px solid ${n.sentiment === 'positive' ? 'var(--positive)' : 'var(--negative)'}`,
                animationDelay: `${i * 0.07}s`,
              }} className="animate-fade-up">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{n.time}</span>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <span className={`badge ${n.sentiment === 'positive' ? 'badge-positive' : 'badge-negative'}`} style={{ fontSize: 10 }}>
                      {n.sentiment === 'positive' ? '利多' : '利空'}
                    </span>
                    <span className="badge badge-neutral" style={{ fontSize: 10 }}>影響 {n.impact}</span>
                  </div>
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.5 }}>{n.title}</div>
                {n.stocks.length > 0 && (
                  <div style={{ marginTop: 6, display: 'flex', gap: 4 }}>
                    {n.stocks.map(s => <span key={s} className="badge badge-blue" style={{ fontSize: 10 }}>{s}</span>)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Sentiment + AI Summary */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title"><span>🎭</span> 情緒指標</div>
            </div>
            <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {SENTIMENT_SCORES.map(s => (
                <div key={s.name}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, fontSize: 12 }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{s.name}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{s.label}</span>
                  </div>
                  <SentimentBar score={s.score} />
                </div>
              ))}
            </div>
          </div>

          <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.2)', background: 'rgba(0,212,255,0.03)' }}>
            <div className="panel-header">
              <div className="panel-title"><span>🤖</span> AI 每日摘要</div>
            </div>
            <div className="panel-body">
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                今日市場整體偏樂觀，台積電法說會提振半導體族群信心。
                AI 伺服器需求延續旺季格局，電子供應鏈受惠。
                航運受運費下滑壓力，建議觀望。
              </p>
              <div className="divider" />
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                ⚠️ 以上為 Claude AI 生成，僅作參考，不構成投資建議
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
