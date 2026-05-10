import React, { useState, useRef, useEffect } from 'react';

const METRIC_INFO = {
  rank_stability: {
    title: '排名穩定性',
    desc: '股票在過去數天連續出現在模型 Top N 排名中。穩定出現代表模型對該股看法一致，不是單日噪音。',
    threshold: 'Top 10 連續 ≥2 天，或 Top 50 連續 ≥3 天',
  },
  high_confidence: {
    title: '模型信心度',
    desc: '基於 MC-Dropout（蒙特卡羅 Dropout）的不確定度。模型跑多次推論，結果越一致代表越有把握。',
    threshold: 'Uncertainty < 0.02 為高信心',
  },
  relative_low: {
    title: '相對低點',
    desc: 'RSI（相對強弱指標）衡量股票近期漲跌動能。低於 40 代表可能處於超賣區間，適合逢低佈局。MA20 為 20 日均線，股價低於均線代表短期回檔。',
    threshold: 'RSI < 40 或 股價 < 20 日均線',
  },
  institutional_buy: {
    title: '機構買賣超',
    desc: '外資/投信等法人的連續買入行為。大型資金的持續買進通常代表機構看好該股基本面或有策略性佈局。',
    threshold: '外資連續 ≥2 天淨買入',
  },
  sharpe: {
    title: 'Sharpe Score',
    desc: '風險調整後報酬指標。計算方式為「預期報酬 ÷ 不確定度」。數值越高代表每承受一單位風險能獲得更多報酬。',
    threshold: '> 3 為高，1-3 為中，< 1 為低',
  },
  kelly: {
    title: 'Kelly 倉位',
    desc: '根據凱利公式（Kelly Criterion）計算的最適資金配置比例。考量預期報酬和風險後，建議投入總資金的百分比。',
    threshold: '通常建議使用 Half-Kelly（一半的建議值）',
  },
  alpha: {
    title: 'Alpha 預測',
    desc: '模型預測該股票相對大盤（加權指數）的超額報酬。正值代表預期跑贏大盤，負值代表預期跑輸。',
    threshold: '以 20d Alpha 為主要參考',
  },
  uncertainty: {
    title: '不確定度',
    desc: '模型對預測結果的不確定程度。透過 MC-Dropout 多次推論取標準差。越低代表模型越有信心。',
    threshold: '< 0.02 高信心，< 0.04 中信心，≥ 0.04 低信心',
  },
  market_regime: {
    title: '大盤環境',
    desc: '以 TWII（加權指數）相對 60 日均線的位置判斷市場環境。大盤在均線之上為正常市場，之下為保守市場。保守市場時入場條件更嚴格。',
    threshold: 'TWII > MA60 → 正常（2/4），TWII < MA60 → 保守（3/4）',
  },
};

export function getMetricInfo(key) {
  return METRIC_INFO[key] || { title: key, desc: '', threshold: '' };
}

export default function MetricTooltip({ metricKey, style }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const info = getMetricInfo(metricKey);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <span ref={ref} style={{ position: 'relative', display: 'inline-flex', ...style }}>
      <button
        onClick={(e) => { e.stopPropagation(); setOpen(!open); }}
        style={{
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--text-muted)', fontSize: 11, padding: '0 2px',
          lineHeight: 1, fontWeight: 700, opacity: 0.6,
          transition: 'opacity 0.15s',
        }}
        onMouseEnter={(e) => e.target.style.opacity = 1}
        onMouseLeave={(e) => e.target.style.opacity = 0.6}
        title={info.title}
      >
        ?
      </button>

      {open && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'absolute', top: '100%', left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 999, width: 280,
            background: 'var(--bg-panel)',
            border: '1px solid var(--border-bright)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-glow-blue)',
            padding: '14px 16px',
            animation: 'fadeInUp 0.15s ease forwards',
            marginTop: 6,
          }}
        >
          <div style={{
            fontSize: 13, fontWeight: 700, color: 'var(--accent-blue)',
            marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <span style={{ fontSize: 14 }}>💡</span>
            {info.title}
          </div>
          <div style={{
            fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6,
            marginBottom: 10,
          }}>
            {info.desc}
          </div>
          <div style={{
            fontSize: 11, color: 'var(--accent-amber)',
            padding: '6px 10px',
            background: 'rgba(255,165,0,0.06)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid rgba(255,165,0,0.15)',
          }}>
            ⚡ {info.threshold}
          </div>
        </div>
      )}
    </span>
  );
}
