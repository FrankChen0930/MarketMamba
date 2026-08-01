import React from 'react';
import ModelStatus from './ModelStatus';

const CHANGELOG = [
  {
    version: 'V6.0', tag: '架構基線', color: 'var(--text-muted)',
    desc: 'Mamba SSM + GATv2 多尺度架構確立（Short/Mid/Long 三分支並行、FactorGroupedEmbedding、知識圖譜引導的 GATv2 融合）。',
  },
  {
    version: 'V6.1', tag: '線上生產版本', color: 'var(--positive)',
    desc: '56 維特徵，目前每日 17:00 實際推論、家人每日查看的 dashboard 依據此版本。checkpoint v6_best.pt（epoch 14，val_ic ≈ 0.0825）。',
  },
  {
    version: 'V6.2', tag: '訓練/資料修復中', color: 'var(--accent-amber)',
    desc: '59 維特徵（新增 RS 相對強度）、Zero-Padding Mask、資料管線大修（多資料源改交易所直連、除權息全歷史還原重建）。尚未上線取代 V6.1。',
  },
  {
    version: '雙模型（v6_short / v6_trend）', tag: '並行實驗上線', color: 'var(--accent-blue)',
    desc: 'rank-based 訓練目標，短線 5d/10d 與趨勢 20d/60d 各自訓練，作為「雙模型比較」子頁維度一內容，與 V6.1 並存、不影響線上主力。',
  },
];

// 廣度模型「版本紀錄」= 簡易 Model Card 時間軸 + 既有 ModelStatus（即時訓練狀態/IC曲線/Scale Gate）
export default function BreadthVersions() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="panel">
        <div className="panel-header"><div className="panel-title">🕰️ Model Card — 版本演進</div></div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {CHANGELOG.map(c => (
            <div key={c.version} style={{ display: 'flex', gap: 12, alignItems: 'flex-start', paddingBottom: 12, borderBottom: '1px solid var(--border)' }}>
              <span style={{ flexShrink: 0, fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-mono)', color: c.color, minWidth: 150 }}>{c.version}</span>
              <div>
                <div style={{ fontSize: 11, color: c.color, marginBottom: 3 }}>{c.tag}</div>
                <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.7 }}>{c.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
      <ModelStatus />
    </div>
  );
}
