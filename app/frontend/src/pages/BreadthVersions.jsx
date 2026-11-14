import React from 'react';
import { History } from 'lucide-react';
import ModelStatus from './ModelStatus';

const CHANGELOG = [
  {
    version: 'V6.0', tag: '打底', color: 'var(--text-muted)',
    desc: '模型骨架定案：用三種不同長度的時間窗同時看同一檔股票（近 20 天、近 60 天、近一年），再加一層讓它參考相關個股的機制。這個骨架後面幾版都沒有再動過。',
  },
  {
    version: 'V6.1', tag: '前一版，已停止更新', color: 'var(--ver-legacy)',
    desc: '第一個真的每天在跑的版本，用 56 項特徵。挑股票的規則是四個條件加權，門檻是自己訂的。內容移到「前一版」那一區保存。',
  },
  {
    version: '雙模型（短線 / 趨勢）', tag: '並行實驗，已停止更新', color: 'var(--ver-legacy)',
    desc: '同時訓練兩顆模型：一顆盯短天期、一顆盯長天期，用來看兩種目標會挑出多不一樣的股票。與 V6.1 同期，現在也收在「前一版」那一區。',
  },
  {
    version: 'V6.2', tag: '目前上線中', color: 'var(--ver-live)',
    desc: '特徵增加到 59 項，補上「相對大盤的強弱」這三欄。更重要的改動在挑股票那一層：改成直接取分數前 50 名等權買進、每 20 個交易日換一次，取代原本手訂的四條件。資料源也整批換成直接向交易所抓，並且把 11 年的除權息全部還原重算了一次。',
  },
];

// 廣度模型「版本紀錄」= 簡易 Model Card 時間軸 + 既有 ModelStatus（即時訓練狀態/IC曲線/Scale Gate）
export default function BreadthVersions() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <History size={16} strokeWidth={1.75} aria-hidden="true" />
            版本演進
          </div>
        </div>
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <p style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.8, margin: 0 }}>
            由舊到新。每一版之間改的東西不一樣：有的是換模型本身，有的是換「拿到分數之後怎麼買賣」——
            後者影響其實更大。
          </p>
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
