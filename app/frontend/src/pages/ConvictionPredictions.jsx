import React from 'react';
import { FileText } from 'lucide-react';
import ComingSoon from '../components/ComingSoon';

export default function ConvictionPredictions() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <ComingSoon
        icon={<FileText size={16} strokeWidth={1.75} />}
        title="個股研究卡"
        desc="這頁之後會放一疊卡片，一檔股票一張。每張卡回答同樣五個問題：為什麼看好、現在的價格算貴還便宜、接下來有什麼事可能推動它、最大的風險是什麼，還有——什麼情況出現就代表我看錯了。"
        bullets={[
          '最後那一項是刻意放進去的。先寫下認錯的條件，逆風的時候才不會臨時替自己找理由。',
          '目前只有版面，還沒有真的內容。要先把「讓 AI 讀完一檔股票並輸出這五段」這條流程做出來。',
        ]}
      />
    </div>
  );
}
