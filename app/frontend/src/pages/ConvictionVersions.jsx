import React from 'react';
import { History } from 'lucide-react';
import ComingSoon from '../components/ComingSoon';

export default function ConvictionVersions() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <ComingSoon
        icon={<History size={16} strokeWidth={1.75} />}
        title="版本紀錄"
        desc="這頁會記兩件事的變動：篩選條件（門檻調高調低、加了什麼新條件），以及問 AI 的方式（每一版的提問怎麼寫、為什麼改）。"
        bullets={[
          '記提問的版本聽起來很瑣碎，但同一檔股票、同一份資料，換個問法就會得到不同的結論——不記下來的話，之後根本分不清是市場變了還是我改了問法。',
          '篩選條件目前先沿用前一版（V6.1）那一套，等這條線真的跑起來再換掉。',
        ]}
      />
    </div>
  );
}
