import React from 'react';
import { ChartLine } from 'lucide-react';
import ComingSoon from '../components/ComingSoon';

export default function ConvictionBacktest() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <ComingSoon
        icon={<ChartLine size={16} strokeWidth={1.75} />}
        title="回測結果"
        desc="這條線有一半的東西是回測不動的，所以這頁打算把它拆成兩塊，不混在一起講。"
        bullets={[
          '可以驗證的部分：那些在歷史上重複發生夠多次的情況——例如財報遠優於預期之後股價會不會續強、法人由賣轉買之後會不會反彈、除權息前後有沒有規律。這些能拿 12 年資料真的跑一遍。',
          '驗證不動的部分：靠人判斷的那一段。同樣的資訊，我今年的判斷跟三年前不會一樣，沒辦法回頭重跑。這塊只能上線之後一天一天記，並且把每次判斷的理由一起寫下來。',
          '把兩者分開列，是因為混在一起會讓後者借用前者的可信度。',
        ]}
      />
    </div>
  );
}
