import client from './client';

// V6.2 組合層（規格 5d/20）：現行持股 + 再平衡倒數
// ⚠️ holdings 是「現在持有的」，不是「今天建議買進的」——
//    只有 is_rebalance_day=true 那天的變動才是交易動作。
export const fetchV62Portfolio = (arm) =>
  client.get('/v62/portfolio', { params: arm ? { arm } : {} }).then((r) => r.data);

// 並行跑的模型清單（實戰紀錄用，集合在起跑日就定案）
export const fetchV62Arms = () => client.get('/v62/arms').then((r) => r.data);

// 前瞻績效（真實 out-of-sample）。
// ⚠️ 與 arms 的 backtest_ann 是**兩種不同的東西**：這裡是實際跑出來的、樣本很小；
//    那裡是 582 天回測、樣本大但只有一個窗。顯示時必須標清楚，不可混在一起比大小。
export const fetchV62Performance = () =>
  client.get('/v62/performance').then((r) => r.data);
