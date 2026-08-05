import client from './client';

// V6.2 組合層（規格 5d/20）：現行持股 + 再平衡倒數
// ⚠️ holdings 是「現在持有的」，不是「今天建議買進的」——
//    只有 is_rebalance_day=true 那天的變動才是交易動作。
export const fetchV62Portfolio = (arm) =>
  client.get('/v62/portfolio', { params: arm ? { arm } : {} }).then((r) => r.data);

// 並行跑的模型清單（實戰紀錄用，集合在起跑日就定案）
export const fetchV62Arms = () => client.get('/v62/arms').then((r) => r.data);
