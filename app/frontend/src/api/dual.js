import client from './client';

// 雙模型訊號（短線 5d/10d + 趨勢 20d/60d），rank-score 語意
export const fetchDualSignals = (params = {}) =>
  client.get('/dual/signals', { params }).then((r) => r.data);

// 雙模型用真實市場走勢驗證的效益分析（IC / ICIR / Top50 實現超額報酬）
export const fetchDualIc = () =>
  client.get('/dual/ic').then((r) => r.data);
