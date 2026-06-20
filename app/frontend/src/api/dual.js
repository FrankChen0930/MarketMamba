import client from './client';

// 雙模型訊號（短線 5d/10d + 趨勢 20d/60d），rank-score 語意
export const fetchDualSignals = (params = {}) =>
  client.get('/dual/signals', { params }).then((r) => r.data);
