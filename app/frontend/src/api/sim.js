import client from './client';

export const fetchSimBacktest = () =>
  client.get('/sim/backtest').then((r) => r.data);
