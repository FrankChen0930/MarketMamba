import client from './client';

export const fetchMarket = () =>
  client.get('/market').then((r) => r.data);

export const fetchTicker = () =>
  client.get('/market/ticker').then((r) => r.data);

export const fetchKline = (ticker, range = '6mo') =>
  client.get(`/market/kline/${ticker}`, { params: { range } }).then((r) => r.data);
