import client from './client';

export const fetchMarket = () =>
  client.get('/market').then((r) => r.data);

export const fetchTicker = () =>
  client.get('/market/ticker').then((r) => r.data);
