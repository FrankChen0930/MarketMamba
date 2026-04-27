import client from './client';

export const fetchPortfolio = () =>
  client.get('/portfolio').then((r) => r.data);
