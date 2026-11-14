import client from './client';

export const fetchPerformance = () =>
  client.get('/performance').then((r) => r.data);

export const fetchICHistory = () =>
  client.get('/performance/ic').then((r) => r.data);
