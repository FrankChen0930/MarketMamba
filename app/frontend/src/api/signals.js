import client from './client';

export const fetchSignals = (params = {}) =>
  client.get('/signals', { params }).then((r) => r.data);

export const fetchSignalsByDate = (date) =>
  client.get(`/signals/${date}`).then((r) => r.data);

export const runInference = () =>
  client.post('/signals/run-inference').then((r) => r.data);

export const fetchRebalanceHistory = () =>
  client.get('/signals/history').then((r) => r.data);

