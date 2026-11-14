import client from './client';

export const fetchSimBacktest     = () => client.get('/sim/backtest').then((r) => r.data);
export const fetchIcAnalysis      = () => client.get('/sim/ic').then((r) => r.data);
export const fetchScannerBacktest = () => client.get('/sim/scanner-backtest').then((r) => r.data);
