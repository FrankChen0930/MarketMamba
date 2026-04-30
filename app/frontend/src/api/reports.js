import { client } from './client';

export async function fetchReport() {
  return client('/api/reports/latest');
}

export async function refreshReportCache() {
  const r = await fetch(
    (import.meta.env.VITE_API_URL || 'https://marketmamba-api.onrender.com') + '/api/reports/cache/refresh',
    { method: 'POST' }
  );
  return r.json();
}
