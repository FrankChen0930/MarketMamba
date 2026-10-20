// This fixture-only slice intentionally uses the local origin, never VITE_API_URL.
export async function getPortfolioHistory() {
  const response = await fetch('/api/v7/portfolio-history', { cache: 'no-store', signal: AbortSignal.timeout(15000) });
  if (!response.ok) return { state: 'error' };
  return response.json();
}
