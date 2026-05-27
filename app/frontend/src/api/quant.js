/**
 * Quant Analysis API
 * Fetches market-wide quantitative data and chart pattern scan results
 * from the MarketMamba backend.
 */

const BASE = import.meta.env.VITE_API_URL || '';

/**
 * GET /api/quant
 * Returns daily market-wide quantitative indicators:
 *   TAIEX technicals, institutional flow, breadth history,
 *   margin/short summary, sector rotation.
 */
export async function fetchQuant() {
  const res = await fetch(`${BASE}/api/quant`);
  if (!res.ok) throw new Error(`Quant API error: ${res.status}`);
  return res.json();
}

/**
 * GET /api/quant/patterns[?dual_confirm_only=true]
 * Returns bullish chart pattern scan results.
 *
 * @param {boolean} dualConfirmOnly - when true, filter to dual-confirm signals only
 */
export async function fetchPatterns(dualConfirmOnly = false) {
  const url = dualConfirmOnly
    ? `${BASE}/api/quant/patterns?dual_confirm_only=true`
    : `${BASE}/api/quant/patterns`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Patterns API error: ${res.status}`);
  return res.json();
}
