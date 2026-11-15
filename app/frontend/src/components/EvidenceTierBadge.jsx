const LABELS = {
  VERIFIED: '已驗證 (VERIFIED)',
  RESEARCH_ONLY: '僅供研究 (RESEARCH_ONLY)',
  BLOCKED: '已阻擋 (BLOCKED)',
};

// Shared with V7 status. Callers validate admission before supplying a tier.
export default function EvidenceTierBadge({ tier }) {
  const safeTier = Object.hasOwn(LABELS, tier) ? tier : 'RESEARCH_ONLY';
  return <strong data-evidence-tier={safeTier}>{LABELS[safeTier]}</strong>;
}
