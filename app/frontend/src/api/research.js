import client from './client.js';
import { normalizeResearch } from './researchContract.mjs';
export async function getResearchSummary() {
  try {
    return normalizeResearch((await client.get('/research/summary')).data);
  } catch {
    return normalizeResearch(null);
  }
}
