import client from './client.js';
import { normalizeV7Status } from './v7Contract.mjs';

export { normalizeV7Status };

export async function getV7Status() {
  try {
    const response = await client.get('/v7/status');
    return normalizeV7Status(response.data);
  } catch (error) {
    if (error?.response?.data) {
      return normalizeV7Status(error.response.data);
    }
    throw error;
  }
}
