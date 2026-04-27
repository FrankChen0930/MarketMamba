import axios from 'axios';

// Local dev  → Vite proxy handles /api → localhost:8000 (no env var needed)
// Production → VITE_API_URL=https://marketmamba-api.onrender.com
const BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

const client = axios.create({
  baseURL: BASE,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// Response interceptor — log errors in dev
client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (import.meta.env.DEV) {
      console.warn('[MarketMamba API]', err.config?.url, err.message);
    }
    return Promise.reject(err);
  }
);

export default client;

