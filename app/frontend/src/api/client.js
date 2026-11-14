import axios from 'axios';

// Local dev  → Vite proxy handles /api → localhost:8000 (no env var needed)
// Production → VITE_API_URL=https://marketmamba-api.onrender.com
const BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

// timeout 15s → 60s（2026-08-01 修正）
// Render 免費方案 15 分鐘無流量會 spin down，冷啟動要 30–60 秒。
// 原本 15 秒的話，**spin down 之後的第一次請求必然 timeout**，
// 使用者看到的就是「一堆 API 連不上」，但後端其實只是還在起來。
// 實測 2026-08-01：/api/signals 在半熱狀態下就要 9.95 秒。
const client = axios.create({
  baseURL: BASE,
  timeout: 60000,
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

