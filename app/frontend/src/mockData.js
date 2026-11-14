// Mock data for development (replace with API calls when backend is ready)

export const MOCK_SIGNALS = [
  { rank: 1,  stock_id: "2330", name: "台積電",    alpha: 0.187, sector: "半導體", ic_contrib: 0.024, vol_ratio: 1.32, signal: "BUY" },
  { rank: 2,  stock_id: "2454", name: "聯發科",    alpha: 0.162, sector: "半導體", ic_contrib: 0.019, vol_ratio: 1.18, signal: "BUY" },
  { rank: 3,  stock_id: "2317", name: "鴻海",      alpha: 0.148, sector: "電子製造", ic_contrib: 0.017, vol_ratio: 0.95, signal: "BUY" },
  { rank: 4,  stock_id: "2382", name: "廣達",      alpha: 0.139, sector: "電子製造", ic_contrib: 0.016, vol_ratio: 1.05, signal: "BUY" },
  { rank: 5,  stock_id: "3034", name: "聯詠",      alpha: 0.131, sector: "半導體", ic_contrib: 0.014, vol_ratio: 0.87, signal: "BUY" },
  { rank: 6,  stock_id: "2357", name: "華碩",      alpha: 0.125, sector: "電子製造", ic_contrib: 0.013, vol_ratio: 0.92, signal: "BUY" },
  { rank: 7,  stock_id: "2308", name: "台達電",    alpha: 0.118, sector: "電子零組件", ic_contrib: 0.012, vol_ratio: 1.11, signal: "BUY" },
  { rank: 8,  stock_id: "2379", name: "瑞昱",      alpha: 0.112, sector: "半導體", ic_contrib: 0.011, vol_ratio: 0.78, signal: "HOLD" },
  { rank: 9,  stock_id: "6505", name: "台塑石化",  alpha: 0.098, sector: "化工", ic_contrib: 0.009, vol_ratio: 0.85, signal: "HOLD" },
  { rank: 10, stock_id: "2881", name: "富邦金",    alpha: 0.087, sector: "金融", ic_contrib: 0.008, vol_ratio: 1.02, signal: "HOLD" },
  { rank: 2879, stock_id: "1301", name: "台塑", alpha: -0.132, sector: "化工", ic_contrib: -0.015, vol_ratio: 0.72, signal: "SELL" },
  { rank: 2880, stock_id: "9904", name: "寶成", alpha: -0.141, sector: "其他", ic_contrib: -0.016, vol_ratio: 0.68, signal: "SELL" },
  { rank: 2881, stock_id: "2002", name: "中鋼", alpha: -0.155, sector: "鋼鐵", ic_contrib: -0.018, vol_ratio: 0.61, signal: "SELL" },
  { rank: 2882, stock_id: "2603", name: "長榮", alpha: -0.168, sector: "航運", ic_contrib: -0.019, vol_ratio: 0.58, signal: "SELL" },
  { rank: 2883, stock_id: "2615", name: "萬海", alpha: -0.179, sector: "航運", ic_contrib: -0.021, vol_ratio: 0.54, signal: "SELL" },
];

export const MOCK_IC_HISTORY = Array.from({ length: 20 }, (_, i) => ({
  epoch: i + 1,
  train_loss: 2.8 - i * 0.08 + Math.random() * 0.05,
  val_loss:   2.5 - i * 0.05 + Math.random() * 0.04,
  val_ic:     Math.min(0.10, 0.01 + i * 0.005 + Math.random() * 0.003),
}));

export const MOCK_PORTFOLIO = [
  { stock_id: "2330", name: "台積電", qty: 1000, avg_price: 830.0, current_price: 915.0, pnl: 85000, pnl_pct: 10.24 },
  { stock_id: "2454", name: "聯發科", qty: 500,  avg_price: 1180.0, current_price: 1205.0, pnl: 12500, pnl_pct: 2.12 },
  { stock_id: "2317", name: "鴻海",   qty: 2000, avg_price: 145.0, current_price: 139.5, pnl: -11000, pnl_pct: -3.79 },
];

export const MOCK_MARKET = {
  taiex:      { value: 22381.4, change: +87.3, change_pct: +0.39 },
  volume:     { value: "2,841億", label: "成交金額" },
  advancing:  563,
  declining:  412,
  model_ic:   0.0744,
  last_run:   "2026-04-27 15:30",
  run_status: "completed",
};

export const MOCK_TICKER = [
  { id: "TAIEX",  name: "加權",  price: "22,381", change: "+87.3",  pct: "+0.39%", up: true },
  { id: "2330",   name: "台積電", price: "915.0",  change: "+12.0",  pct: "+1.33%", up: true },
  { id: "2454",   name: "聯發科", price: "1,205",  change: "+25.0",  pct: "+2.12%", up: true },
  { id: "2317",   name: "鴻海",   price: "139.5",  change: "-2.5",   pct: "-1.76%", up: false },
  { id: "2382",   name: "廣達",   price: "278.0",  change: "+6.5",   pct: "+2.40%", up: true },
  { id: "2881",   name: "富邦金", price: "82.5",   change: "-0.8",   pct: "-0.96%", up: false },
  { id: "TAIEX",  name: "加權",  price: "22,381", change: "+87.3",  pct: "+0.39%", up: true },
  { id: "2330",   name: "台積電", price: "915.0",  change: "+12.0",  pct: "+1.33%", up: true },
  { id: "2454",   name: "聯發科", price: "1,205",  change: "+25.0",  pct: "+2.12%", up: true },
];
