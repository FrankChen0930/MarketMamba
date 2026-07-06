import React, { useEffect, useRef, useState } from 'react';
import { init, dispose } from 'klinecharts';
import { fetchKline } from '../api/market';

// 台股慣例：紅漲綠跌
const UP_COLOR = '#f92855';
const DOWN_COLOR = '#2dc08e';

const CHART_STYLES = {
  grid: {
    horizontal: { color: 'rgba(255,255,255,0.05)' },
    vertical: { color: 'rgba(255,255,255,0.03)' },
  },
  candle: {
    bar: {
      upColor: UP_COLOR, downColor: DOWN_COLOR,
      upBorderColor: UP_COLOR, downBorderColor: DOWN_COLOR,
      upWickColor: UP_COLOR, downWickColor: DOWN_COLOR,
    },
    priceMark: {
      last: { upColor: UP_COLOR, downColor: DOWN_COLOR },
    },
    tooltip: {
      text: { color: '#c9d1d9' },
    },
  },
  indicator: {
    bars: [{ upColor: 'rgba(249,40,85,0.6)', downColor: 'rgba(45,192,142,0.6)' }],
    tooltip: { text: { color: '#8b949e' } },
  },
  xAxis: { axisLine: { color: 'rgba(255,255,255,0.1)' }, tickText: { color: '#8b949e' } },
  yAxis: { axisLine: { color: 'rgba(255,255,255,0.1)' }, tickText: { color: '#8b949e' } },
  separator: { color: 'rgba(255,255,255,0.08)' },
  crosshair: {
    horizontal: { line: { color: '#484f58' }, text: { backgroundColor: '#30363d' } },
    vertical: { line: { color: '#484f58' }, text: { backgroundColor: '#30363d' } },
  },
};

const RANGES = [
  { key: '3mo', label: '3月' },
  { key: '6mo', label: '6月' },
  { key: '1y', label: '1年' },
];

/**
 * 個股日 K 線圖（蠟燭 + 成交量 + MA5/20/60）。
 * failureStop / targetPrice 有值時畫成水平參考線（型態失敗止損價 / 目標價）。
 */
export default function KLineChart({ ticker, failureStop, targetPrice, height = 320 }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const [range, setRange] = useState('6mo');
  const [status, setStatus] = useState('loading');   // loading | ok | empty

  useEffect(() => {
    if (!containerRef.current) return undefined;
    const chart = init(containerRef.current);
    chart.setStyles(CHART_STYLES);
    chart.createIndicator({ name: 'MA', calcParams: [5, 20, 60] }, false, { id: 'candle_pane' });
    chart.createIndicator('VOL', false, { height: 60 });
    chartRef.current = chart;
    return () => {
      dispose(containerRef.current);
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!ticker) return;
    let cancelled = false;
    setStatus('loading');
    fetchKline(ticker, range)
      .then((data) => {
        if (cancelled || !chartRef.current) return;
        const candles = data?.candles || [];
        if (!candles.length) {
          setStatus('empty');
          return;
        }
        chartRef.current.applyNewData(candles.map((c) => ({
          timestamp: new Date(`${c.date}T00:00:00`).getTime(),
          open: c.open, high: c.high, low: c.low, close: c.close,
          volume: c.volume,
        })));

        // 關鍵價位水平線（先清舊的再畫，避免切 range 重複）
        chartRef.current.removeOverlay();
        const lines = [
          failureStop != null && {
            value: Number(failureStop), color: '#ff4757', text: `失敗止損 ${failureStop}`,
          },
          targetPrice != null && {
            value: Number(targetPrice), color: '#f5a623', text: `目標價 ${targetPrice}`,
          },
        ].filter(Boolean);
        lines.forEach((l) => {
          chartRef.current.createOverlay({
            name: 'horizontalStraightLine',
            points: [{ value: l.value }],
            lock: true,
            styles: {
              line: { color: l.color, style: 'dashed', size: 1 },
            },
            extendData: l.text,
          });
        });
        setStatus('ok');
      })
      .catch(() => { if (!cancelled) setStatus('empty'); });
    return () => { cancelled = true; };
  }, [ticker, range, failureStop, targetPrice]);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          日 K 線 · MA5/20/60 · 成交量
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          {RANGES.map((r) => (
            <button
              key={r.key}
              onClick={() => setRange(r.key)}
              className="btn btn-ghost"
              style={{
                padding: '2px 10px', fontSize: 11,
                color: range === r.key ? 'var(--accent-blue)' : 'var(--text-muted)',
                border: range === r.key ? '1px solid var(--accent-blue)' : '1px solid transparent',
                borderRadius: 6,
              }}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ position: 'relative' }}>
        <div ref={containerRef} style={{ width: '100%', height }} />
        {status !== 'ok' && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            fontSize: 12, color: 'var(--text-muted)',
            background: status === 'loading' ? 'transparent' : 'var(--bg-panel)',
          }}>
            {status === 'loading' ? '載入走勢中…' : '查無此股走勢資料'}
          </div>
        )}
      </div>

      {(failureStop != null || targetPrice != null) && status === 'ok' && (
        <div style={{ display: 'flex', gap: 14, marginTop: 6, fontSize: 11 }}>
          {failureStop != null && (
            <span style={{ color: '#ff4757' }}>▬ 型態失敗止損 {failureStop}</span>
          )}
          {targetPrice != null && (
            <span style={{ color: '#f5a623' }}>▬ 型態目標價 {targetPrice}</span>
          )}
        </div>
      )}
    </div>
  );
}
