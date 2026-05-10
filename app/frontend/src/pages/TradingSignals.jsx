import React, { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { fetchScannerSignals } from '../api/signals';
import MetricTooltip from '../components/MetricTooltip';
import StockModal from '../components/StockModal';
import { SkeletonCard, SkeletonBlock, ApiError } from '../components/SkeletonLoader';

// ── Helpers ──────────────────────────────────────────────────────────────────

function ConditionDot({ met, label, metricKey }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 5,
      fontSize: 12, color: met ? 'var(--positive)' : 'var(--text-muted)',
    }}>
      <span style={{ fontSize: 13 }}>{met ? '✅' : '❌'}</span>
      <span>{label}</span>
      <MetricTooltip metricKey={metricKey} />
    </div>
  );
}

function SignalCard({ signal, type, onClick }) {
  const isBuy = type === 'buy';
  const isExit = type === 'exit';
  const borderColor = isBuy ? 'rgba(0,255,136,0.25)' : isExit ? 'rgba(255,71,87,0.25)' : 'var(--border)';
  const bgTint = isBuy ? 'rgba(0,255,136,0.03)' : isExit ? 'rgba(255,71,87,0.03)' : 'transparent';

  return (
    <div
      className="panel animate-fade-up"
      onClick={onClick}
      style={{
        borderColor, background: bgTint,
        cursor: 'pointer', transition: 'all 0.2s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = isBuy ? 'rgba(0,255,136,0.5)' : isExit ? 'rgba(255,71,87,0.5)' : 'var(--border-bright)';
        e.currentTarget.style.boxShadow = isBuy ? 'var(--shadow-glow-green)' : isExit ? '0 0 20px rgba(255,71,87,0.15)' : 'var(--shadow-glow-blue)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = borderColor;
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      <div className="panel-body" style={{ padding: '16px 20px' }}>
        {/* Top row: ticker + alpha */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
              {signal.name && signal.name !== signal.ticker ? (
                <>{signal.name} <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 400 }}>{signal.ticker}</span></>
              ) : signal.ticker}
              <span style={{
                marginLeft: 8, fontSize: 11,
                color: isBuy ? 'var(--positive)' : isExit ? 'var(--negative)' : 'var(--accent-amber)',
                background: isBuy ? 'rgba(0,255,136,0.1)' : isExit ? 'rgba(255,71,87,0.1)' : 'rgba(255,165,0,0.1)',
                padding: '2px 8px', borderRadius: 99,
                fontFamily: 'var(--font-mono)', fontWeight: 600,
              }}>
                {isBuy ? '買入推薦' : isExit ? '退場警告' : '觀察中'}
              </span>
            </div>
            {signal.confidence && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                {signal.confidence} · 滿足 {signal.conditions_met}/{signal.conditions_total} 條件
              </div>
            )}
          </div>
          {signal.alpha_20d != null && (
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Alpha 20d <MetricTooltip metricKey="alpha" />
              </div>
              <div className="mono" style={{
                fontSize: 20, fontWeight: 700,
                color: signal.alpha_20d >= 0 ? 'var(--positive)' : 'var(--negative)',
              }}>
                {signal.alpha_20d >= 0 ? '+' : ''}{(signal.alpha_20d * 100).toFixed(1)}%
              </div>
            </div>
          )}
        </div>

        {/* Conditions grid */}
        {signal.rank_stability && (
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px',
            padding: '10px 14px', background: 'var(--bg-panel-2)',
            borderRadius: 'var(--radius-sm)', marginBottom: 12,
          }}>
            <ConditionDot met={signal.rank_stability?.met} label={signal.rank_stability?.detail} metricKey="rank_stability" />
            <ConditionDot met={signal.high_confidence?.met} label={signal.high_confidence?.detail} metricKey="high_confidence" />
            <ConditionDot met={signal.relative_low?.met} label={signal.relative_low?.detail} metricKey="relative_low" />
            <ConditionDot met={signal.institutional_buy?.met} label={signal.institutional_buy?.detail} metricKey="institutional_buy" />
          </div>
        )}

        {/* Exit reasons */}
        {isExit && signal.reasons && (
          <div style={{
            padding: '10px 14px', background: 'rgba(255,71,87,0.06)',
            borderRadius: 'var(--radius-sm)', marginBottom: 12,
            border: '1px solid rgba(255,71,87,0.15)',
          }}>
            {signal.reasons.map((r, i) => (
              <div key={i} style={{ fontSize: 12, color: 'var(--negative)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>🔴</span> {r}
              </div>
            ))}
          </div>
        )}

        {/* Bottom metrics row */}
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {signal.sharpe != null && (
            <div style={{ fontSize: 12 }}>
              <span style={{ color: 'var(--text-muted)' }}>Sharpe </span>
              <MetricTooltip metricKey="sharpe" />
              <span className="mono" style={{
                color: signal.sharpe > 3 ? 'var(--positive)' : 'var(--text-secondary)',
                fontWeight: 600,
              }}>{signal.sharpe.toFixed(1)}</span>
            </div>
          )}
          {signal.uncertainty != null && (
            <div style={{ fontSize: 12 }}>
              <span style={{ color: 'var(--text-muted)' }}>不確定度 </span>
              <MetricTooltip metricKey="uncertainty" />
              <span className="mono" style={{
                color: signal.uncertainty < 0.02 ? 'var(--positive)' : 'var(--accent-amber)',
                fontWeight: 600,
              }}>±{(signal.uncertainty * 100).toFixed(1)}%</span>
            </div>
          )}
          {signal.suggested_weight != null && (
            <div style={{ fontSize: 12 }}>
              <span style={{ color: 'var(--text-muted)' }}>Kelly </span>
              <MetricTooltip metricKey="kelly" />
              <span className="mono" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>
                {(signal.suggested_weight * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


// ── Main Page ────────────────────────────────────────────────────────────────

export default function TradingSignals() {
  const { data, loading, error, refetch } = useApi(fetchScannerSignals);
  const [selectedStock, setSelectedStock] = useState(null);

  const buySignals = data?.buy_signals || [];
  const exitSignals = data?.exit_signals || [];
  const watchList = data?.watch_list || [];

  const regimeColor = data?.market_regime === 'CAUTIOUS' ? 'var(--accent-amber)' : 'var(--positive)';
  const regimeLabel = data?.market_regime === 'CAUTIOUS' ? '保守模式' : '正常市場';
  const regimeIcon = data?.market_regime === 'CAUTIOUS' ? '🟡' : '🟢';

  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header"><div className="page-title">交易訊號掃描</div></div>
      <ApiError message={error} onRetry={refetch} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {selectedStock && <StockModal stock={selectedStock} onClose={() => setSelectedStock(null)} />}

      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-title">🎯 交易訊號掃描</div>
          <div className="page-subtitle">
            MarketMamba V6.1 Signal Scanner · {data?.date || '—'}
          </div>
        </div>
        <button className="btn btn-primary" onClick={refetch}>🔄 重新掃描</button>
      </div>

      {/* Market Regime Strip */}
      <div className="grid-4">
        {loading ? Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />) : <>
          <div className="stat-card" style={{
            borderColor: regimeColor,
            background: `color-mix(in srgb, ${regimeColor} 5%, var(--bg-panel))`,
          }}>
            <div className="label">
              大盤環境 <MetricTooltip metricKey="market_regime" />
            </div>
            <div className="value" style={{ color: regimeColor }}>
              {regimeIcon} {regimeLabel}
            </div>
            <div className="sub">TWII vs MA60: {data?.twii_vs_ma60 || '—'}</div>
          </div>

          <div className="stat-card">
            <div className="label">入場門檻</div>
            <div className="value mono">{data?.entry_threshold || '—'}</div>
            <div className="sub">滿足條件數 / 總條件</div>
          </div>

          <div className="stat-card" style={{
            borderColor: buySignals.length > 0 ? 'rgba(0,255,136,0.3)' : 'var(--border)',
          }}>
            <div className="label">🔥 買入推薦</div>
            <div className="value mono text-positive">{buySignals.length} 檔</div>
            <div className="sub">滿足入場條件</div>
          </div>

          <div className="stat-card" style={{
            borderColor: exitSignals.length > 0 ? 'rgba(255,71,87,0.3)' : 'var(--border)',
          }}>
            <div className="label">⚠️ 退場警告</div>
            <div className="value mono" style={{
              color: exitSignals.length > 0 ? 'var(--negative)' : 'var(--text-secondary)',
            }}>{exitSignals.length} 檔</div>
            <div className="sub">觸發退場條件</div>
          </div>
        </>}
      </div>

      {/* Buy Signals Section */}
      {!loading && (
        <div>
          <div style={{
            fontSize: 14, fontWeight: 700, color: 'var(--positive)',
            marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <span style={{ fontSize: 18 }}>🔥</span>
            買入推薦
            <span className="badge badge-positive" style={{ fontSize: 10 }}>{buySignals.length} 檔</span>
          </div>
          {buySignals.length === 0 ? (
            <div className="panel">
              <div className="panel-body" style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>
                目前沒有股票達到入場條件 — 等待更好的機會 🧘
              </div>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: 12 }}>
              {buySignals.map((s, i) => (
                <div key={s.ticker} style={{ animationDelay: `${i * 0.05}s` }}>
                  <SignalCard signal={s} type="buy" onClick={() => setSelectedStock({
                    stock_id: s.ticker, name: s.name || s.ticker, sector: s.sector || '—',
                    alpha_5d: 0, alpha_20d: s.alpha_20d, alpha_60d: 0,
                    uncertainty: s.uncertainty, vol_ratio: 1, signal: 'BUY',
                    suggested_weight: s.suggested_weight, confidence: s.confidence,
                    rank: buySignals.indexOf(s) + 1,
                  })} />
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Exit Signals Section */}
      {!loading && exitSignals.length > 0 && (
        <div>
          <div style={{
            fontSize: 14, fontWeight: 700, color: 'var(--negative)',
            marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <span style={{ fontSize: 18 }}>🔴</span>
            退場警告
            <span className="badge badge-negative" style={{ fontSize: 10 }}>{exitSignals.length} 檔</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: 12 }}>
            {exitSignals.map((s, i) => (
              <div key={s.ticker} style={{ animationDelay: `${i * 0.05}s` }}>
                <SignalCard signal={s} type="exit" />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Watch List Section */}
      {!loading && watchList.length > 0 && (
        <div>
          <div style={{
            fontSize: 14, fontWeight: 700, color: 'var(--accent-amber)',
            marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <span style={{ fontSize: 18 }}>👀</span>
            觀察清單
            <span className="badge badge-neutral" style={{ fontSize: 10 }}>
              差 1 個條件 · {watchList.length} 檔
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: 12 }}>
            {watchList.map((s, i) => (
              <div key={s.ticker} style={{ animationDelay: `${i * 0.05}s` }}>
                <SignalCard signal={s} type="watch" onClick={() => setSelectedStock({
                  stock_id: s.ticker, name: s.name || s.ticker, sector: s.sector || '—',
                  alpha_5d: 0, alpha_20d: s.alpha_20d, alpha_60d: 0,
                  uncertainty: s.uncertainty, vol_ratio: 1, signal: 'HOLD',
                  suggested_weight: s.suggested_weight, confidence: s.confidence,
                  rank: '-',
                })} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* How it works */}
      {!loading && (
        <div className="panel" style={{ borderColor: 'rgba(0,212,255,0.15)', background: 'rgba(0,212,255,0.02)' }}>
          <div className="panel-body" style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
            <span style={{ fontSize: 20 }}>💡</span>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
              <strong style={{ color: 'var(--accent-blue)' }}>運作原理：</strong>
              掃描模型 Top 50 股票，評估 4 個維度的入場條件（排名穩定性、模型信心、相對低點、機構資金方向）。
              {data?.market_regime === 'CAUTIOUS'
                ? '目前大盤低於 60 日均線，進入保守模式，需滿足 3/4 條件才推薦。'
                : '正常市場環境下，滿足 2/4 條件即推薦買入。'
              }
              點擊每個指標旁的 <strong>?</strong> 查看詳細說明。
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
