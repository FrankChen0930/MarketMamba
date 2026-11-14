import React from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  ResponsiveContainer, CartesianGrid, Legend, ReferenceLine
} from 'recharts';
import { useApi } from '../hooks/useApi';
import { fetchPerformance } from '../api/performance';
import { fetchMarket } from '../api/market';
import { SkeletonBlock, SkeletonCard, ApiError } from '../components/SkeletonLoader';
import MetricTooltip from '../components/MetricTooltip';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass" style={{ padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontFamily: 'var(--font-mono)' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
};

// 訓練狀態 → badge 樣式與文字
function statusBadge(training) {
  if (!training) return { cls: 'badge-neutral', text: '尚無訓練紀錄' };
  switch (training.status) {
    case 'training':
      return { cls: 'badge-blue', text: `⚙️ 訓練中 ep ${training.epoch}/${training.epochs_max}` };
    case 'early_stopped':
      return { cls: 'badge-positive', text: `✓ 訓練完成（早停 @ep${training.epoch}）` };
    case 'completed':
      return { cls: 'badge-positive', text: '✓ 訓練完成' };
    default:
      return { cls: 'badge-neutral', text: training.status };
  }
}

const fmt = (v, d = 4, signed = false) =>
  (v === null || v === undefined || Number.isNaN(v))
    ? '—'
    : `${signed && v > 0 ? '+' : ''}${Number(v).toFixed(d)}`;

export default function ModelStatus() {
  const { data: perf, loading, error, refetch } = useApi(fetchPerformance);
  const { data: market } = useApi(fetchMarket);

  const training   = perf?.training || null;
  const icHist     = perf?.ic_history || [];
  const scaleGates = perf?.scale_gates || [];
  const onlineIC   = perf?.online_ic || [];
  const summary    = perf?.online_summary || [];
  const cfg        = training?.config || {};

  const sum5d  = summary.find(s => s.horizon === '5d');
  const sum20d = summary.find(s => s.horizon === '20d');
  const badge  = statusBadge(training);
  const hasAnyData = !!training || onlineIC.length > 0;

  if (error) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header"><div className="page-title">模型狀態</div></div>
      <ApiError message={error} onRetry={refetch} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="page-header">
        <div>
          <div className="page-title">🧠 MarketMamba {training?.model_version || 'V6'} 模型狀態</div>
          <div className="page-subtitle">
            Multi-Scale Mamba + GATv2
            {cfg.input_dim ? ` · ${cfg.input_dim} 因子` : ''}
            {training?.updated_at ? ` · 更新於 ${training.updated_at.replace('T', ' ')}` : ''}
          </div>
        </div>
        <span className={`badge ${badge.cls}`}>{badge.text}</span>
      </div>

      {/* 空狀態：尚無 training_status.json 也無線上 IC */}
      {!loading && !hasAnyData && (
        <div className="panel">
          <div className="panel-body" style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
            <div style={{ fontSize: 28, marginBottom: 12 }}>📭</div>
            <div style={{ fontSize: 14, marginBottom: 6 }}>尚無訓練紀錄</div>
            <div style={{ fontSize: 12 }}>
              Colab 訓練會將 training_status.json 寫入 Google Drive；
              訓練完成後複製到 V6/results/ 並 git push，本頁即會顯示。
            </div>
          </div>
        </div>
      )}

      {/* KPIs */}
      <div className="grid-4">
        {loading ? Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />) : <>
          <div className="stat-card">
            <div className="label">訓練 Best Val IC <MetricTooltip metricKey="ic" /></div>
            <div className={`value mono ${training?.best_val_ic >= 0.05 ? 'text-positive' : ''}`} style={{ fontSize: 20 }}>
              {fmt(training?.best_val_ic, 4, true)}
            </div>
            <div className="sub">
              {training ? `@ epoch ${training.best_ic_epoch} / ${training.epoch}・目標 ≥ 0.05${training.best_val_ic >= 0.05 ? ' ✓' : ''}` : '等待訓練資料'}
            </div>
          </div>
          <div className="stat-card">
            <div className="label">線上 IC（5d） <MetricTooltip metricKey="ic" /></div>
            <div className={`value mono ${sum5d?.mean_ic >= 0.05 ? 'text-positive' : ''}`} style={{ fontSize: 20 }}>
              {fmt(sum5d?.mean_ic, 4, true)}
            </div>
            <div className="sub">
              {sum5d ? `ICIR ${fmt(sum5d.icir, 2)}・${sum5d.n_days} 天・IC>0 ${fmt(sum5d.ic_gt0_pct, 0)}%` : '尚無資料'}
            </div>
          </div>
          <div className="stat-card">
            <div className="label">線上 IC（20d） <MetricTooltip metricKey="ic" /></div>
            <div className={`value mono ${sum20d?.mean_ic >= 0.05 ? 'text-positive' : ''}`} style={{ fontSize: 20 }}>
              {fmt(sum20d?.mean_ic, 4, true)}
            </div>
            <div className="sub">
              {sum20d
                ? `ICIR ${fmt(sum20d.icir, 2)}・${sum20d.n_days} 天${sum20d.n_days < 30 ? '（樣本少，觀察中）' : ''}`
                : '尚無資料'}
            </div>
          </div>
          <div className="stat-card">
            <div className="label">最後推論日期</div>
            <div className="value mono" style={{ fontSize: 16 }}>{market?.last_run || '—'}</div>
            <div className="sub">
              {training
                ? `best val_loss ${fmt(training.best_val_loss, 3)}・${training.status === 'training' ? '訓練進行中' : 'checkpoint 已產出'}`
                : 'checkpoint: v6_best.pt'}
            </div>
          </div>
        </>}
      </div>

      {/* 訓練學習曲線（真實 training_status.json） */}
      {(loading || icHist.length > 0) && (
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>📈</span> 訓練學習曲線</div>
            {training && (
              <span className="badge badge-blue">
                {icHist.length} epochs · best val_ic {fmt(training.best_val_ic, 4, true)} @ep{training.best_ic_epoch}
              </span>
            )}
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={200} /> : (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={icHist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <ReferenceLine y={0.05} stroke="rgba(0,255,136,0.4)" strokeDasharray="5 5"
                    label={{ value: 'IC=0.05目標', fill: 'var(--positive)', fontSize: 10 }} />
                  <Line type="monotone" dataKey="train_loss" stroke="var(--accent-red)"   strokeWidth={1.5} dot={false} name="Train Loss" />
                  <Line type="monotone" dataKey="val_loss"   stroke="var(--accent-amber)" strokeWidth={1.5} dot={false} name="Val Loss" />
                  <Line type="monotone" dataKey="val_ic"     stroke="var(--accent-blue)"  strokeWidth={2}   dot={false} name="Val IC" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      {/* Scale Gate + 線上 IC 時序 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>⚖️</span> Scale Gate（Short / Mid / Long）</div>
            {scaleGates.length > 0 && (
              <span className="badge badge-neutral" style={{ fontSize: 10 }}>
                最新 S {fmt(scaleGates.at(-1)?.short, 3)} / M {fmt(scaleGates.at(-1)?.mid, 3)} / L {fmt(scaleGates.at(-1)?.long, 3)}
              </span>
            )}
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={180} /> : scaleGates.length === 0 ? (
              <div style={{ textAlign: 'center', padding: 30, fontSize: 12, color: 'var(--text-muted)' }}>
                尚無 scale_gate 紀錄（V6.2 訓練開始後顯示）
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={scaleGates}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine y={0.333} stroke="rgba(160,160,160,0.35)" strokeDasharray="4 4"
                    label={{ value: '均衡 0.33', fill: 'var(--text-muted)', fontSize: 9 }} />
                  <Line type="monotone" dataKey="short" stroke="var(--accent-red)"   strokeWidth={2} dot={false} name="Short(20d)" />
                  <Line type="monotone" dataKey="mid"   stroke="var(--accent-blue)"  strokeWidth={2} dot={false} name="Mid(60d)" />
                  <Line type="monotone" dataKey="long"  stroke="var(--accent-amber)" strokeWidth={2} dot={false} name="Long(252d)" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>📡</span> 線上 IC 時序（5d）</div>
            {perf?.online_period && (
              <span className="badge badge-neutral" style={{ fontSize: 10 }}>{perf.online_period}</span>
            )}
          </div>
          <div className="panel-body">
            {loading ? <SkeletonBlock height={180} /> : onlineIC.length === 0 ? (
              <div style={{ textAlign: 'center', padding: 30, fontSize: 12, color: 'var(--text-muted)' }}>
                尚無線上 IC 資料（每日推論累積 5 天後產生）
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={onlineIC}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
                  <XAxis dataKey="pred_date" tick={{ fontSize: 9, fill: 'var(--text-muted)' }}
                    interval={Math.max(0, Math.floor(onlineIC.length / 8) - 1)}
                    tickFormatter={d => d?.slice(5)} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={0} stroke="rgba(160,160,160,0.5)" />
                  <ReferenceLine y={0.05} stroke="rgba(0,255,136,0.4)" strokeDasharray="4 4" />
                  <Bar dataKey="ic" name="IC(5d)" radius={[3, 3, 0, 0]} fillOpacity={0.85}>
                    {onlineIC.map((p, i) => (
                      <Cell key={i} fill={p.ic >= 0 ? 'var(--accent-blue)' : 'var(--accent-red)'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* 模型架構摘要（由 training_status.json 的 config 快照動態帶出） */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title"><span>🏗️</span> 模型架構摘要</div>
          {cfg.gpu && <span className="badge badge-neutral" style={{ fontSize: 10 }}>訓練 GPU：{cfg.gpu}</span>}
        </div>
        <div className="panel-body">
          {!training ? (
            <div style={{ textAlign: 'center', padding: 20, fontSize: 12, color: 'var(--text-muted)' }}>
              架構資訊將於下次訓練時自動記錄（config 快照）
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              {[
                { label: '骨幹架構', value: 'Multi-Scale Mamba + GATv2',
                  desc: `分支層數 ${(cfg.multi_scale_layers || []).join('/') || '—'}・padding mask ${cfg.use_padding_mask ? 'ON' : 'OFF'}` },
                { label: '輸入特徵', value: cfg.input_dim ? `${cfg.input_dim} 維 × ${cfg.seq_len || '—'} 天` : '—',
                  desc: '價量 + 法人籌碼 + 基本面 + 宏觀' },
                { label: '模型維度', value: cfg.d_model ? `d_model ${cfg.d_model} / d_state ${cfg.d_state}` : '—',
                  desc: '預測目標 5d / 20d / 60d Alpha' },
                { label: '參數量', value: cfg.n_parameters ? `${(cfg.n_parameters / 1e6).toFixed(2)}M` : '訓練完成後記錄',
                  desc: cfg.n_parameters ? `${cfg.n_parameters.toLocaleString()} trainable params` : '' },
                { label: '訓練資料', value: cfg.train_range ? `${cfg.train_range[0]?.slice(0, 7)} ~ ${cfg.train_range[1]?.slice(0, 7)}` : '—',
                  desc: cfg.train_days ? `${cfg.train_days} 交易日・${cfg.n_stocks?.toLocaleString() || '—'} 支股票` : '' },
                { label: '驗證資料', value: cfg.val_range ? `${cfg.val_range[0]?.slice(0, 7)} ~ ${cfg.val_range[1]?.slice(0, 7)}` : '—',
                  desc: cfg.val_days ? `${cfg.val_days} 交易日・early stop patience ${training.early_stop_patience ?? '—'}` : '' },
              ].map(m => (
                <div key={m.label} style={{ padding: '10px 14px', background: 'var(--bg-panel-2)', borderRadius: 8 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{m.label}</div>
                  <div className="mono" style={{ fontSize: 14, color: 'var(--text-primary)' }}>{m.value}</div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{m.desc}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
