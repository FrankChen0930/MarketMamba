import React, { useState, useRef, useEffect } from 'react';

const METRIC_INFO = {
  // ── Signal Scanner metrics ──
  rank_stability: {
    title: '排名穩定性',
    desc: '股票在過去數天連續出現在模型 Top N 排名中。穩定出現代表模型對該股看法一致，不是單日噪音。',
    threshold: 'Top 10 連續 ≥2 天，或 Top 50 連續 ≥3 天',
  },
  high_confidence: {
    title: '模型信心度',
    desc: '基於 MC-Dropout（蒙特卡羅 Dropout）的不確定度。模型跑多次推論，結果越一致代表越有把握。',
    threshold: 'Uncertainty < 0.02 為高信心',
  },
  relative_low: {
    title: '相對低點',
    desc: 'RSI（相對強弱指標）衡量股票近期漲跌動能。低於 40 代表可能處於超賣區間，適合逢低佈局。MA20 為 20 日均線，股價低於均線代表短期回檔。',
    threshold: 'RSI < 40 或 股價 < 20 日均線',
  },
  institutional_buy: {
    title: '機構買賣超',
    desc: '外資/投信等法人的連續買入行為。大型資金的持續買進通常代表機構看好該股基本面或有策略性佈局。',
    threshold: '外資連續 ≥2 天淨買入',
  },
  sharpe: {
    title: 'Sharpe Score',
    desc: '風險調整後報酬指標。計算方式為「預期報酬 ÷ 不確定度」。數值越高代表每承受一單位風險能獲得更多報酬。',
    threshold: '> 3 為高，1-3 為中，< 1 為低',
  },
  kelly: {
    title: 'Kelly 倉位',
    desc: '根據凱利公式（Kelly Criterion）計算的最適資金配置比例。考量預期報酬和風險後，建議投入總資金的百分比。',
    threshold: '通常建議使用 Half-Kelly（一半的建議值）',
  },
  alpha: {
    title: 'Alpha 預測',
    desc: '模型預測該股票相對大盤（加權指數）的超額報酬。正值代表預期跑贏大盤，負值代表預期跑輸。',
    threshold: '以 20d Alpha 為主要參考',
  },
  uncertainty: {
    title: '不確定度',
    desc: '模型對預測結果的不確定程度。透過 MC-Dropout 多次推論取標準差。越低代表模型越有信心。',
    threshold: '< 0.02 高信心，< 0.04 中信心，≥ 0.04 低信心',
  },
  market_regime: {
    title: '大盤環境',
    desc: '以 TWII（加權指數）相對 60 日均線的位置判斷市場環境。大盤在均線之上為正常市場，之下為保守市場。保守市場時入場條件更嚴格。',
    threshold: 'TWII > MA60 → 正常（2/4），TWII < MA60 → 保守（3/4）',
  },

  // ── Technical indicators ──
  rsi: {
    title: 'RSI 相對強弱指標',
    desc: 'Relative Strength Index — 衡量股票近 14 日漲跌動能的技術指標。計算公式為 100 - 100/(1+平均漲幅/平均跌幅)。超買超賣的經典指標。',
    threshold: '> 70 超買（可能反轉下跌），< 30 超賣（可能反彈），40-60 中性區間',
  },
  macd: {
    title: 'MACD 趨動指標',
    desc: 'Moving Average Convergence Divergence — 由快線（12日EMA - 26日EMA）與慢線（9日EMA of MACD）組成。用於判斷趨勢方向與動能變化。',
    threshold: 'MACD > 0 偏多，< 0 偏空。金叉（快線上穿慢線）為買入信號，死叉為賣出信號',
  },
  kd: {
    title: 'KD 隨機指標',
    desc: 'Stochastic Oscillator — K 值為近 9 日收盤價在最高最低價間的相對位置，D 值為 K 的 3 日移動平均。反映價格在近期波動範圍中的位置。',
    threshold: 'K > 80 超買，K < 20 超賣。K 線向上穿越 D 線為金叉買入信號',
  },
  bollinger: {
    title: 'Bollinger Bands 布林通道',
    desc: '以 20 日均線為中軌，上下各加減 2 倍標準差形成通道。%B 指標表示股價在通道中的相對位置（0=下軌，1=上軌）。',
    threshold: '%B > 1 突破上軌（強勢但可能過熱），< 0 跌破下軌（弱勢但可能超賣）',
  },
  ma_alignment: {
    title: '均線排列',
    desc: '觀察短期（5/10日）、中期（20/60日）、長期（120/240日）移動平均線的排列關係。多頭排列代表趨勢向上，空頭排列代表趨勢向下。',
    threshold: '多頭排列：短均線 > 中均線 > 長均線。空頭排列反之',
  },
  atr: {
    title: 'ATR 平均真實波幅',
    desc: 'Average True Range — 衡量股票的波動程度。取最高-最低、最高-前收盤、最低-前收盤中的最大值，再取 14 日平均。數值越大代表波動越劇烈。',
    threshold: '用於設定止損距離和評估市場波動。通常以 2x ATR 作為止損參考',
  },
  obv: {
    title: 'OBV 能量潮',
    desc: 'On Balance Volume — 將成交量以正負號累加：上漲日加量，下跌日減量。OBV 趨勢與股價趨勢一致代表量價配合，背離則可能反轉。',
    threshold: 'OBV 上升 + 股價上漲 = 量價齊揚（健康多頭）。OBV 下降 + 股價上漲 = 量價背離（警告）',
  },
  bias: {
    title: '乖離率',
    desc: '股價偏離移動平均線的百分比。計算方式為 (收盤價 - MA) / MA × 100%。反映股價短期是否偏離長期趨勢太遠。',
    threshold: '20日乖離率 > +5% 代表過度偏離（可能回調），< -5% 可能超賣反彈',
  },

  // ── Market breadth ──
  ad_ratio: {
    title: '漲跌比 (A/D Ratio)',
    desc: 'Advance/Decline Ratio — 上漲家數除以下跌家數。衡量市場上漲是否為廣泛參與，還是少數權值股拉動。',
    threshold: '> 1.5 代表多數股票上漲（健康多頭），< 0.7 代表多數股票下跌',
  },
  new_high_52w: {
    title: '52 週新高',
    desc: '近一年內創新高的股票數量。大量新高代表市場整體處於強勢趨勢中，為市場廣度的重要觀察指標。',
    threshold: '新高股數遠大於新低 → 強勢市場',
  },
  above_ma20: {
    title: '站上 MA20 比例',
    desc: '全市場中股價高於 20 日均線的股票佔比。高比例代表短期多頭排列的股票多，市場整體偏多。',
    threshold: '> 60% 偏多，< 40% 偏空',
  },
  vol_vs_5ma: {
    title: '成交量比 5MA',
    desc: '今日成交量除以 5 日平均成交量。大於 1 代表量能放大（可能有主力進場或重大事件），小於 1 代表量能萎縮。',
    threshold: '> 1.5x 量能顯著放大，< 0.7x 量能明顯萎縮',
  },

  // ── Institutional / margin ──
  foreign_net: {
    title: '外資淨買超',
    desc: '外資法人在台股市場的買入金額減去賣出金額。外資是台股最大的法人，其動向對大盤有顯著影響力。',
    threshold: '連續買超通常推升指數，連續賣超則壓抑',
  },
  margin_balance: {
    title: '融資餘額',
    desc: '投資人向券商借款買股的未償還總額。反映市場散戶的槓桿使用程度。融資大幅增加可能代表散戶過度樂觀。',
    threshold: '快速增加 → 留意過熱，急速下降 → 可能為斷頭賣壓',
  },
  margin_maint: {
    title: '融資維持率',
    desc: '融資擔保品市值 / 融資金額的比率。低於 130% 時券商會發出追繳通知（斷頭），低於 120% 強制平倉。',
    threshold: '> 160% 安全，130-160% 留意，< 130% 危險（斷頭區）',
  },
  short_balance: {
    title: '融券餘額',
    desc: '投資人借入股票賣出的未回補總額。融券代表看空。融券大增可能有軋空風險（空方被迫回補推升股價）。',
    threshold: '觀察融券增減趨勢，而非絕對數值',
  },
  margin_ratio: {
    title: '資券比',
    desc: '融資餘額除以融券餘額。高倍數代表市場多方槓桿遠大於空方。極端值可能代表方向共識過度集中。',
    threshold: '偏高時留意軋空風險，偏低時留意多殺多風險',
  },

  // ── Model metrics ──
  sector_alpha: {
    title: '產業平均 Alpha',
    desc: '模型對各產業內所有股票預測的 20 日超額報酬的平均值。正值代表模型看好該產業整體表現優於大盤。',
    threshold: '用於判斷資金應該往哪些產業配置',
  },
  confidence_dist: {
    title: '信心分佈',
    desc: '模型對所有推論股票的不確定度分級。高信心（Unc < 0.02）代表模型預測一致性高，低信心代表模型不確定。',
    threshold: '高信心比例越高代表模型對今日市場判斷越明確',
  },
  vix: {
    title: 'VIX 恐慌指數',
    desc: 'CBOE 波動率指數 — 衡量 S&P 500 選擇權的隱含波動率，反映市場對未來 30 天波動的預期。俗稱「恐慌指標」。',
    threshold: '< 15 低波動（市場平靜），15-25 正常，25-35 波動偏高，> 35 恐慌',
  },

  // ── Model status metrics ──
  ic: {
    title: 'IC (Information Coefficient)',
    desc: '模型預測排名與實際報酬排名的相關係數。衡量模型區分好壞股票的能力。IC = 1 代表完美預測，IC = 0 代表隨機。',
    threshold: '> 0.05 為有效量化模型，> 0.1 為優秀',
  },
  icir: {
    title: 'ICIR (IC Information Ratio)',
    desc: 'IC 的均值除以 IC 的標準差。衡量模型預測能力的穩定性。高 ICIR 代表模型不只預測準，而且持續穩定地準。',
    threshold: '> 0.5 為穩定有效，> 1.0 為極優秀',
  },
  val_loss: {
    title: 'Validation Loss',
    desc: '模型在驗證集上的損失函數值。越低代表模型在未見過的數據上表現越好。如果 val_loss 開始上升代表過擬合。',
    threshold: '觀察趨勢比絕對值重要。val_loss 不再下降時停止訓練（early stopping）',
  },
  listnet: {
    title: 'ListNet Loss',
    desc: 'ListNet 排序損失 — 衡量模型預測排序與真實報酬排序的差距。用 cross-entropy 比較兩個排序分佈的差異。',
    threshold: '越低代表排序預測越準確',
  },
  realized_vol: {
    title: '實現波動率',
    desc: '基於歷史價格計算的年化波動率。使用過去 N 日的日報酬標準差乘以 √252 得到年化值。',
    threshold: '< 15% 低波動，15-25% 正常，> 30% 高波動',
  },
  beta: {
    title: 'Beta 係數',
    desc: '個股或市場相對於基準指數（如 S&P 500）的系統性風險衡量。Beta = 1 代表與大盤同步，> 1 代表波動更大。',
    threshold: '< 0.8 防禦型，0.8-1.2 與大盤同步，> 1.2 進攻型',
  },
  foreign_holding: {
    title: '外資持股比例',
    desc: '外資法人持有台股的總市值佔比。反映國際資金對台灣市場的參與程度和信心。',
    threshold: '通常在 40-45% 之間波動，大幅變動代表國際資金流向改變',
  },
};

export function getMetricInfo(key) {
  return METRIC_INFO[key] || { title: key, desc: '', threshold: '' };
}

export default function MetricTooltip({ metricKey, style }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const info = getMetricInfo(metricKey);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <span ref={ref} style={{ position: 'relative', display: 'inline-flex', ...style }}>
      <button
        onClick={(e) => { e.stopPropagation(); setOpen(!open); }}
        style={{
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--text-muted)', fontSize: 11, padding: '0 2px',
          lineHeight: 1, fontWeight: 700, opacity: 0.6,
          transition: 'opacity 0.15s',
        }}
        onMouseEnter={(e) => e.target.style.opacity = 1}
        onMouseLeave={(e) => e.target.style.opacity = 0.6}
        title={info.title}
      >
        ?
      </button>

      {open && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'absolute', top: '100%', left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 999, width: 280,
            background: 'var(--bg-panel)',
            border: '1px solid var(--border-bright)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-glow-blue)',
            padding: '14px 16px',
            animation: 'fadeInUp 0.15s ease forwards',
            marginTop: 6,
          }}
        >
          <div style={{
            fontSize: 13, fontWeight: 700, color: 'var(--accent-blue)',
            marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <span style={{ fontSize: 14 }}>💡</span>
            {info.title}
          </div>
          <div style={{
            fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6,
            marginBottom: 10,
          }}>
            {info.desc}
          </div>
          <div style={{
            fontSize: 11, color: 'var(--accent-amber)',
            padding: '6px 10px',
            background: 'rgba(255,165,0,0.06)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid rgba(255,165,0,0.15)',
          }}>
            ⚡ {info.threshold}
          </div>
        </div>
      )}
    </span>
  );
}
