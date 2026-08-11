import React from 'react';
import { VERSION_STATES } from '../versions';

/**
 * 版本狀態徽章。三種狀態：上線中 / 前一版 / 規劃中。
 * 版本字串本身在 src/versions.js，不要在這裡或任何頁面另外寫死。
 */
export default function VersionBadge({ state = 'live', showVersion = true, style }) {
  const s = VERSION_STATES[state] ?? VERSION_STATES.planned;
  const text = showVersion && s.version ? `${s.version} · ${s.label}` : s.label;

  return (
    <span className={`badge ${s.cls}`} title={s.hint} style={style}>
      <span
        aria-hidden="true"
        style={{
          width: 6, height: 6, borderRadius: '50%',
          background: 'currentColor', flexShrink: 0,
        }}
      />
      {text}
    </span>
  );
}
