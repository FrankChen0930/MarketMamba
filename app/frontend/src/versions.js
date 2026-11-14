/**
 * 全站版本狀態 —— 唯一真相來源。
 *
 * 版本字串（"V6.2" / "V6.1"）刻意只寫在這一個檔案裡。
 * 過去踩過的坑：同一個數字被寫死在前端文案、後端 fallback、腳本 footer
 * 三個地方，改版之後全部過時，而且看起來還是很像正確的對照。
 * 要改版號 → 只改這裡。
 *
 * 這裡不放 React 元件（放了會讓 fast refresh 失效），畫面請用 VersionBadge。
 */
export const VERSION_STATES = {
  live: {
    version: 'V6.2',
    label: '上線中',
    cls: 'badge-live',
    hint: '目前每天實際在跑、也是正在累積實戰紀錄的版本',
  },
  legacy: {
    version: 'V6.1',
    label: '前一版',
    cls: 'badge-legacy',
    hint: '已經凍結，保留下來對照用，不再更新',
  },
  planned: {
    version: null,
    label: '規劃中',
    cls: 'badge-planned',
    hint: '設計好了，但還沒有真實資料可看',
  },
};

/** 給文案用：取版本號字串，避免各頁自己硬寫 "V6.2" */
export const versionOf = (state) => VERSION_STATES[state]?.version ?? '';
