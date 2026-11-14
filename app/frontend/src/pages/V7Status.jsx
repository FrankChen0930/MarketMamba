import { useCallback, useEffect, useState } from 'react';

import { getV7Status } from '../api/v7';
import RunStatusSummary from '../components/RunStatusSummary';

const INITIAL = {
  state: 'not_ready',
  publishAllowed: null,
  dataId: null,
  generatedAt: null,
  message: '正在讀取 V7 資料健康摘要…',
  counts: { entries: null, affectedDates: null, affectedStocks: null },
};

const display = (value) => value ?? '—';

export default function V7Status() {
  const [status, setStatus] = useState(INITIAL);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setStatus(await getV7Status());
    } catch {
      setStatus({
        ...INITIAL,
        state: 'error',
        message: '目前無法連線到 V7 狀態服務，沒有沿用舊資料。',
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    getV7Status()
      .then((nextStatus) => {
        if (active) setStatus(nextStatus);
      })
      .catch(() => {
        if (active) {
          setStatus({
            ...INITIAL,
            state: 'error',
            message: '目前無法連線到 V7 狀態服務，沒有沿用舊資料。',
          });
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const decision = status.publishAllowed === null
    ? '—'
    : status.publishAllowed ? '允許發布資料摘要' : '禁止發布';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      <header className="page-header" style={{ minWidth: 0 }}>
        <div style={{ minWidth: 0 }}>
          <h1 className="page-title">V7 系統狀態</h1>
          <p className="page-subtitle" style={{ overflowWrap: 'anywhere' }}>
            這裡只顯示資料健康與發布條件，不代表 V7 模型已完成、已推論或已上線。
          </p>
        </div>
        <button className="btn btn-ghost" type="button" onClick={load} disabled={loading}>
          {loading ? '讀取中…' : '重新整理'}
        </button>
      </header>

      <RunStatusSummary
        state={status.state}
        title="V7 資料健康"
        updatedAt={status.generatedAt}
        message={status.message}
        counts={status.counts}
      />

      <section className="panel" aria-labelledby="v7-publication-heading">
        <div className="panel-header">
          <h2 className="panel-title" id="v7-publication-heading">發布依據</h2>
        </div>
        <div className="panel-body">
          <dl style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(min(180px, 100%), 1fr))',
            gap: 'var(--space-4)',
          }}>
            <div style={{ minWidth: 0 }}>
              <dt className="text-muted">資料 ID</dt>
              <dd className="mono" style={{ marginTop: 'var(--space-1)', overflowWrap: 'anywhere' }}>{display(status.dataId)}</dd>
            </div>
            <div style={{ minWidth: 0 }}>
              <dt className="text-muted">摘要產生時間</dt>
              <dd className="mono" style={{ marginTop: 'var(--space-1)', overflowWrap: 'anywhere' }}>{display(status.generatedAt)}</dd>
            </div>
            <div style={{ minWidth: 0 }}>
              <dt className="text-muted">發布決策</dt>
              <dd style={{ marginTop: 'var(--space-1)' }}>{decision}</dd>
            </div>
          </dl>
        </div>
      </section>
    </div>
  );
}
