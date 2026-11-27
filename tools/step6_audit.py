"""Label/price-blind H-0 Step6 structural and latest-revenue support audit.

Only git blobs, observation metadata, Step3 calendar/cross-source pairs and
monthly revenue are read. No Step3 admission/prepare driver is called (those
also inspect prices or targets). All row-level outputs stay private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from tools import step3
from tools.step3_prepare import admitted_events
from tools.step6_features import FEATURES, customer_features, eligible_edges, revenue_snapshot, select_version, utc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/frank/projects/MarketMamba')
REL = 'v1/tw/relations/edges.json'
SPEC = 'tasks/spec-step6-customer-momentum-h20-20261128.md'
SPEC_SHA = 'd567d6cd493cfc716431e6003ee9e8bed6fbf0e2f8fd32032d52de3672fb9639'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def load_catalog(repo, observation_path):
    """Restore all change commits plus HEAD; verify observations without pull.

    Hash verification here is integrity evidence only. It does not create a
    validated_at semantic admission timestamp in the external ledger.
    """
    log = git(repo, 'log', '--reverse', '--topo-order', '--format=%H', 'HEAD', '--', REL).decode().splitlines()
    head = git(repo, 'rev-parse', 'HEAD').decode().strip()
    versions = []
    for order, commit in enumerate(log + [head]):
        blob = git(repo, 'show', f'{commit}:{REL}')
        edges = json.loads(blob)
        if not isinstance(edges, list) or not all(isinstance(e, dict) for e in edges):
            raise ValueError('STOP relation schema')
        versions.append(dict(commit=commit, committed_at=git(repo, 'show', '-s', '--format=%cI', commit).decode().strip(),
                             sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob), order=order,
                             edges=edges, latest=order == len(log)))
    observations = []
    for line in Path(observation_path).read_text().splitlines():
        o = json.loads(line)
        if o.get('source') != 'shioaji-data':
            continue
        record = dict(o, valid=False, ancestors=[])
        try:
            if o.get('error'):
                raise ValueError('observation error')
            utc(o['observed_at'])
            blob = git(repo, 'show', f"{o['head']}:{REL}")
            if hashlib.sha256(blob).hexdigest() != o['files'][REL]:
                raise ValueError('observation blob hash')
            committed = git(repo, 'show', '-s', '--format=%cI', o['head']).decode().strip()
            if utc(committed) != utc(o['head_committed_at']):
                raise ValueError('HEAD commit timestamp')
            record.update(valid=True, ancestors=git(repo, 'rev-list', o['head']).decode().splitlines())
            # A ledger field alone cannot invent semantic validation. This
            # loader only provides the integrity level supported by this log.
            record.pop('validated_at', None)
        except (KeyError, ValueError, subprocess.CalledProcessError) as exc:
            record['integrity_error'] = type(exc).__name__
        observations.append(record)
    for v in versions:
        matches = [utc(o['observed_at']) for o in observations if o['valid'] and v['commit'] in o['ancestors']]
        v['first_observed_at'] = min(matches).isoformat() if matches else None
        v['observed_visible_at'] = max(utc(v['committed_at']), min(matches)).isoformat() if matches else None
    return versions, observations


def support_gate(suppliers, keys, stats, source_qualified):
    return 'PASS' if (source_qualified and suppliers >= 50 and keys >= 1000
                      and len(stats) == 3 and all(s['variable'] for s in stats)) else 'STOP'


def summarize_version(edges, known_codes, revenue):
    """One descriptive snapshot, never historical feature construction."""
    relations = eligible_edges(edges, known_codes)
    suppliers = sorted({e['src'][3:] for e in edges if e.get('type') == 'supplier'
                        and isinstance(e.get('src'), str) and e['src'].startswith('tw.')})
    rows = [dict(supplier=s, **customer_features(s, relations, revenue)) for s in suppliers]
    a = np.array([[r[n] for n in FEATURES] for r in rows], dtype=float).reshape(-1, 3)
    stats = []
    for j, name in enumerate(FEATURES):
        f = a[np.isfinite(a[:, j]), j]
        stats.append(dict(feature=name, rows=len(a), finite=int(len(f)),
                          finite_ratio=len(f)/len(a) if len(a) else None,
                          distinct=int(len(np.unique(f))), variable=bool(len(np.unique(f)) >= 2)))
    finite = np.isfinite(a[:, :2]).all(axis=1)
    effective = int(finite.sum())
    degree = Counter(len(v) for v in relations['customers'].values())
    missing = Counter()
    for r in rows:
        missing.update(r['missing_reasons'])
    shares = [r['max_customer_share'] for r in rows if r['max_customer_share'] is not None]
    valid_customers = {c for r in rows for c in r['valid_customers']}
    months = Counter(revenue[c].get('period', 'UNKNOWN') for c in valid_customers)
    ages = Counter(revenue[c].get('age_months', 'UNKNOWN') for c in valid_customers)
    session_ages = Counter(revenue[c].get('age_sessions', 'UNKNOWN') for c in valid_customers)
    summary = dict(eligible_edges=sum(len(v) for v in relations['customers'].values()),
                   suppliers=len(relations['customers']), structural_ge2_suppliers=sum(n for k, n in degree.items() if k >= 2),
                   effective_suppliers=effective, effective_company_session_keys=effective,
                   valid_revenue_companies=len(valid_customers),
                   unknown_suppliers=len(relations['unknown_suppliers']), features=stats,
                   max_customer_share=max(shares) if shares else None,
                   max_customer_share_ge2=max((r['max_customer_share'] for r in rows if r['reason'] == 'OK'), default=None),
                   customer_hhi_max=max((r['max_customer_share'] for r in rows if r['reason'] == 'OK'), default=None),
                   customer_revenue_months=dict(sorted(months.items())),
                   customer_revenue_age_months=dict(sorted(ages.items(), key=lambda v: str(v[0]))),
                   customer_revenue_age_sessions=dict(sorted(session_ages.items(), key=lambda v: str(v[0]))),
                   missing_reasons=dict(missing), excluded=relations['excluded'],
                   snapshot_support_status=support_gate(effective, effective, stats, True),
                   status=support_gate(effective, effective, stats, False), historical_folds='NOT_RUN')
    return summary, rows, relations['dispositions']


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def input_identity(path):
    path = Path(path)
    return dict(path=str(path), realpath=str(path.resolve()), sha256=step3.sha256(path), bytes=path.stat().st_size)


def audit(out, report):
    step3.verify_file(ROOT/SPEC, SPEC_SHA)
    out = Path(out).absolute()
    private = SOURCE/'.artifacts/step6'
    if not out.is_relative_to(private) or out == private or out.resolve() != out:
        raise ValueError('STOP private output namespace/symlink')
    # Independent run namespace; never replace prior evidence or shared files.
    out.mkdir(parents=True, exist_ok=False)
    repo = SOURCE/'.artifacts/external/shioaji-data'
    observation_path = SOURCE/'.artifacts/external/observations.jsonl'
    versions, observations = load_catalog(repo, observation_path)
    decision_path = SOURCE/'tasks/decision-step6-freeze-20261128.md'
    step3.verify_file(decision_path, step3.sha256(ROOT/'tasks/decision-step6-freeze-20261128.md'))
    bpath = SOURCE/'.artifacts/step3/bundle-72/bundle.json'
    bundle = json.loads(bpath.read_text())
    inputs = dict(bundle=input_identity(bpath), observations=input_identity(observation_path))
    # Explicit allowlist: never follow bundle jobs, labels, prices or matrices.
    paths = {}
    for key in ('calendar', 'cross_source_pairs', 'source_manifest'):
        ref = bundle['files'][key]
        path = Path(ref['path'])
        if not path.is_relative_to(SOURCE/'.artifacts') or '.worktrees' in path.parts:
            raise ValueError('STOP noncanonical input path')
        step3.verify_file(path, ref['sha256'], ref['bytes'])
        paths[key] = path
        inputs[key] = input_identity(path)
    raw_path = SOURCE/'.artifacts/tej/pit/monthly_revenue.parquet'
    sm = json.loads(paths['source_manifest'].read_text())
    info = sm['outputs']['monthly_revenue']
    step3.verify_file(raw_path, info['sha256'], info['bytes'])
    inputs['monthly_revenue'] = input_identity(raw_path)
    days = pd.DatetimeIndex(np.load(paths['calendar'], allow_pickle=False))
    raw = pd.read_parquet(raw_path, columns=['stock_id','period','available_date','單月營收(千元)','currency','input_row_number'])
    pairs = pd.read_parquet(paths['cross_source_pairs'], columns=['input_row_number','agreement'])
    snapshot = revenue_snapshot(admitted_events(raw, pairs), days, days[-1])
    write_json(out/'latest_revenue_snapshot.json', snapshot)
    summaries = []
    previous = set()
    for v in versions:
        summary, rows, dispositions = summarize_version(v['edges'], set(raw.stock_id.astype(str)), snapshot)
        current = {(e.get('src'), e.get('dst'), e.get('type')) for e in v['edges']}
        changes = dict(added=len(current-previous), removed=len(previous-current), baseline=not summaries)
        summary.update({k: val for k, val in v.items() if k != 'edges'})
        summary['edge_changes'] = changes
        summaries.append(summary)
        write_json(out/f"version-{v['order']:02d}-rows.json", rows)
        write_json(out/f"version-{v['order']:02d}-dispositions.json", dispositions)
        previous = current
    now = pd.Timestamp.now(tz='UTC').isoformat()
    result = dict(schema='step6-support-audit-v1', artifact_class='GENERATED',
                  evidence_class='DESCRIPTIVE_LATEST_REVENUE_WITH_UPSTREAM_RELATION_VERSIONS',
                  state='RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED',
                  source_commit=git(ROOT, 'rev-parse','HEAD').decode().strip(),
                  observed_at=now, contract=dict(path=SPEC, sha256=SPEC_SHA, history='H-0', evidence='E-PM', features=FEATURES),
                  decision=input_identity(decision_path),
                  inputs=inputs, environment=dict(python=platform.python_version(), executable=sys.executable,
                    platform=platform.platform(), MARKETMAMBA_DATA_ROOT=os.environ.get('MARKETMAMBA_DATA_ROOT'),
                    data_realpath=str((SOURCE/'Data').resolve()), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
                  implementation={f'tools/{name}': step3.sha256(ROOT/'tools'/name)
                                  for name in ('step6_features.py','step6_audit.py','step3.py','step3_prepare.py')},
                  revenue_snapshot_session=str(days[-1].date()), revenue_snapshot_companies=len(snapshot),
                  source_qualified=False, status='STOP', fits=0, historical_folds={str(y): 'NOT_RUN' for y in range(2015,2027)},
                  blockers=['NO_STRICT_RELATION_ENTITY_GROUP_ADMISSION', 'NO_STRICT_REVENUE_PIT_ADMISSION',
                            'NO_VALIDATED_OBSERVATION_HEALTH_CHAIN', 'SINGLE_SNAPSHOT_NOT_1000_QUALIFIED_SESSION_KEYS'],
                  model_identity=None, split_support='H-0 NOT_RUN',
                  strict_version_selected=select_version(versions, observations, now, strict=True),
                  observations=[{k: val for k, val in o.items() if k != 'ancestors'} for o in observations], versions=summaries)
    result['outputs'] = {p.name: dict(sha256=step3.sha256(p), bytes=p.stat().st_size) for p in sorted(out.iterdir())}
    write_json(out/'audit.json', result)
    render_report(result, report)
    return result


def render_report(result, report):
    lines = ['# 第六步無標籤支援盤點 — 2026-11-28', '',
             '`RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED`。完成不是 Human acceptance。', '',
             '依凍結 H-0／E-PM；0 fits，歷史 2015–2026 各折皆 `NOT_RUN`。未讀標籤、報酬、價格或 IC。',
             '沿用既有 tools／tests／reports 與私有 `.artifacts/step6/`；逐列關係與排除紀錄僅存本機，不進 git。', '',
             f"營收以第三步函式重播首次單月值，最新可用日曆 session 為 {result['revenue_snapshot_session']}，營收可見公司 {result['revenue_snapshot_companies']}。",
             '此為各上游版本搭配同一最新營收的描述性橫斷面；不建歷史客戶特徵，不把觀測前版本認證為當時可見。',
             '日曆沿用第三步受限來源；股票代碼映射僅與營收來源交集，不代表當時上市或可交易證券認證。', '',
             '| 版本（commit 前8碼） | 合格候選邊 | 供應商 | ≥2客戶結構 | ≥2有效客戶供應商／鍵 | 有效營收公司 | 新增／刪除鍵 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for v in result['versions']:
        lines.append(f"| {v['commit'][:8]}{' HEAD' if v['latest'] else ''} | {v['eligible_edges']} | {v['suppliers']} | {v['structural_ge2_suppliers']} | {v['effective_suppliers']} | {v['valid_revenue_companies']} | {v['edge_changes']['added']}/{v['edge_changes']['removed']} |")
    lines += ['', '首列增刪以空集合為基準；其後依 `(src,dst,type)` 集合。metadata 變更另由完整 blob SHA 留鏈。', '',
              '| 版本 | 三欄有限值比例（固定順序） | 三欄不同值數 | 最大客戶占比（≥2有效） | UNKNOWN供應商 |',
              '|---|---|---|---:|---:|']
    for v in result['versions']:
        ratios = '/'.join(f"{s['finite_ratio']:.3f}" if s['finite_ratio'] is not None else 'UNKNOWN' for s in v['features'])
        distinct = '/'.join(str(s['distinct']) for s in v['features'])
        lines.append(f"| {v['commit'][:8]}{' HEAD' if v['latest'] else ''} | {ratios} | {distinct} | {v['max_customer_share_ge2']} | {v['unknown_suppliers']} |")
    last = result['versions'][-1]
    lines += ['', '三欄比例的分母為該版本出現的台股供應商，非正式研究列；有效鍵僅計一次橫斷面，不乘歷史日數。',
              '等權集中度為 1/有效客戶數，等權 HHI 相同；這是已知候選集合內的集中度，不是營收依賴占比。', '',
              '最新版本排除原因（含非 supplier）：`'+json.dumps(last['excluded'], ensure_ascii=False)+'`。',
              '最新版本營收缺值原因：`'+json.dumps(last['missing_reasons'], ensure_ascii=False)+'`。',
              '最新版本有效客戶營收年月分布：`'+json.dumps(last['customer_revenue_months'])+'`。',
              '最新版本有效客戶月齡分布：`'+json.dumps(last['customer_revenue_age_months'])+'`。', '',
              '**STOP**：50家／1,000鍵門檻、三欄逐區段可變異與來源准入須同時成立；目前只做單一描述性橫斷面，所有版本皆未通過准入。',
              '觀測僅提供可還原 HEAD／hash，沒有語義驗證完成時間與連續7日健康證據；證券映射、transaction子公司實體與完整集團排除尚未核驗。',
              '文字線索配對隔離為 UNKNOWN；已知同集團排除；不能因未見 subsidiary 邊推定其餘獨立。未知範圍的 count 為 NaN。',
              '第三步歷史營收容差不等於 strict 前瞻 PIT；本輪不啟動前瞻、擬合、資料取得或排程。', '',
              '私有 manifest 包含來源 commit、規格／輸入／程式／輸出 SHA、環境、每版本排除及缺值分類。',
              f"規格 SHA-256：`{SPEC_SHA}`。私有 audit SHA-256 請見交接；公開不列逐條第三方關係或文字。", '']
    Path(report).write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report', type=Path, default=ROOT/'reports/step6_support_audit_20261128.md')
    args = p.parse_args()
    result = audit(args.out, args.report)
    print(json.dumps(dict(status=result['status'], versions=len(result['versions']), fits=0,
                          historical_folds='NOT_RUN', latest_effective_suppliers=result['versions'][-1]['effective_suppliers'])))


if __name__ == '__main__':
    main()
