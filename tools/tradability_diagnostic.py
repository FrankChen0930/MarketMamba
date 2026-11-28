"""Exposed-history diagnostics only. No fitting, acquisition or trading authority.

Reuses Step3's fully validated loader and its exact mature-support seed ranks.
Missing amount/reference-price evidence blocks liquidity/tradable portfolios;
OHLCV price*volume is deliberately not promoted to actual turnover amount.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tools.step3 import SPEC_SHA, sha256, verify_file
from tools.step3_evaluation import _spearman

SOURCE = Path('/home/frank/projects/MarketMamba')
EVIDENCE = 'EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED'


def composite(keys, labels, predictions):
    """Exactly rank each seed on COMMON MATURE daily support, then average."""
    out = np.full(len(keys), np.nan)
    for rows in day_groups(keys):
        rows = rows[np.isfinite(labels[rows])]
        if len(rows):
            out[rows] = np.mean([rankdata(p[rows], method='average') / len(rows)
                                 for p in predictions], axis=0)
    return out


def day_groups(keys):
    if not len(keys):
        return []
    if np.any(np.diff(keys[:, 0]) < 0):
        raise ValueError('STOP unsorted sessions')
    return np.split(np.arange(len(keys)), np.flatnonzero(np.diff(keys[:, 0])) + 1)


def daily_ic(keys, score, labels, ndays, mask=None):
    out = np.full(ndays, np.nan)
    valid = np.isfinite(score) & np.isfinite(labels)
    if mask is not None:
        valid &= mask
    for rows in day_groups(keys):
        d = keys[rows[0], 0]
        rows = rows[valid[rows]]
        if len(rows) >= 20:
            out[d] = _spearman(score[rows], labels[rows])
    return out


def quintiles(values, stock):
    """1=smallest; deterministic stock-index ties; missing/nonpositive=0."""
    q = np.zeros(len(values), dtype=np.int8)
    rows = np.flatnonzero(np.isfinite(values) & (values > 0))
    rows = rows[np.lexsort((stock[rows], values[rows]))]
    if len(rows):
        q[rows] = np.arange(len(rows)) * 5 // len(rows) + 1
    return q


def tick_size(price):
    return float(np.select([price < 10, price < 50, price < 100,
                            price < 500, price < 1000], [.01, .05, .1, .5, 1.], default=5.))


def top_decile(score, stock):
    """Positions in top ceil(N/10), ties by ascending stock code index."""
    return np.lexsort((stock, -score))[:(len(score) + 9) // 10]


def limit_locked(open_price, previous_close, entry_date):
    """Conservative requested one-tick proxy, NOT verified executable status.

    Caller must supply admissible previous/reference close; never silently infer
    an exchange reference price across a corporate action. Unknown is blocked.
    """
    if not np.isfinite(open_price) or not np.isfinite(previous_close) or min(open_price, previous_close) <= 0:
        return True
    limit = .07 if np.datetime64(entry_date) < np.datetime64('2015-06-01') else .10
    ceiling = previous_close * (1 + limit)
    return bool(open_price >= ceiling - tick_size(ceiling) - 1e-12)


def turnover(before, after):
    """Buy/sell traded notional / NAV; before must be drifted pretrade weights.

    Includes cash initiation/termination; half (buy+sell) is one-way turnover.
    Short-cover is a buy and short-initiation a sell (diagnostic only).
    """
    delta = [after.get(k, 0.) - before.get(k, 0.) for k in set(before) | set(after)]
    if not all(np.isfinite(v) for v in delta):
        raise ValueError('STOP unknown weights')
    return sum(max(v, 0.) for v in delta), sum(max(-v, 0.) for v in delta)


def trading_cost(buy, sell, slippage=0.):
    if not all(np.isfinite(v) and v >= 0 for v in (buy, sell, slippage)):
        raise ValueError('STOP invalid cost inputs')
    return buy * (.001425 + slippage) + sell * (.001425 + .003 + slippage)


def _mean(x):
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def _table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + ''.join(
        '| ' + ' | '.join('N/A' if v is None else f'{v:.10f}' if isinstance(v, float) else str(v) for v in row) + ' |\n' for row in rows)


def run(bundle_dir, private_out, report):
    from tools.run_step3_evaluation import load_inputs

    if shutil.disk_usage('/mnt/d').free < 60 * 1024**3:
        raise ValueError('STOP /mnt/d free <60GiB')
    root = (SOURCE / '.artifacts/tradability').resolve()
    private_out = private_out.resolve()
    if not private_out.is_relative_to(root) or private_out == root or private_out.exists():
        raise ValueError('STOP use fresh run namespace under .artifacts/tradability')
    print('Validating frozen Step3 inputs/72 COMPLETE outputs', flush=True)
    inputs = load_inputs(bundle_dir, SOURCE / '.artifacts/step3/human-source-gate-20261126.json')
    keys = inputs['label_keys']; y = inputs['labels']; days = inputs['calendar']; bundle = inputs['bundle']
    scores = {a: composite(keys, y, [inputs['predictions'][a, s] for s in (17, 29, 43)]) for a in 'AB'}
    baseline = {a: daily_ic(keys, scores[a], y, len(days)) for a in 'AB'}
    year = days.astype('datetime64[Y]').astype(int) + 1970
    periods = [('2015-2020', 2015, 2020), ('2021-2026', 2021, 2026)] + [(str(v), v, v) for v in range(2015, 2027)]
    reference_path = Path(__file__).resolve().parents[1] / 'reports/step3_results_20261126.json'
    reference = json.loads(reference_path.read_text())
    reproduction = []
    for name, lo, hi in periods[:2]:
        mask = (year >= lo) & (year <= hi)
        for a in 'AB':
            actual = _mean(baseline[a][mask]); expected = reference['periods'][name]['mean_' + a.lower()]
            error = abs(actual - expected)
            if error > 1e-9:
                raise ValueError(f'STOP IC mismatch {name} {a}: {actual} vs {expected}')
            reproduction.append([name, a, expected, actual, error])
    print('IC reproduction passed <=1e-9; reading source schemas', flush=True)
    price_path = Path(bundle['files']['prices']['path'])
    price_manifest = json.loads(Path(bundle['files']['price_manifest']['path']).read_text())
    cap_path = price_path.with_name('market_value_raw.parquet')
    cap_identity = price_manifest['files']['market_value_raw.parquet']
    verify_file(cap_path, cap_identity['sha256'], cap_identity['bytes'])
    import pyarrow.parquet as pq
    schemas = {'prices': pq.read_schema(price_path).names, 'market_value': pq.read_schema(cap_path).names}
    # This frozen source has no turnover amount or official reference close.
    # Stop dependent diagnostics rather than synthesize those fields.
    if set(schemas['prices']) != {'Date', 'stock_id', 'Open', 'High', 'Low', 'Close', 'Volume'}:
        raise ValueError('STOP source schema changed; review amount/reference fields explicitly')
    ids = np.load(bundle['files']['membership']['path'], mmap_mode='r')
    stock = np.load(bundle['files']['stock_index']['path'], mmap_mode='r')
    _, first = np.unique(stock, return_index=True)
    codes = np.asarray(ids[first])
    for start in range(0, len(ids), 100000):
        end = min(start + 100000, len(ids))
        if not np.array_equal(codes[stock[start:end]], ids[start:end]):
            raise ValueError('STOP stock code mapping')
    cap = pd.read_parquet(cap_path)
    cap['Date'] = pd.to_datetime(cap['Date'])
    if cap.duplicated(['Date', 'stock_id']).any():
        raise ValueError('STOP duplicate cap keys')
    lookup = cap.set_index(['Date', 'stock_id'])['market_value']
    row_index = pd.MultiIndex.from_arrays([pd.to_datetime(days[keys[:, 0]]), codes[keys[:, 1]]], names=['Date', 'stock_id'])
    values = lookup.reindex(row_index).to_numpy(dtype=float)
    q = np.zeros(len(keys), dtype=np.int8)
    # Stratify on ALL original inference keys, before mature-label filtering.
    for rows in day_groups(keys):
        q[rows] = quintiles(values[rows], keys[rows, 1])
    stratified = {(a, g): daily_ic(keys, scores[a], y, len(days), q == g) for a in 'AB' for g in range(1, 6)}
    # Concentration uses inference-only rankings, without future label selection.
    inference_scores = {a: composite(keys, np.zeros(len(y)), [inputs['predictions'][a, s] for s in (17, 29, 43)]) for a in 'AB'}
    concentration = {a: np.full((len(days), 6), np.nan) for a in 'AB'}
    coverage = np.full((len(days), 2), np.nan)
    for rows in day_groups(keys):
        d = keys[rows[0], 0]
        coverage[d] = [len(rows), np.count_nonzero(q[rows]) / len(rows)]
        for a in 'AB':
            selected = rows[top_decile(inference_scores[a][rows], keys[rows, 1])]
            concentration[a][d] = np.bincount(q[selected], minlength=6) / len(selected)
    ic_rows = []; conc_rows = []; support_rows = []; stratum_support = []
    for name, lo, hi in periods:
        mask = (year >= lo) & (year <= hi)
        support_rows.append([name, int(np.isfinite(baseline['B'][mask]).sum()), _mean(coverage[mask, 0]), _mean(coverage[mask, 1])])
        for a in 'BA':
            ic_rows.append([name, a, _mean(baseline[a][mask])] + [_mean(stratified[a, g][mask]) for g in range(1, 6)])
            stratum_support.append([name, a] + [int(np.isfinite(stratified[a, g][mask]).sum()) for g in range(1, 6)])
            conc_rows.append([name, a] + [_mean(concentration[a][mask, g]) for g in range(1, 6)] + [_mean(concentration[a][mask, 0])])
    private_out.mkdir(parents=True)
    np.savez_compressed(private_out / 'daily_aggregates.npz', days=days, **{'baseline_' + a: baseline[a] for a in 'AB'},
                        **{f'cap_ic_{a}_{g}': v for (a, g), v in stratified.items()},
                        **{'cap_concentration_' + a: concentration[a] for a in 'AB'}, coverage=coverage)
    manifest = dict(evidence_class=EVIDENCE, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, runtime='local CPU; CUDA unused',
                                     MARKETMAMBA_DATA_ROOT=os.environ.get('MARKETMAMBA_DATA_ROOT'), Data_realpath=str(Path('Data').resolve())),
                    bundle_sha256=inputs['gate']['bundle']['sha256'], spec_sha256=SPEC_SHA,
                    human_source_acceptance=inputs['acceptance'],
                    fold_support=inputs['folds'], seeds=[17, 29, 43],
                    model_manifest_sha256={m['job']: m['sha256'] for m in inputs['manifests']},
                    code_sha256=sha256(__file__), reference_sha256=sha256(reference_path), source_schemas=schemas,
                    cap=dict(path=str(cap_path), **cap_identity), price=bundle['files']['prices'],
                    reproduction=reproduction, skipped=['liquidity IC/concentration: no actual amount', 'tradable universe at 10m/50m: no amount/reference close',
                                                       'all 20-offset portfolios and realized costs: no qualified universe'],
                    outputs={'daily_aggregates.npz': sha256(private_out / 'daily_aggregates.npz')})
    (private_out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    text = '# 可交易性歷史診斷（2026-11-28）\n\n' + EVIDENCE + '\n\n已曝光歷史，只作診斷，不作確認、不新增擬合、不改凍結規格或前瞻家族；完成不等於 Human acceptance。\n\n'
    text += '## IC 重現 gate\n\n三 seed（17/29/43）在每日共同成熟標籤支援內各取平均秩／股數後平均，再與標籤作 Spearman；每日至少20股、兩側非恆定。完整原鍵映射到 frozen full calendar，使用第三步原 loader 驗證72個 COMPLETE 輸出、凍結標籤及身分。所有四格誤差≤1e-9，通過後才做分層。\n\n'
    text += _table(['期間', '臂', '原始 IC', '重現 IC', '絕對誤差'], reproduction)
    text += '\n## 市值分層 IC（診斷1）\n\n同價格快照 manifest 的 market_value_raw.parquet，非第三步新增特徵。以 t 日實際列值，不 forward fill；Q1最小、Q5最大，等值以股票代碼順序決定。每日先在原 inference universe 分五等分，再在各層成熟標籤支援計IC；分層沿用原共同成熟支援合成分數，不重新選 seed。各層不足20股記NaN；各期為有效日IC等權平均。市值單位不影響分層，未宣稱官方／strict PIT。\n\n'
    text += _table(['期間', '臂', '原宇宙 IC', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'], ic_rows)
    text += '\n各層有效IC日數（未成熟／不足20股／恆定不記零）：\n\n' + _table(['期間', '臂', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'], stratum_support)
    text += '\n## 前10%市值集中度（診斷5）\n\n每日全體 inference keys 各 seed 秩平均，降序前 ceil(10%×N)，等值以股票代碼排序，不用未來成熟標籤挑選名單。表為每日占比等權平均；缺市值獨立列示，不重分配。\n\n'
    text += _table(['期間', '臂', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', '市值未知'], conc_rows)
    text += '\n## 支援\n\n' + _table(['期間', 'B有效IC日', '每日平均原宇宙股數', '市值可分層比例'], support_rows)
    text += '\n## 預指定未完成項與 STOP\n\n|診斷|狀態／原因|\n|---|---|\n|1 成交金額五等分IC|SKIPPED：第三步價格快照及原始 processed_v6 僅 Date/stock_id/OHLCV，無實際成交金額；未以 close×volume 猜測。|\n|2 可交易宇宙≥1,000萬及≥5,000萬，含IC與保留率|SKIPPED：成交金額與官方前收／公司行動調整後漲停參考價皆缺；不把未知當可交易，不報0%或替代保留率。|\n|3 前10%／前30／多空、全宇宙等權，20偏移，分期逐年績效|NOT_RUN：無合格可交易宇宙。年化報酬／超額／波動／Sharpe／最大回撤／換手均N/A；未實作完整投組引擎。|\n|4 毛／淨報酬、每邊0／0.1%／0.3%滑價、損益兩平成本|NOT_RUN：依賴3；只有可執行且經合成測試的成本／換手純函式，不能當已完成投組結果。|\n|5 流動性集中度|SKIPPED：無成交金額；市值部分如上。|\n\n'
    text += '成本純函式：買進notional×(0.001425+s)，賣出notional×(0.001425+0.003+s)，不打折；換手輸入需先按價格漂移，計實際買／賣notional，單邊換手=(買+賣)/2。漲停純函式依進場日2015-06-01切7%／10%，以理論上限價所在級距一檔作保守容差；未知一律阻擋。這些函式僅合成驗證，不構成來源或成交證據。\n\n'
    text += '## 限制與解讀\n\n市值分層可描述高IC分布，不能量化真實可獲利部分；沒有可交易宇宙與投組結果，不能推算看得到但賺不到的百分比。原標籤 close(t+20)/open(t+1)−1 未含股利，高殖利率股存在偏差；無借券成本／台股放空受限；等權假設與成交量容量未建模。原價公司行動、PIT、研究成員選樣／存活者偏差、觀測日曆、2026未成熟尾端等第三步限制全保留；t+1日總量>0亦不能證明開盤可成交。未讀 blind、未取得新資料、未改正式Data／V6/models／排程／production。\n\n'
    text += '## Artifact identity／重播\n\n既有 tools/reports/tests 結構可承載本案，未建立新頂層樹；新的私有 tradability namespace 用於隔離本診斷，避免覆寫第三步。\n\n'
    text += f"Source HEAD `{manifest['source_commit']}`；bundle `{manifest['bundle_sha256']}`；價格 `{bundle['files']['prices']['sha256']}`；市值 `{cap_identity['sha256']}`。私有彙總／環境／逐檔hash：`{private_out}`（無逐股公開輸出）。\n\n"
    text += '重播：`python -m tools.tradability_diagnostic --private-out /home/frank/projects/MarketMamba/.artifacts/tradability/<fresh-run>`；每次拒絕覆寫已存在namespace，先檢查/mnt/d≥60GiB。\n'
    text += f'\n來源commit `{manifest["source_commit"]}`；bundle SHA-256 `{manifest["bundle_sha256"]}`；spec SHA-256 `{SPEC_SHA}`。價格SHA-256 `{manifest["price"]["sha256"]}`；市值SHA-256 `{cap_identity["sha256"]}`。\n\n'
    text += f'私有輸出 `{private_out}`：manifest.json及daily_aggregates.npz（只含每日彙總、不含逐股）；後者SHA-256 `{manifest["outputs"]["daily_aggregates.npz"]}`。完整輸入／程式／環境身分見manifest。\n'
    report.write_text(text)
    print(json.dumps(dict(reproduction=reproduction, report=str(report), private_out=str(private_out)), ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-dir', type=Path, default=SOURCE / '.artifacts/step3/bundle-72')
    p.add_argument('--private-out', type=Path, required=True)
    p.add_argument('--report', type=Path, default=Path('reports/tradability_diagnostic_20261128.md'))
    args = p.parse_args()
    run(args.bundle_dir, args.private_out, args.report)


if __name__ == '__main__':
    main()
