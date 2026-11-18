"""Paired IC significance over stored predictions; no training. PYTHONPATH=V6 python -m research_lab.significance_report"""
import argparse, json
from pathlib import Path
import pandas as pd
from .metrics import daily_metrics
from .significance import compare_models


def daily_ic(pred_path, data):
    p = pd.read_parquet(pred_path)
    h = int(p.horizon.iloc[0])
    e = p.merge(data[['code', 'date', f'fwd_return_{h}d']], on=['code', 'date'], validate='one_to_one')
    e = e.rename(columns={f'fwd_return_{h}d': 'target'})
    days = pd.DatetimeIndex(data.date.unique()).sort_values()
    d = daily_metrics(e, days[(days >= p.date.min()) & (days <= p.date.max())])
    return d.set_index('date').ic


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--dataset', type=Path, required=True)
    a.add_argument('--predictions', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--baseline', default='reversal5')
    a.add_argument('--models', nargs='+', default=['momentum20', 'reversal5', 'lowvol', 'ridge', 'lightgbm', 'gru', 'selective'])
    a.add_argument('--n-boot', type=int, default=2000)
    args = a.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    data = pd.read_parquet(args.dataset)
    result = {}
    for h in (5, 20):
        ic = {m: daily_ic(args.predictions/f'{m}_h{h}.parquet', data) for m in args.models}
        # common sessions across all models so the family is evaluated on identical dates
        common = pd.concat(ic, axis=1).dropna().index
        ic = {m: s.loc[common] for m, s in ic.items()}
        result[f'h{h}'] = dict(common_sessions=len(common), baseline=args.baseline,
                               comparisons=compare_models(ic, args.baseline, h, n_boot=args.n_boot))
    result['caveats'] = ['Yahoo 150-name survivorship-biased universe', 'HAC lag=h-1 on daily IC differences',
                         'Holm over models within each horizon only; not across horizons or baselines',
                         'IC significance != portfolio value']
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    for k in ('h5', 'h20'):
        print(k, 'sessions', result[k]['common_sessions'])
        for r in result[k]['comparisons']:
            print(f"  {r['model']:11s} vs {r['baseline']}: dIC {r['mean_diff']:+.4f} t {r['t_hac']:+.2f} p_holm {r['p_hac_holm']:.3f} boot_holm {r['p_boot_holm']:.3f}")


if __name__ == '__main__':
    main()
