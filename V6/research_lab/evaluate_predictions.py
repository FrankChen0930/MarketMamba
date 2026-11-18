"""Re-evaluate immutable predictions on the full observed-session grid; no training."""
import json
from pathlib import Path
import time
import lightgbm
import sklearn
import scipy
import pandas as pd
from .run_experiment import sha
from .metrics import daily_metrics, summarize


def main():
    root = Path('V6/research_lab/experiments')
    output = root/'results_evaluation.json'
    report = root/'REPORT.md'
    if output.exists() or report.exists():
        raise FileExistsError('evaluation/report exists; refusing overwrite')
    started = time.monotonic()
    original = root/'results.json'
    r = json.loads(original.read_text())
    data = pd.read_parquet(r['dataset'])
    if sha(r['dataset']) != r['dataset_sha256']:
        raise ValueError('dataset changed')
    days = pd.DatetimeIndex(data.date.unique()).sort_values()
    r['initial_results_sha256'] = sha(original)
    r['evaluation_policy'] = 'full observed-session grid; missing IC covariance pairs omitted, no time compression; deciles among ex-post finite-label support'
    r['evaluation_source_hashes'] = {str(p):sha(p) for p in Path('V6/research_lab').rglob('*.py')}
    r['environment'].update(lightgbm=lightgbm.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__)
    for name,v in r['models'].items():
        path = Path(v['prediction_path'])
        if sha(path) != v['prediction_sha256']:
            raise ValueError(f'prediction changed: {path}')
        p = pd.read_parquet(path)
        assert list(p.columns) == ['code','date','fold','model','horizon','score']
        assert not p.duplicated(['code','date']).any()
        h = int(p.horizon.iloc[0])
        e = p.merge(data[['code','date',f'fwd_return_{h}d']],on=['code','date'],validate='one_to_one').rename(columns={f'fwd_return_{h}d':'target'})
        grid = days[(days >= p.date.min()) & (days <= p.date.max())]
        d = daily_metrics(e,grid)
        v['summary'] = summarize(d,h)
        v['years'] = {str(year):summarize(g,h) for year,g in d.groupby(d.date.dt.year)}
        v['fold_metrics'] = {}
        for fold,g in e.groupby('fold'):
            interval = days[(days>=g.date.min()) & (days<=g.date.max())]
            v['fold_metrics'][str(fold)] = summarize(d.loc[d.date.isin(interval)],h)
        v['label_coverage_fraction'] = float(e.target.notna().mean())
        v['prediction_date_min'],v['prediction_date_max'] = str(p.date.min().date()),str(p.date.max().date())
    r['reevaluation_seconds'] = time.monotonic()-started
    with output.open('x') as stream:
        stream.write(json.dumps(r,indent=2,allow_nan=False)+'\n')
    names = ['momentum20','reversal5','lowvol','ridge','lightgbm','gru','selective']
    lines = ['# pilot09-w2 排序研究實驗', '', '`RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED`。Yahoo 價格、150 個當期大型股／ETF 的存活者偏差 universe，沒有 fundamentals；不是投資或可成交證據。', '',
             f"HEAD `{r['git_head']}`（實作未提交）；dataset bytes SHA-256 `{r['dataset_sha256']}`；manifest SHA-256 `{r['manifest_sha256']}`。", '',
             f"CPU 完整實驗一次 {r['compute_seconds']:.2f}s，僅重評估既有 predictions {r['reevaluation_seconds']:.2f}s；2 torch/BLAS threads，無 GPU／網路／安裝。Python {r['environment']['python']}、torch {r['environment']['torch']}、LightGBM {lightgbm.__version__}、sklearn {sklearn.__version__}。", '',
             '沿用 research_lab expanding split：min_train=504、test=252、purge=20，7 個有效 folds；首個候選 fold 因成熟標籤日期支援不足被既有 helper 略去；最後不滿 252-session 的區段不評估。所有模型共同要求完整60-session歷史，沒有依未來label篩預測。', '',
             '11 個當日橫斷面 z 特徵，不跨日期 fit scaler；raw baseline 分別為20d return、負5d return、負20d volatility。Ridge alpha100；LightGBM rank-label regression，60 trees、leaves15、depth5、lr.05、min_child100、L2=10、seed17。', '',
             '神經模型為雙頭 rank-label MSE、GRU width12、selective SSM width12/state4；input-selective delta/B/C、negative diagonal A、gated skip，純 PyTorch scan，不聲稱等同正式 Mamba。seeds17/29/43；每 fold 從頭重訓，2 epochs、Adam .003、gradient clip1；固定 seed 抽最多4096 fit／1024 validation rows。內層 validation 最多64 sessions（day>=maxday-63），前隔20 sessions，選 validation BEST；尚不足充分收斂比較。兩種集成均為逐日 seed percentile rank 等權平均。', '',
             '下表日期等權；ICIR未年化，NW Bartlett lag=h-1、95% normal CI；保留完整 observed-session grid，不壓縮缺失IC日期。缺值 covariance pair 不貢獻、SE 分母為有效IC數；不對多重比較校正。spread 為 top-minus-bottom decile 的重疊 h-session報酬差，非投資組合損益；hit 為top decile正報酬比例。decile 各 floor(N/10)，平手按code遞增，top取末端。', '',
             '|模型|h|IC|ICIR|NW t|IC 95% CI|spread|hit|有效IC日|', '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for h in (20,5):
        for name in names:
            m = r['models'][f'{name}_h{h}']['summary']
            ci = m['ic_ci95']
            lines.append(f"|{name}|{h}|{m['ic_mean']:.4f}|{m['icir']:.3f}|{m['nw_t']:.2f}|[{ci[0]:.4f}, {ci[1]:.4f}]|{m['decile_spread']:.2%}|{m['top_hit']:.1%}|{m['days']}|")
    lines += ['', '20d LightGBM 的平均 IC 最高；GRU 高於簡單基準但低於 LightGBM，selective 低於 reversal/ridge。5d selective 也低於 reversal、lowvol、ridge 與 LightGBM；沒有證據支持 SSM 優越性。模型間差異未做 paired 顯著性測試，不將單模型 CI 當勝出證明；不同 metric 的排序也不一致。', '', '逐年度平均 IC（完整年度 spread/hit/CI、所有seed與fold支援見 JSON）：', '']
    for h in (20,5):
        lines += [f'**h={h}**', '', '|year|'+'|'.join(names)+'|','|---|'+'---:|'*len(names)]
        years = sorted(r['models'][f'gru_h{h}']['years'])
        for year in years:
            lines.append('|'+year+'|'+'|'.join(f"{r['models'][f'{name}_h{h}']['years'][year]['ic_mean']:.4f}" for name in names)+'|')
        lines.append('')
    lines += ['三seed個別平均 IC：', '', '|模型|h|s17|s29|s43|', '|---|---:|---:|---:|---:|']
    for name in ('gru','selective'):
        for h in (20,5):
            lines.append(f'|{name}|{h}|'+ '|'.join(f"{r['models'][f'{name}_s{s}_h{h}']['summary']['ic_mean']:.4f}" for s in (17,29,43))+'|')
    v = r['models']['lightgbm_h20']
    lines += ['',f"預測區間 {v['prediction_date_min']}～{v['prediction_date_max']}；每 model/horizon {v['prediction_rows']} predictions；20d label coverage {v['label_coverage_fraction']:.2%}。", '',
              '重要限制：先去除未知label再做 decile，是事後可評估股票子集合，不能冒充事前可成交 portfolio；預測本身仍保存未知label rows。Yahoo vintage、raw-price corporate actions、非官方 observed-union calendar、當期股票選樣、特徵完整歷史篩選與label缺漏造成選擇偏差；無官方 PIT/universe/tradability、無成本、無 fundamentals。樣本不足10股日不算指標，缺失日留在NW時間軸。', '',
              '未保存 fitted weights/checkpoints，不能續跑或提供權重級模型驗證；本次模型 identity 由 frozen input/source hashes、fold/date support、config、seed與prediction SHA限定，不宣稱 checkpoint artifact identity。原 run 的 source hashes 於結束時計算，模型/runner/metrics 當時未改；之後只修正評估／防覆寫邊界，原 source hashes 保留，evaluation_source_hashes 另列。', '',
              '原始 [results.json](results.json) 保留初始評估；[results_evaluation.json](results_evaluation.json) 是本報告的完整session-grid校正評估與各檔 hash authority。不重訓、不修改既有 predictions；不變更 corrected V7／production authority。完整測試與治理結果見本報告後續驗證紀錄。']
    with report.open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(models=len(r['models']),reevaluation_seconds=r['reevaluation_seconds'],report=str(report))))

if __name__ == '__main__':
    main()
