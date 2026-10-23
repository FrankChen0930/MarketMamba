"""Read-only Window05 source audit. No dataset discovery, fetch or admission."""
from pathlib import Path
import ast
import hashlib
import json
import logging
import numpy as np
import pandas as pd

ENGINEER='V6/marketmamba/data/feature_engineer.py'
MANIFEST='research/v7/corrected-baseline-v1/feature-manifest.json'
POLICY='research/v7/corrected-baseline-v1/policy.json'

# Source-level lineage, not proof of the historical payload/vintage consumed by E5.
FAMILIES=[
 ('price', 'TWSE/TPEx eligible OHLCV; legacy cache may mix providers', 'prices_raw / immutable prepared per-stock cache', '_add_price_momentum_features',
  'Open High Low Close Volume Return_1d Return_5d Return_20d MA_20 MA_60 RSI_14 ATR_14 Volatility_20d',
  'Rolling warmup NaN; upstream missing values may already be filled; final NaN->0',
  'Adjustment/split consistency; per-stock rows are not necessarily consecutive sessions'),
 ('relative_strength','TWII macro + stock close', 'macro_raw + prices_raw','_add_rs_features','RS_5d RS_20d RS_60d',
  'Missing TWII helper defaults to0; per-stock ffill then0','Existing market-relative momentum; TWII coverage and same-session identity required'),
 ('institutional','TWSE/TPEx or FinMind; payload provenance unverified','institutional_raw','_merge_institutional','Foreign_Buy Foreign_Sell Foreign_Net Investment_Trust_Net Dealer_Net',
  'Absent source/join miss ->0','Post-close publication time unknown; unit/share-vs-lot consistency'),
 ('margin','TWSE/TPEx or FinMind; payload provenance unverified','margin_raw','_merge_margin','Margin_Purchase Margin_Repay Short_Sale Short_Cover Margin_Balance Short_Balance',
  'Absent source ->0; source function normalizes/fills missing','Publication time unknown; purchases/repayments vs stock/flow semantics'),
 ('technical','Stock OHLCV','prices_raw','_add_technical_b_features','KD_K KD_D OBV',
  'Warmup/default technical fills; final NaN->0','Cumulative OBV normalized by trailing window; split/volume units matter'),
 ('daytrade','TWSE/TPEx or FinMind','daytrade_raw','_merge_daytrade','Day_Trade_Volume',
  'Absent source default0; ratio clipped[0,1]','Name is volume but implementation consumes ratio; percent-vs-fraction input must match'),
 ('holdings','TDCC through provider; exact vintage unverified','holdings_raw','_merge_holdings','Holdings_Large_Pct Holdings_Large_Change',
  'per-stock ffill then0; change diff(5)->0','Weekly observation date is not publication date; ownership threshold and revision lineage'),
 ('lending','TWSE/TPEx or provider','securities_raw','_merge_securities','Securities_Balance',
  'Heuristic balance column; per-stock ffill then0','Heuristic column selection and after-close publication'),
 ('foreign_holdings','Exchange/FinMind','foreign_shareholding_raw','_merge_foreign_shareholding','Foreign_Holding_Pct',
  'Chosen ratio column; per-stock ffill then0','Ratio-column fallback/units and publication time unknown'),
 ('capitalization','Daily market-value provider observation','market_value_raw','_merge_market_value_feature','Market_Cap_Log',
  'Negative market value clipped0; log1p; per-stock ffill then0','Policy admits daily value by close; not independently verified per-row; no historical share-vintage proof'),
 ('macro','yfinance/FinMind/other macro cache; see per-feature detail','macro_raw and optional separate fallback tables','_merge_macro',
  'TWII_Return SPX_Return VIX TNX Gold_Return Oil_Return USD_TWD Futures_OI_Foreign Options_PC_Ratio Fear_Greed Business_Signal FED_Rate',
  'Macro defaults/ffill; macro warmup252 prior dates then NaN->0','Date-only joins; timezone/publication/revisions; macro common across stocks')]
CALC={
 'Open':'raw session open','High':'raw session high','Low':'raw session low','Close':'raw session close','Volume':'raw session volume',
 'Return_1d':'Close/lag1(Close)-1','Return_5d':'Close/lag5(Close)-1','Return_20d':'Close/lag20(Close)-1',
 'MA_20':'trailing20 mean Close,min10','MA_60':'trailing60 mean Close,min30','RSI_14':'EWM gain/loss from Close differences,com13,min14; zero loss->NaN',
 'ATR_14':'TR=max(H-L,abs(H-prevC),abs(L-prevC)); EWM span14,min7','Volatility_20d':'trailing20 std log Close/lagClose,min10',
 'RS_5d':'Return_5d - TWII5 return','RS_20d':'Return_20d - TWII20 return','RS_60d':'Close/lag60(Close)-1 - TWII60 return',
 'KD_K':'rolling9 stochastic RSV then EWM com2','KD_D':'KD_K EWM com2','OBV':'cumsum(sign(delta Close)*Volume), trailing60 zscore,min20',
 'Holdings_Large_Pct':'1-retail ratio/100, else selected large holder ratio/100; clip[0,1]',
 'Holdings_Large_Change':'Holdings_Large_Pct - lag5(Holdings_Large_Pct)',
 'Market_Cap_Log':'log1p(max(market_value,0))','Foreign_Net':'raw signed net shares; inspect buy/sell consistency',
 'Day_Trade_Volume':'provided Day_Trade_Volume ratio clipped[0,1]',
 'Futures_OI_Foreign':'foreign long open interest - foreign short open interest, sum Date',
 'Options_PC_Ratio':'put volume/call volume; zero call->default1',
 'TWII_Return':'TWII_Close pct_change1','SPX_Return':'SPX pct_change1; US_QQQ renamed SPX if present',
 'Gold_Return':'Gold pct_change1','Oil_Return':'Oil pct_change1',
 'VIX':'raw index level','TNX':'raw yield proxy; provider units require validation','USD_TWD':'raw FX quote; provider close/session convention unknown',
 'Fear_Greed':'CNN_FearGreed or separate provider value, default50',
 'Business_Signal':'TW_Biz_Signal primary Date join; fallback month-start+2 calendar months asof; default23',
 'FED_Rate':'macro primary or separate backward asof; default4'}

def source_function(name, root=Path('.')):
    """Compile only inspected pure transforms, avoiding production config/data imports."""
    tree=ast.parse((root/ENGINEER).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={'pd':pd,'np':np,'logger':logging.getLogger(__name__),'MACRO_TS_MIN_HIST':252}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),ENGINEER,'exec'),ns)
    return ns[name]

def inventory(root):
    manifest=json.loads((root/MANIFEST).read_text())
    policy=json.loads((root/POLICY).read_text())
    order=manifest['feature_order']
    if len(order)!=48 or order!=policy['features']['kept'] or len(set(order))!=48:
        raise ValueError('frozen manifest/policy mismatch')
    definitions={name:(group,provider,artifact,fn,missing,risk) for group,provider,artifact,fn,names,missing,risk in FAMILIES for name in names.split()}
    if set(definitions)!=set(order): raise ValueError('audit inventory is not exact frozen membership')
    rows=[]
    for i,name in enumerate(order):
        group,provider,artifact,fn,missing,risk=definitions[name]
        macro=group=='macro'
        conditional=group in {'price','relative_strength','technical','capitalization'} or name=='TWII_Return'
        rows.append(dict(canonical_name=name,ordinal=i,feature_group=('A' if i<15 else 'B' if i<35 else 'C' if i==35 else 'D'),
            source_family=group,source_provider=provider,source_artifact=artifact,raw_or_derived=('DERIVED' if name in CALC and not CALC[name].startswith('raw') else 'RAW_OR_PROVIDER_TRANSFORM'),
            calculation_dependencies=CALC.get(name,'Provided source column '+name),observation_session='Date label; not proof of actual availability',
            publication_evidence='UNKNOWN: no row-level timestamp ledger audited',availability_evidence='UNKNOWN: date joins do not prove cutoff compliance',retrieval_timestamp=None,
            revision_vintage_semantics='UNKNOWN per-row vintage; source cache mutable, prepared hash fixes bytes but not historical knowledge',
            missing_value_semantics=missing,scaling=('prior-date expanding zscore,min252,clip+-3; final0' if macro else 'per-Date winsor1/99pct then cross-sectional zscore; final0'),
            current_pit_admission='IN_FROZEN_48: historical simulation policy only; strict executable STOP',
            known_risk=risk,evidence_location=[MANIFEST,POLICY,ENGINEER+'::'+fn,ENGINEER+'::clean_and_scale'],
            quality_status='UNKNOWN',quality_reason='No certified raw/unscaled feature value table supplied; no matrix reconstructed or mutable operational payload read',
            audit_conclusion='CONDITIONAL' if conditional else 'UNKNOWN',
            audit_reason='Source/formula traced; valid only if row availability, vintage, adjustment and universe assumptions hold; not end-to-end VERIFIED' if conditional else 'Required historical publication/vintage provenance not established'))
    return rows

def quality(frame,names):
    """Summarize caller-supplied bounded frame. UNKNOWN never becomes zero missingness."""
    out={}
    for name in names:
        if frame is None or name not in frame:
            out[name]={'status':'UNKNOWN'};continue
        s=pd.to_numeric(frame[name],errors='coerce');finite=np.isfinite(s)
        yearly={}
        for year in sorted(pd.to_datetime(frame.Date).dt.year.unique()):
            ix=pd.to_datetime(frame.Date).dt.year==year
            yearly[str(year)]={'rows':int(ix.sum()),'finite':int(finite[ix].sum())}
        out[name]={'status':'SUPPLIED_SAMPLE_ONLY','rows':len(s),'nonfinite':int((~finite).sum()),'zero':int((s==0).sum()),'by_year':yearly,
                   'quantiles':{str(q):(None if not finite.any() else float(s[finite].quantile(q))) for q in [0.,.01,.5,.99,1.]}}
    return out

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    # Output only to explicitly chosen research namespace; inputs are versioned source.
    result={'schema':'window05-f0-v1','evidence_class':'SOURCE_INTEGRITY_AUDIT_NOT_ADMISSION','features':inventory(Path('.')),
            'source_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [ENGINEER,MANIFEST,POLICY]},'empirical_quality':quality(None,json.loads(Path(MANIFEST).read_text())['feature_order'])}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
