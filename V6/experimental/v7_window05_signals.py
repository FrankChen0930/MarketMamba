"""PROPOSED exploratory formulas on caller-supplied snapshots; no data/label IO.

Session availability ledger is mandatory, never inferred from a Date label.
No formal feature or label contract is changed by these research helpers.
"""
import math
from datetime import date,datetime
from zoneinfo import ZoneInfo
import numpy as np

NAMES=('abnormal_volume','market_relative_momentum','foreign_flow_intensity','short_reversal')

def _aware(value):
    try:
        t=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        if t.tzinfo is None or t.utcoffset() is None: raise ValueError('naive timestamp')
        return t
    except (TypeError,ValueError):
        raise ValueError('timezone-aware timestamp required') from None

def candidates(frame,calendar,stock,session,decision_at,*,volume_unit='shares',flow_unit='shares'):
    """One target snapshot; O(number of rows + 21 lookups), constant rolling horizon.

    Input values must already be the proper as-of vintage at decision_at. Timestamp
    checks cannot establish provider truth, tradability or population eligibility.
    Missing calendar sessions stay missing; there is no carry, backfill or label.
    """
    days=[date.fromisoformat(str(d)).isoformat() for d in calendar]
    if not days or days!=sorted(set(days)) or session not in days:raise ValueError('invalid session calendar')
    cut=_aware(decision_at).astimezone(ZoneInfo("Asia/Taipei"))
    if cut.date().isoformat()<session:raise ValueError('decision precedes observation session')
    if volume_unit!=flow_unit or volume_unit not in {'shares','lots'}:raise ValueError('flow/volume units must agree')
    required={'stock_id','session','close','volume','foreign_net','benchmark','price_available_at','flow_available_at','benchmark_available_at'}
    if not required.issubset(frame.columns):raise ValueError('missing required source/availability fields')
    if frame.duplicated(['stock_id','session']).any():raise ValueError('duplicate stock/session')
    if not set(frame.session).issubset(days):raise ValueError('source outside supplied calendar')
    rows=frame[(frame.stock_id==stock)&(frame.session<=session)].set_index('session')
    pos=days.index(session)
    def value(offset,col,time_col,positive=False):
        if pos+offset<0 or days[pos+offset] not in rows.index:return np.nan
        row=rows.loc[days[pos+offset]]
        try:
            available=_aware(row[time_col]);v=float(row[col])
        except (ValueError,TypeError):return np.nan
        if available>cut or not math.isfinite(v) or (positive and v<=0):return np.nan
        return v
    def px(k):return value(k,'close','price_available_at',True)
    def vol(k):
        v=value(k,'volume','price_available_at');return v if v>=0 else np.nan
    v=vol(0);past=[vol(-k) for k in range(1,21)]
    mean=float(np.mean(past)) if all(math.isfinite(x) for x in past) else np.nan
    p=px(0);p1=px(-1);p20=px(-20)
    b=value(0,'benchmark','benchmark_available_at',True);b20=value(-20,'benchmark','benchmark_available_at',True)
    flow=value(0,'foreign_net','flow_available_at')
    return dict(zip(NAMES,[v/mean-1 if mean>0 else np.nan,
        p/p20-b/b20 if p20>0 and b20>0 else np.nan,
        flow/v if v>0 else np.nan,-(p/p1-1) if p1>0 else np.nan]))
