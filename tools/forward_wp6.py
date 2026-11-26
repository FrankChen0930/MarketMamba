"""WP6 endpoint A adapter, SYNTHETIC_CPU_ONLY / UNMERGED / NOT_DEPLOYED.

Explicit s0..s252, fractional raw-price accounting; never calls the legacy
simulator. Evidence fields below are synthetic fixtures, NOT source admission.
No loaders, automatic jobs, result publication or real forward admission.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from research_lab.portfolio import transaction_cost
from research_lab.portfolio_eval import p2_primary_config
from tools.forward_wp1 import Stop, timestamp
from tools import forward_wp5 as wp5


@dataclass(frozen=True)
class Bar:
    open: float | None
    close: float | None
    listed: bool | None
    suspended: bool
    executable: bool
    source: str
    asof: str
    event: bool = False


@dataclass(frozen=True)
class Signal:
    scores: dict[str, float]
    eligible: tuple[str, ...]
    dollar_volume: dict[str, float]
    model_id: str
    asof: str
    source: str


@dataclass
class _State:
    cash: float = 1.
    q: dict = field(default_factory=dict)
    marks: dict = field(default_factory=dict)
    pending: set = field(default_factory=set)
    buffer: tuple = ()
    cost: float = 0.


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _invalid(reason):
    raise Stop('INVALID_FOR_CONFIRMATION: '+reason)


def _support(bar, day):
    if not isinstance(bar, Bar) or not bar.source or not bar.source.startswith('SYNTHETIC'):
        _invalid('missing synthetic source/status evidence')
    try:
        known = timestamp(bar.asof)
        if known > timestamp(day+'T09:00:00+08:00'):
            _invalid('status evidence after open')
    except (Stop, TypeError, ValueError) as exc:
        _invalid('missing/late status asof: '+str(exc))
    if (bar.listed is not True or type(bar.suspended) is not bool or
            type(bar.executable) is not bool or bar.event is not False):
        _invalid('unknown lifecycle/delisting/corporate event settlement')


def _signal(value, day):
    if not isinstance(value, Signal) or not value.model_id or not value.source.startswith('SYNTHETIC'):
        _invalid('missing synthetic signal identity')
    if timestamp(value.asof) != timestamp(day+'T18:00:00+08:00'):
        _invalid('signal cutoff mismatch')
    if (not isinstance(value.eligible, tuple) or len(set(value.eligible)) != len(value.eligible)
            or not all(isinstance(c,str) and c for c in value.eligible)):
        _invalid('unknown universe support')
    if not set(value.scores) <= set(value.eligible):
        _invalid('scores outside eligible universe')
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v)
           for v in value.scores.values()):
        _invalid('unknown score')
    if any(not _positive(value.dollar_volume.get(c)) for c in value.eligible):
        _invalid('unknown liquidity')


def endpoint_sessions(calendar, sessions):
    """Reject partial, shifted, duplicate, skipped or 253-return windows."""
    if len(sessions) != 253:
        raise Stop('exactly 253 calendar points s0..s252 required')
    s0 = calendar.require_session(sessions[0])
    expected = tuple(calendar.shift(s0, k) for k in range(253))
    if tuple(sessions) != expected or expected[-1] != calendar.p2_endpoint(s0):
        raise Stop('missing/shifted endpoint sessions')
    return expected


def _rebalance(state, signal, bars, day, config, benchmark):
    held = {c for c,q in state.q.items() if q > 0}
    liquid = {c for c in signal.eligible if signal.dollar_volume[c] >= config.min_dollar_volume}
    ranked = sorted((c for c in signal.scores if c in liquid), key=lambda c:(-signal.scores[c], c))
    if benchmark:
        selected = sorted(liquid)
    else:
        retained = [c for c in ranked[:config.hold_buffer_rank] if c in held]
        additions = [c for c in ranked if c not in retained][:max(0,config.top_k-len(retained))]
        selected = retained+additions
    state.buffer = tuple(selected)
    codes = sorted(held | set(selected))
    tradable = set()
    for code in codes:
        bar = bars.get(code)
        _support(bar,day)
        if bar.executable and not bar.suspended and _positive(bar.open):
            tradable.add(code)
            state.marks[code] = float(bar.open)
    frozen = held-tradable
    blocked = sorted((held | set(selected))-tradable)
    state.pending = set(blocked)
    if any(c not in state.marks for c in held): _invalid('held without open mark')
    old = {c:state.q.get(c,0.)*state.marks.get(c,0.) for c in codes}
    nav = state.cash+sum(old.values())
    frozen_value = sum(old[c] for c in frozen)
    preserve = not benchmark and len(set(selected)|frozen) > config.top_k
    # Equal weights and primary cap=1. Missing opens leave cash; never replace
    # a blocked selection using a lower-ranked future-open name.
    weights = {c:1/len(selected) for c in selected} if selected else {}

    def orders(net):
        if preserve: return {c:0. for c in codes}
        return {c: (max(0.,net-frozen_value)*weights.get(c,0.)-old[c])
                if c in tradable else 0. for c in codes}

    def fee(delta):
        return transaction_cost(sum(max(v,0.) for v in delta.values()),
            -sum(min(v,0.) for v in delta.values()), config, list(delta.values()))

    no_orders = preserve or not any(abs(v)>1e-13 for v in orders(nav).values())
    lo, hi = frozen_value, nav
    if no_orders: lo = nav
    for _ in range(0 if no_orders else 60):
        mid = (lo+hi)/2
        if mid+fee(orders(mid)) > nav: hi = mid
        else: lo = mid
    principal = max(frozen_value,lo-32*math.ulp(nav))
    delta = orders(principal)
    # Never move unlogged sub-threshold quantities or charge phantom fees.
    delta = {c:v if abs(v)>1e-13 else 0. for c,v in delta.items()}
    cost = fee(delta)
    state.cash = nav-sum(old.values())-sum(delta.values())-cost
    if state.cash < -1e-12: _invalid('negative cash')
    state.cost += cost
    trades = []
    for code, amount in delta.items():
        if amount:
            state.q[code] = (old[code]+amount)/bars[code].open
            if state.q[code] < 1e-14: state.q.pop(code)
            trades.append(dict(date=day, code=code, notional=amount,
                               price=bars[code].open, signal_model=signal.model_id))
    return trades, blocked, cost


def _close(state,bars,day):
    stale=[]
    for code,q in state.q.items():
        if q <= 0: continue
        bar=bars.get(code)
        _support(bar,day)
        if _positive(bar.close) and not bar.suspended:
            state.marks[code]=float(bar.close)
        elif bar.close is None and bar.suspended and code in state.marks:
            stale.append(code)
        else: _invalid('held close unavailable without qualified suspension')
    nav=state.cash+sum(q*state.marks[c] for c,q in state.q.items())
    if not _positive(nav): _invalid('nonfinite/nonpositive NAV')
    return nav,sorted(stale)


def _arm(sessions,market,signals,config,benchmark):
    state=_State()
    daily=[dict(at=sessions[0]+'T13:30:00+08:00',nav=1.,cash=1.,quantities={},
                cost=0.,cumulative_cost=0.,phase=0,buffer=[],pending=[],stale=[],
                blocked=[],signal=None,model_id=None)]
    trades=[]
    for k,day in enumerate(sessions[1:],1):
        signal_day=sessions[k-1]
        signal=signals[signal_day]
        blocked=[]; cost=0.; order_signal=None
        if (k-1)%20 == 0:
            filled,blocked,cost=_rebalance(state,signal,market[day],day,config,benchmark)
            trades.extend(filled); order_signal=signal_day
        nav,stale=_close(state,market[day],day)
        daily.append(dict(at=day+'T13:30:00+08:00',nav=nav,cash=state.cash,
            quantities=dict(sorted(state.q.items())),cost=cost,cumulative_cost=state.cost,
            phase=(k-1)%20,buffer=list(state.buffer),pending=sorted(state.pending),
            stale=stale,blocked=blocked,signal=order_signal,model_id=signal.model_id))
    returns=[b['nav']/a['nav']-1 for a,b in zip(daily,daily[1:])]
    return dict(daily=daily,returns=returns,trades=trades)


def run_endpoint(calendar,sessions,market,signals,*,config=None,phase='FORMAL',initial_state=None):
    """Pure paired synthetic endpoint. Never consumes a trial/resume NAV.

    Daily signals are sealed upstream; model identities may change, while this
    adapter owns one state per arm. Caller-supplied evidence is no admission.
    Returned detailed rows remain PRIVATE under .artifacts/forward if persisted.
    """
    if initial_state is not None: raise Stop('cannot inherit trial/other initial state')
    if phase not in ('FORMAL','TRIAL'): raise Stop('unknown phase')
    if calendar.identity.get('synthetic') is not True:
        raise Stop('synthetic-only adapter; real admission NOT implemented')
    sessions=endpoint_sessions(calendar,sessions)
    config=config or p2_primary_config()
    if (config.rebalance_sessions != 20 or config.price_basis != 'raw' or
        config.initial_capital != 1e6 or config.min_fee != 20. or
        config.weighting != 'equal' or config.weight_cap != 1. or
        not config.preserve_frozen_overflow or not config.benchmark_proportional_only or
        config.benchmark_code is not None or config.hold_buffer_rank is None):
        raise Stop('unsupported primary endpoint config')
    for day in sessions[1:]:
        if day not in market or not isinstance(market[day],dict): _invalid('missing market day')
    for day in sessions[:-1]:
        if day not in signals: _invalid('missing daily signal')
        _signal(signals[day],day)
    strategy=_arm(sessions,market,signals,config,False)
    benchmark=_arm(sessions,market,signals,replace(config,min_fee=0.),True)
    return dict(schema='forward-wp6-synthetic-endpoint-a-v1',evidence_class='SYNTHETIC_CPU_ONLY',
        endpoint='A',phase=phase,forward_admitted=False,sessions=list(sessions),
        strategy=strategy,benchmark=benchmark,
        excess_returns=[a-b for a,b in zip(strategy['returns'],benchmark['returns'])])


def endpoint_unblind(result,*,calendar,**kwargs):
    """Delegate integrity, publication, s252+11 and Human/manual gates to WP5.

    Does not publish result rows, authenticate Human, or open real results.
    """
    if (result.get('phase') != 'FORMAL' or result.get('endpoint') != 'A'
            or result.get('schema') != 'forward-wp6-synthetic-endpoint-a-v1'):
        raise Stop('invalid formal endpoint')
    sessions=endpoint_sessions(calendar,result['sessions'])
    for arm in ('strategy','benchmark'):
        rows=result[arm]['daily']; returns=result[arm]['returns']
        if (len(rows)!=253 or len(returns)!=252 or rows[0]['nav']!=1. or
            [r['at'] for r in rows] != [d+'T13:30:00+08:00' for d in sessions] or
            returns != [b['nav']/a['nav']-1 for a,b in zip(rows,rows[1:])]):
            raise Stop('invalid paired endpoint rows/returns')
    return wp5.manual_unblind(calendar=calendar,expected_sessions=list(sessions[1:]),**kwargs)
