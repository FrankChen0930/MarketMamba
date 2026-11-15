"""Deterministic expanding walk-forward index frames on observed union sessions."""
from __future__ import annotations

import pandas as pd
from .labels import positive_integer, purged_training_mask


def walk_forward_splits(frame: pd.DataFrame, *, min_train_sessions=252,
                        test_sessions=63, horizon=20, purge=None, embargo=0) -> list[dict]:
    """Return [{'fold': int, 'train': index frame, 'test': index frame}].

    row_index references the caller's unique index. Test includes feature-only
    tail rows; consumers must exclude unknown targets only for evaluation, never
    remove predictions based on future label availability. Partial final fold is
    omitted. min_train_sessions counts date support before the safety gap.
    """
    min_train_sessions = positive_integer(min_train_sessions,'min_train_sessions')
    test_sessions = positive_integer(test_sessions,'test_sessions')
    horizon = positive_integer(horizon,'horizon')
    purge = horizon if purge is None else purge
    if not frame.index.is_unique or frame.duplicated(['code','date']).any():
        raise ValueError('unique row indices and stock/date keys required')
    days = pd.DatetimeIndex(frame.date.unique()).sort_values()
    if len(days)==0:
        raise ValueError('empty calendar')
    # Validate even if no full folds can be produced.
    purged_training_mask(frame,days[0],horizon=horizon,purge=purge,embargo=embargo)
    end = f'label_end_{horizon}d'
    columns = ['code','date',end]
    base = frame[columns].copy()
    base.insert(0,'row_index',frame.index)
    folds = []
    start = min_train_sessions + purge + embargo
    for i in range(start,len(days)-test_sessions+1,test_sessions):
        mask = purged_training_mask(frame,days[i],horizon=horizon,purge=purge,embargo=embargo)
        train = base.loc[mask].sort_values(['date','code']).reset_index(drop=True)
        test = base.loc[frame.date.isin(days[i:i+test_sessions])].sort_values(['date','code']).reset_index(drop=True)
        if train.empty or train.date.nunique() < min_train_sessions:
            continue
        if train[end].max() >= test.date.min():
            raise ValueError('label interval overlaps test')
        folds.append(dict(fold=len(folds),train=train,test=test))
    return folds
