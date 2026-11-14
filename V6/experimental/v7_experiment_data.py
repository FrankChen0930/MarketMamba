"""Vectorized calendar windows and bounded CPU prefetch for V7 experiments."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

class FastDataset:
    """Store each observation once; materialize only the requested daily window."""
    def __init__(self, index, graph, dates, length=60):
        self.index, self.graph = index, graph
        self.dates = tuple(dates)
        self.length = length
    def __len__(self):
        return len(self.dates)
    def __getitem__(self, item):
        day = self.dates[item]
        ix = self.index
        end = ix.day_positions[day]
        stocks = ix.eligible[end]
        if not len(stocks):
            raise ValueError("no eligible stocks: " + day)
        days = np.arange(max(0, end-self.length+1), end+1)
        rows = ix.rows[days[:, None], stocks[None, :]].T
        observation = rows >= 0
        x = ix.features[np.maximum(rows, 0)].copy()
        x[~observation] = 0
        x[..., 47:] = 0
        padding = days[None, :] >= ix.first[stocks, None]
        left = self.length-len(days)
        if left:
            x = np.pad(x, ((0,0),(left,0),(0,0)))
            observation = np.pad(observation, ((0,0),(left,0)))
            padding = np.pad(padding, ((0,0),(left,0)))
        labels = ix.labels[ix.rows[end, stocks]].copy()
        ids = tuple(ix.stocks[i] for i in stocks)
        edges, attrs = self.graph.edges_for(ids)
        first = padding.argmax(1)
        groups = [(int(start), torch.from_numpy(np.flatnonzero(first == start)))
                  for start in np.unique(first)]
        return {"x": torch.from_numpy(x), "padding_mask": torch.from_numpy(padding),
                "observation_mask": torch.from_numpy(observation), "labels": torch.from_numpy(labels),
                "stock_ids": ids, "edge_index": edges, "edge_attr": attrs, "groups": groups, "Date": day}

class PackedIndex:
    def __init__(self, original):
        self.stocks = tuple(sorted(original.histories))
        self.day_positions = {str(d): i for i,d in enumerate(original.calendar)}
        nrows = sum(len(h.dates) for h in original.histories.values())
        self.features = np.empty((nrows,59),np.float32)
        self.labels = np.empty((nrows,2),np.float32)
        self.rows = np.full((len(original.calendar),len(self.stocks)),-1,np.int32)
        self.first = np.empty(len(self.stocks),np.int32)
        offset = 0
        for col, stock in enumerate(self.stocks):
            h = original.histories[stock]
            n = len(h.dates)
            positions = np.searchsorted(original.calendar,h.dates)
            self.features[offset:offset+n] = h.features
            self.labels[offset:offset+n] = h.labels
            self.rows[positions[h.valid],col] = np.arange(offset,offset+n,dtype=np.int32)[h.valid]
            self.first[col] = positions[0]
            offset += n
        self.eligible = [np.flatnonzero(row>=0) for row in self.rows]

class FastGraph:
    def __init__(self, graph):
        self.lookup = graph.lookup
        self.size = len(graph.stock_ids)
        self.sources = np.repeat(np.arange(self.size),np.diff(graph.indptr))
        self.targets = graph.indices
        self.weights = graph.weights.astype(np.float32)
    def edges_for(self, stock_ids):
        mapping = np.full(self.size,-1,np.int64)
        for i, stock in enumerate(stock_ids):
            if stock in self.lookup:
                mapping[self.lookup[stock]] = i
        sources, targets = mapping[self.sources], mapping[self.targets]
        keep = (sources>=0)&(targets>=0)
        return (torch.from_numpy(np.stack((sources[keep],targets[keep]))),
                torch.from_numpy(self.weights[keep]))

def batches(dataset, start=0, prefetch=True):
    if not prefetch:
        for i in range(start,len(dataset)):
            yield i,dataset[i]
        return
    # One worker, at most one future and the current batch. No RNG in data preparation.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(dataset.__getitem__,start) if start<len(dataset) else None
        for i in range(start,len(dataset)):
            sample = future.result()
            future = pool.submit(dataset.__getitem__,i+1) if i+1<len(dataset) else None
            yield i,sample

def to_device(sample, device):
    out = dict(sample)
    for name in ("x","padding_mask","observation_mask","labels","edge_index","edge_attr"):
        t=sample[name]
        out[name]=t.pin_memory().to(device,non_blocking=True) if device.type=="cuda" else t
    out["groups"]=[(s,ids.to(device,non_blocking=True)) for s,ids in sample["groups"]]
    return out
