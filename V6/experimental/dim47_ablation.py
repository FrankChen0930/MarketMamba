"""
dim47_ablation.py — 把 Group D **整組拿掉**（59 → 47 維）訓一組，對已完成的 mask 組
=====================================================================================
為什麼這不是「順便清理」而是一個新 arm
--------------------------------------
2×2 已經證實 Group D 是負貢獻，但那是用 **mask（值歸零）** 量的，
`groupd_ablation.py` 當時刻意選 mask 而非砍維度，理由記在該檔：

    砍維度會改 GROUP_DIMS → sub_dim 重新分配、參數量變、RNG 分岔，
    完全重演 GAT 消融 A vs B 的三個干擾項。

`FactorGroupedEmbedding` 按**組別大小比例**分配 d_model=256。拿掉 12 維之後
A/B/C 三組會分到更多投影容量 → **這是不同的模型**，不是「同一個模型少吃 12 維」。
目前被證明有 +38.0% 的設定是 **59 維 + Group D 歸零**，不是 47 維。
所以要用就得先量。

設定（除了維度，其餘與 `v2_kg_nomacro` 逐項相同）
------------------------------------------------
  v2 圖 GAT ✅｜dropout 0.2｜epochs 10 / early_stop 5｜同 seed｜同切分
  控制組＝已完成的 `no_macro + gatv2`（IC +0.1145），從 JSON 匯入、不重跑。

⚠️ 判讀紀律（跑之前定死）
------------------------
  47 維要「勝出」必須 **重評 IC ≥ 控制組 + 0.009**（實務門檻，F5 沿用）。
  達不到就**維持 59 維 + mask**——那是已經被組合層驗證過的設定，
  而換架構的成本是推論端接線要整套重做一次。
  **不可以因為「參數比較少、比較乾淨」就採用**，那不是證據。

Colab 用法
----------
    # ⚠️ 必須在**全新的 runtime** 或至少在 import 任何 marketmamba.models.* 之前跑
    from experimental.dim47_ablation import run_dim47
    run_dim47(df, drive_dir="/content/drive/MyDrive/MarketMamba_V6")
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

MACRO_GROUP = "macro_environment"
CONTROL_JSON = "groupd_ablation_result_gatv2.json"
CONTROL_KEY = "no_macro"
WIN_THRESHOLD = 0.009          # 勝出門檻，跑之前定死


def patch_config_47d() -> int:
    """
    把 config 切成 47 維（拿掉 Group D）。

    ⚠️ **必須在 import 任何 `marketmamba.models.*` 之前呼叫。**
    `architecture.FactorGroupedEmbedding.__init__` 在 def 執行當下就把
    `GROUP_DIMS` / `INPUT_DIM` 綁進預設參數 → 已經 import 過就改不動了，
    而且**完全不會報錯**（CLAUDE.md 記載的 Colab 三個坑之一：模組快取）。
    下方 `_assert_model_is_47d()` 就是為了在那種情況下當場失敗。
    """
    import marketmamba.config as cfg

    # 59 維前置：RS 要先在 group A（與其他 F6 實驗同基礎），再拿掉 Group D
    rs = ["RS_5d", "RS_20d", "RS_60d"]
    if not all(r in cfg.FEATURE_GROUPS["price_momentum"] for r in rs):
        cfg.FEATURE_GROUPS["price_momentum"] = cfg.FEATURE_GROUPS["price_momentum"] + rs

    cfg.FEATURE_GROUPS.pop(MACRO_GROUP, None)
    cfg.FEATURE_COLS = [c for g in cfg.FEATURE_GROUPS.values() for c in g]
    cfg.GROUP_DIMS = {k: len(v) for k, v in cfg.FEATURE_GROUPS.items()}
    cfg.INPUT_DIM = len(cfg.FEATURE_COLS)
    assert cfg.INPUT_DIM == 47, f"expected 47, got {cfg.INPUT_DIM}：{cfg.GROUP_DIMS}"
    print(f"[47d] config 已切成 {cfg.INPUT_DIM} 維｜組別 {cfg.GROUP_DIMS}", flush=True)
    return cfg.INPUT_DIM


def _make_embedding_3g():
    """
    47 維專用的 3 組 embedding（A/B/C，沒有 D）。

    ⚠️ 為什麼要另寫一個：`architecture.FactorGroupedEmbedding` 把四個組名
    **寫死**在程式裡（`raw_dims["macro_environment"]`，architecture.py:121），
    拿掉 Group D 會直接 KeyError。而 `marketmamba/models/` 是受保護目錄，
    所以照本專案既有慣例——**monkeypatch，不改受保護檔案**。

    比例分配、餘數給最大組、LayerNorm、Dropout 全部**逐行照抄**原始實作，
    唯一的差別就是少一組。另寫一份等於在「有沒有 Group D」之外多塞一個
    實作差異的變因（F5 方法紀律第 ③ 條）。
    """
    import torch
    import torch.nn as nn

    import marketmamba.config as cfg
    from marketmamba.config import D_MODEL, DROPOUT as _DO

    class FactorGroupedEmbedding3G(nn.Module):
        def __init__(self, group_dims: dict[str, int] | None = None,
                     d_model: int = D_MODEL):
            super().__init__()
            gd = dict(group_dims or cfg.GROUP_DIMS)
            assert MACRO_GROUP not in gd, f"3 組版本不該收到 Group D：{gd}"
            total = sum(gd.values())
            raw = {k: int(d_model * v / total) for k, v in gd.items()}
            raw[max(gd, key=gd.get)] += d_model - sum(raw.values())   # 餘數給最大組

            self.proj_A = nn.Linear(gd["price_momentum"], raw["price_momentum"])
            self.proj_B = nn.Linear(gd["institutional_flow"], raw["institutional_flow"])
            self.proj_C = nn.Linear(gd["fundamentals"], raw["fundamentals"])
            self.norm = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(_DO)
            a, b, c = (gd["price_momentum"], gd["institutional_flow"], gd["fundamentals"])
            self._slices = {"A": (0, a), "B": (a, a + b), "C": (a + b, a + b + c)}
            print(f"[47d] embedding 子維度分配 {raw}（合計 {sum(raw.values())} = d_model）",
                  flush=True)

        def forward(self, x):
            sA, eA = self._slices["A"]; sB, eB = self._slices["B"]; sC, eC = self._slices["C"]
            out = self.norm(torch.cat([self.proj_A(x[..., sA:eA]),
                                       self.proj_B(x[..., sB:eB]),
                                       self.proj_C(x[..., sC:eC])], dim=-1))
            return self.drop(out)

    return FactorGroupedEmbedding3G


def _assert_model_is_47d() -> int:
    """
    建一個模型、**實測**它的 embedding 真的只吃 47 維，並印出參數量。

    這一道是擋「config 改了但 architecture 早就 import 過」的唯一防線——
    那種情況下模型仍是 59 維、訓練會照常跑完、數字看起來合理但整個實驗作廢。
    """
    import torch

    import experimental.short_model as sm
    sm.FactorGroupedEmbedding = _make_embedding_3g()      # monkeypatch（不改受保護檔案）
    m = sm.ShortModelV6(use_gat=True, dropout=0.2)
    # 逐組投影的輸入維度加總 = 模型實際吃的特徵數
    dims = [p.in_features for n, p in m.embedding.named_modules()
            if isinstance(p, torch.nn.Linear) and n.startswith("proj")]
    total = sum(dims)
    n_par = m.n_parameters
    print(f"[47d] 模型實測：投影輸入維度 {dims} 合計 {total}｜參數 {n_par:,}", flush=True)
    print(f"[47d] （對照：59 維版本是 1,659,005 參數。**數字不同是預期的**，"
          f"那正是這個 arm 要量的東西）", flush=True)
    if total != 47:
        raise SystemExit(
            f"❌ 模型實際吃 {total} 維、不是 47 維。\n"
            f"   幾乎確定是 `marketmamba.models.architecture` 在 patch_config_47d() "
            f"之前就被 import 了（Colab 模組快取）。\n"
            f"   解法：Runtime → Restart，然後**第一件事**就是 patch_config_47d()。")
    del m
    return n_par


def _load_control(drive_dir: Optional[str]) -> Optional[dict]:
    """控制組沿用已完成的 `no_macro + gatv2`（不重跑，省 3.6h）。"""
    for d in filter(None, (drive_dir, "/content", ".")):
        p = os.path.join(d, CONTROL_JSON)
        if os.path.exists(p):
            arm = json.load(open(p, encoding="utf-8"))["arms"].get(CONTROL_KEY)
            if arm:
                print(f"[47d] 控制組匯入 {p}:{CONTROL_KEY}"
                      f"｜重評 IC {arm.get('rescored_mean_ic')}"
                      f"｜epochs {arm.get('epochs')} / early_stop {arm.get('early_stop')}",
                      flush=True)
                return arm
    print(f"[47d] ⚠ 找不到 {CONTROL_JSON} → 本輪無控制組可對照，只會印出自己的數字",
          flush=True)
    return None


def run_dim47(df, drive_dir: Optional[str] = None, epochs: int = 10,
              early_stop: int = 5, kg_file: str = "knowledge_graph_v2.npz",
              cutoff_train_end: str = "2023-12-31") -> dict:
    """訓練 47 維 arm（v2 圖 + GAT），與控制組同 harness。"""
    from experimental.groupd_ablation import _train_one_arm
    from experimental.kg_ablation import build_dates

    n_par = _assert_model_is_47d()
    ctrl = _load_control(drive_dir)

    train_dates, val_dates = build_dates(df, cutoff_train_end)
    print(f"[47d] train {len(train_dates)} 天 / val {len(val_dates)} 天", flush=True)

    # `_train_one_arm` 用 ARMS[arm] 決定要不要 mask macro——47 維下 Group D 已經
    # 不在 FEATURE_COLS 裡，沒有東西可 mask，所以借用 "with_macro"（mask=False）。
    # 這不是「保留 macro」，是「沒有 macro 可保留」。
    res = _train_one_arm(df, train_dates, val_dates, "with_macro",
                         epochs, early_stop, use_gat=True, kg_file=kg_file,
                         drive_dir=drive_dir, tag="_dim47")
    res["input_dim"] = 47
    res["n_parameters"] = n_par

    mine = res.get("rescored_mean_ic")
    print("\n" + "=" * 72)
    print("47 維（砍 Group D） vs 59 維 + mask（歸零）")
    print("=" * 72)
    if ctrl and mine is not None:
        base = ctrl.get("rescored_mean_ic")
        delta = mine - base
        ok = delta >= WIN_THRESHOLD
        print(f"{'arm':22s}{'重評 IC':>10s}{'參數量':>12s}")
        print(f"{'59維 + mask（現行）':22s}{base:10.4f}{ctrl.get('n_parameters', 0):12,}")
        print(f"{'47維（砍掉）':22s}{mine:10.4f}{n_par:12,}")
        print(f"\nΔ = {delta:+.4f}（門檻 +{WIN_THRESHOLD}）")
        print(f"→ {'✅ 達標，可考慮換 47 維' if ok else '❌ 未達標 → **維持 59 維 + mask**'}")
        if not ok:
            print("   理由：59 維 + mask 是已經被組合層驗證過的設定；換架構要把"
                  "推論端接線整套重做一次，沒有證據支持就不換。")
        res["control_mean_ic"] = base
        res["delta"] = round(delta, 4)
        res["passed"] = bool(ok)
    else:
        print(f"47 維重評 IC = {mine}｜參數 {n_par:,}（無控制組可對照）")
    print("=" * 72, flush=True)

    if drive_dir:
        out = os.path.join(drive_dir, "dim47_ablation_result.json")
        json.dump({"arms": {"dim47": res}}, open(out, "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        print(f"[47d] 結果 → {out}", flush=True)
    return res
