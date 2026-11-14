"""
progress_window.py — V6.2 推論進度視窗（WSLg / tkinter）
========================================================
沿用 V6.1 的視覺語言（淺色卡片 + 狀態色 + 進度條），修掉它的兩個實際問題：

 ① **豆腐方塊是靜默發生的**。V6.1 的字型 fallback 迴圈在找不到任何候選時
    直接放棄、改用 `TkDefaultFont`，而 WSL 預設**一個中文字型都沒有**
    （實測 `fc-list :lang=zh-tw` 是空的）→ 整個視窗的中文變成方框，
    看起來像壞掉，但程式不會說任何話。
    → 本檔改成：找不到 CJK 字型就**自動改用英文標籤**並印出明確警告，
      告訴使用者要裝什麼。**寧可顯示英文，也不要顯示方框。**

 ② 沒有 DISPLAY（純 headless 執行）時要安靜跳過，不能讓進度視窗
    變成整個每日流程的新失敗點。

用法
----
    def work(ui) -> int:
        ui.update(0, "running"); ...; ui.update(0, "done", "8/10 源當日")
        return 0                       # 回傳 exit code

    ui = ProgressWindow(["抓取資料", "建特徵矩陣", "模型推論", "組合層"],
                        ["Fetch data", "Build features", "Inference", "Portfolio"])
    sys.exit(ui.run(work))    # 成功 3 秒自動關；失敗保持開啟並置頂
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger("progress_window")

_BG, _CARD, _FG, _DIM = "#F4F6F8", "#FFFFFF", "#1C1C2E", "#8A8FA8"
_ACCENT, _SEP = "#3B6FE8", "#E8EAF0"
_COLORS = {"pending": "#B0B8C8", "running": _ACCENT, "done": "#22A96A",
           "failed": "#E53935", "skipped": "#F59E0B"}
_ICONS = {"pending": "○", "running": "◉", "done": "✓", "failed": "✗", "skipped": "—"}

# 依偏好排序。前三個是 Ubuntu 上 `fonts-noto-cjk` 會裝的。
_CJK_FONTS = [
    "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "Noto Sans TC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
    "Droid Sans Fallback", "AR PL UMing TW", "Microsoft JhengHei",
]
INSTALL_HINT = ("WSL 內沒有任何中文字型 → 視窗改用英文標籤。\n"
                "  要顯示中文請在 WSL 執行（需 sudo 密碼）：\n"
                "    sudo apt-get update && sudo apt-get install -y fonts-noto-cjk\n"
                "  裝完不需重開機，下次推論就會是中文。")


def _pick_cjk_font(tkfont) -> Optional[str]:
    """回傳可用的中文字型名，沒有就回 None（呼叫端據此改用英文標籤）。"""
    try:
        fams = set(tkfont.families())
    except Exception:                                       # noqa: BLE001
        return None
    for f in _CJK_FONTS:
        if f in fams:
            return f
    # 名稱可能帶後綴（如 "Noto Sans CJK TC Regular"）→ 再做一次寬鬆比對
    low = {f.lower(): f for f in fams}
    for key in ("noto sans cjk", "wenquanyi", "droid sans fallback", "uming"):
        for k, orig in low.items():
            if key in k:
                return orig
    return None


class ProgressWindow:
    """步驟進度視窗。所有對外方法都是 thread-safe（只往 queue 推）。"""

    def __init__(self, steps_zh: list[str], steps_en: Optional[list[str]] = None,
                 title: str = "MarketMamba V6.2"):
        self.steps_zh = steps_zh
        self.steps_en = steps_en or steps_zh
        self.title = title
        self._q: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._enabled = False

    # ── 對外 API（在工作執行緒呼叫）──────────────────────────────
    def update(self, idx: int, status: str, note: str = "") -> None:
        if self._enabled:
            self._q.put(("step", idx, status, note))

    def set_info(self, text: str) -> None:
        if self._enabled:
            self._q.put(("info", text))

    def finish(self, ok: bool) -> None:
        """一般不需要手動呼叫——`run()` 會在 work 回傳後自動送出。"""
        if self._enabled:
            self._q.put(("finish", ok))

    def _available(self) -> bool:
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            logger.info("[UI] 無 DISPLAY → 不開進度視窗（不影響推論）")
            return False
        try:
            import tkinter  # noqa: F401
        except Exception as e:                              # noqa: BLE001
            logger.info(f"[UI] tkinter 不可用（{e}）→ 不開進度視窗")
            return False
        return True

    def run(self, work) -> int:
        """
        **Tk 跑在主執行緒、工作跑在背景執行緒**（與 V6.1 相同的結構）。

        ⚠️ 反過來（Tk 在背景執行緒）會在程式結束時印
           `Tcl_AsyncDelete: async handler deleted by the wrong thread`——
           那是 Python 從主執行緒去 finalize 一個在別的執行緒建立的 Tcl
           interpreter。訊息本身無害，但每天在 log 尾巴留一行紅字，
           會讓人以為推論出錯了。實測過：只能靠「Tk 在主執行緒」根治，
           在背景執行緒裡怎麼 destroy 都消不掉。

        `work(ui) -> int`：在背景執行緒執行，回傳 exit code。
        UI 不可用時直接同步呼叫 `work`，行為完全一致。
        """
        if not self._available():
            return work(self)

        self._enabled = True
        rc = {"code": 1}

        def _worker() -> None:
            try:
                rc["code"] = work(self)
            except Exception as e:                          # noqa: BLE001
                logger.error(f"工作執行緒例外：{e}")
                rc["code"] = 1
            finally:
                self._q.put(("finish", rc["code"] == 0))

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        self._run()                                          # 主執行緒跑 mainloop
        self._thread.join(timeout=30)
        return rc["code"]

    # ── UI 執行緒 ────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import font as tkfont

            root = tk.Tk()
            root.withdraw()                       # 先隱藏，字型決定後再顯示

            fam = _pick_cjk_font(tkfont)
            has_cjk = fam is not None
            # 介面字串跟著字型走：有中文字型才用中文，否則整套換英文。
            # 混用會出現「一半中文一半方框」，比全英文更糟。
            T = ({"done": "完成", "failed": "失敗 — 視窗保持開啟", "wait": "準備中…"}
                 if has_cjk else
                 {"done": "Done", "failed": "FAILED — window stays open", "wait": "Starting…"})
            if has_cjk:
                names = self.steps_zh
                logger.info(f"[UI] 中文字型：{fam}")
            else:
                names = self.steps_en
                fam = "TkDefaultFont"
                # 規則 7：不可只做邏輯而不輸出。這裡如果不講，
                # 使用者看到的就是一個「怎麼變英文了」的謎題。
                logger.warning("[UI] " + INSTALL_HINT)

            root.title(self.title)
            root.configure(bg=_BG)
            root.geometry(f"460x{150 + 34 * len(names)}")
            root.resizable(False, False)

            f_h1 = tkfont.Font(family=fam, size=15, weight="bold")
            f_meta = tkfont.Font(family=fam, size=10)
            f_step = tkfont.Font(family=fam, size=12)
            f_icon = tkfont.Font(family=fam, size=13, weight="bold")
            f_stat = tkfont.Font(family=fam, size=11)

            head = tk.Frame(root, bg=_CARD)
            head.pack(fill="x")
            inner = tk.Frame(head, bg=_CARD)
            inner.pack(fill="x", padx=18, pady=(14, 12))
            tk.Label(inner, text=self.title, font=f_h1, bg=_CARD, fg=_FG).pack(anchor="w")
            lbl_meta = tk.Label(inner, text="", font=f_meta, bg=_CARD, fg=_DIM)
            lbl_meta.pack(anchor="w")
            tk.Frame(root, bg=_SEP, height=1).pack(fill="x")

            card = tk.Frame(root, bg=_CARD)
            card.pack(fill="both", expand=True)
            icons, labels, notes = [], [], []
            for nm in names:
                row = tk.Frame(card, bg=_CARD)
                row.pack(fill="x", padx=18, pady=3)
                ic = tk.Label(row, text=_ICONS["pending"], font=f_icon,
                              bg=_CARD, fg=_COLORS["pending"], width=2)
                ic.pack(side="left")
                lb = tk.Label(row, text=nm, font=f_step, bg=_CARD, fg=_FG, anchor="w")
                lb.pack(side="left")
                nt = tk.Label(row, text="", font=f_meta, bg=_CARD, fg=_DIM, anchor="e")
                nt.pack(side="right")
                icons.append(ic); labels.append(lb); notes.append(nt)

            tk.Frame(root, bg=_SEP, height=1).pack(fill="x")
            foot = tk.Frame(root, bg=_BG)
            foot.pack(fill="x")
            lbl_stat = tk.Label(foot, text=T["wait"], font=f_stat, bg=_BG, fg=_DIM)
            lbl_stat.pack(anchor="w", padx=18, pady=10)

            root.deiconify()
            state = {"closing": False}

            def pump() -> None:
                try:
                    while True:
                        msg = self._q.get_nowait()
                        if msg[0] == "step":
                            _, i, st, note = msg
                            if 0 <= i < len(icons):
                                icons[i].config(text=_ICONS.get(st, "○"),
                                                fg=_COLORS.get(st, _DIM))
                                labels[i].config(
                                    fg=_COLORS["failed"] if st == "failed" else _FG)
                                notes[i].config(text=note)
                                lbl_stat.config(text=f"{names[i]} — {st}")
                        elif msg[0] == "info":
                            lbl_meta.config(text=msg[1])
                        elif msg[0] == "finish":
                            if msg[1]:
                                lbl_stat.config(text=T["done"], fg=_COLORS["done"])
                                state["closing"] = True
                                root.after(3000, root.destroy)   # 成功 3 秒自動關
                            else:
                                lbl_stat.config(text=T["failed"], fg=_COLORS["failed"])
                                root.attributes("-topmost", True)  # 失敗保持開啟並置頂
                except queue.Empty:
                    pass
                if not state["closing"]:
                    root.after(120, pump)

            root.after(100, pump)
            root.mainloop()

            # ── 收尾：在**建立 Tk 的這個執行緒裡**把 interpreter 拆乾淨 ──
            # 不做的話結束時會印 `Tcl_AsyncDelete: async handler deleted by
            # the wrong thread`——那是 Python 結束時從主執行緒去 finalize
            # 一個在別的執行緒建立的 Tcl interpreter。訊息無害，但每天在
            # log 尾巴留一行紅字會讓人以為推論出錯了。
            try:
                for w in list(root.children.values()):
                    w.destroy()
                root.quit()
                root.destroy()
            except Exception:                               # noqa: BLE001
                pass                                        # 已經被 destroy 過
            finally:
                del root
        except Exception as e:                              # noqa: BLE001
            # 進度視窗絕不能成為每日流程的新失敗點
            logger.warning(f"[UI] 進度視窗異常（不影響推論）：{e}")
