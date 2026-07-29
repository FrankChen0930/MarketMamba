"""
TAIFEX 三大法人端點探測（2026-07-29）
=====================================
在寫正式 fetcher 前先確認：端點形狀、欄名、日期參數是否真的生效、歷史深度。

依本專案雷區清單，寫任何 fetcher 前必須先回答：
  1. 端點是否真的認日期參數？（回傳的日期要與請求核對）
  2. 欄位對映靠什麼？（一律用欄名，不硬編索引）
  3. 歷史能回溯到哪一年？（不可假設與其他端點相同）
  4. 非交易日回什麼？（骨架列 vs 空表）

用法：python V6/scripts/probe_taifex.py
"""
from __future__ import annotations

import requests

H = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.taifex.com.tw/"}
FUT = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
OPT = "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown"


def probe(name: str, url: str, d1: str, d2: str) -> None:
    data = {
        "firstDate": d1, "lastDate": d2,
        "queryStartDate": d1, "queryEndDate": d2,
        "commodityId": "",
    }
    try:
        r = requests.post(url, data=data, headers=H, timeout=45)
    except Exception as e:                                    # noqa: BLE001
        print(f"--- {name} {d1}: ERROR {e}")
        return
    t = r.text
    lines = [x for x in t.splitlines() if x.strip()]
    print(f"--- {name} {d1}~{d2}: HTTP {r.status_code}｜{len(t):,} bytes"
          f"｜{r.headers.get('content-type', '')[:28]}｜{len(lines)} 非空列")
    for ln in lines[:3]:
        print("      " + ln[:190])


def main() -> None:
    print("=" * 78)
    print("■ 1. 基本形狀（近期交易日）")
    print("=" * 78)
    probe("期貨", FUT, "2026/07/28", "2026/07/28")
    probe("選擇權", OPT, "2026/07/28", "2026/07/28")

    print()
    print("=" * 78)
    print("■ 2. 日期參數是否生效（不同日期應回不同內容）")
    print("=" * 78)
    probe("期貨", FUT, "2026/05/06", "2026/05/06")

    print()
    print("=" * 78)
    print("■ 3. 非交易日（2026/07/10 已確認非交易日）")
    print("=" * 78)
    probe("期貨", FUT, "2026/07/10", "2026/07/10")

    print()
    print("=" * 78)
    print("■ 4. 歷史深度")
    print("=" * 78)
    for y in ("2008", "2012", "2018"):
        probe("期貨", FUT, f"{y}/06/02", f"{y}/06/02")

    print()
    print("=" * 78)
    print("■ 5. 區間查詢是否支援（能省下逐日迴圈）")
    print("=" * 78)
    probe("期貨", FUT, "2026/07/01", "2026/07/28")


if __name__ == "__main__":
    main()
