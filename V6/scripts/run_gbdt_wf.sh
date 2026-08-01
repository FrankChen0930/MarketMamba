#!/usr/bin/env bash
# GBDT walk-forward（11 年連續 OOS）——2026-08-01
cd /d/Desktop/work/ProjectForMe/MarketMamba 2>/dev/null || cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba
export MM_PROTOCOL=v2
python V6/experimental/wf_scores.py --model gbdt --years 2015 2026
python V6/experimental/wf_scores.py --model gbdt --merge
