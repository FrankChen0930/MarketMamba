#!/bin/bash
# 一次性重跑腳本：8/12 晚間排程因電腦當機中斷，補跑完整三條鏈
# 用法（從 PowerShell）：wsl -d Ubuntu -- bash V6/scripts/rerun_20260812.sh
source /home/frank/miniconda3/etc/profile.d/conda.sh
conda activate mamba_env
cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba || exit 1

echo "=== [1/3] V6.1 run_daily_inference.py @ $(date '+%F %T') ==="
python V6/run_daily_inference.py
RC1=$?
echo "### EXIT_V61=$RC1 @ $(date '+%F %T')"

echo "=== [2/3] run_dual_inference.py @ $(date '+%F %T') ==="
python V6/run_dual_inference.py
RC2=$?
echo "### EXIT_DUAL=$RC2 @ $(date '+%F %T')"

echo "=== [3/3] V6.2 run_v62_daily.py @ $(date '+%F %T') ==="
python V6/run_v62_daily.py
RC3=$?
echo "### EXIT_V62=$RC3 @ $(date '+%F %T')"

echo "=== ALL DONE @ $(date '+%F %T') | v61=$RC1 dual=$RC2 v62=$RC3 ==="
