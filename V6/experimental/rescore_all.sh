#!/usr/bin/env bash
# 重評分所有 Mamba arm（新面板）。可續跑：已有 __live 輸出就跳過。
# log 一律寫 /mnt/c/...（Windows 端），WSL 重啟後仍在。
# 注意：不可用 `set -u` —— conda 的 activate 鉤子引用了未設定的 SYS_SYSROOT，會當場中止
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_env
cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba
LOGDIR="/mnt/c/Users/Master/AppData/Local/Temp/claude/D--Desktop-work-ProjectForMe-MarketMamba/146d6f54-0a58-4b61-b12f-7e2f65c71daa/scratchpad"
SC="V6/experimental/result/scores"
for arm in v2_kg_nomacro_h10 head10d head20d v2_kg v3_kg old_kg no_gat; do
  if [ -f "$SC/${arm}__live.parquet" ]; then
    echo "[skip] $arm 已有輸出"; continue
  fi
  echo "[run ] $arm  $(date +%H:%M:%S)"
  python V6/run_v62_inference.py --arm "$arm" --score-window > "$LOGDIR/sw_${arm}.log" 2>&1
  echo "[done] $arm exit=$? $(date +%H:%M:%S)"
done
echo "ALL DONE $(date +%H:%M:%S)"
