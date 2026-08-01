#!/usr/bin/env bash
# GRU 在 v2 基礎重訓（h64 only、5d only）——2026-08-01
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba_env
cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba
export MM_PROTOCOL=v2
exec python V6/experimental/baseline_rnn.py --cell gru --hidden 64 --skip-20d
