#!/bin/bash
# detached launcher（避免多層引號在指令列上把 & 之後整段吃掉）
cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba || exit 1
setsid nohup bash V6/scripts/rerun_20260812.sh > V6/logs/rerun_20260812.out 2>&1 < /dev/null &
echo "launched pid=$!"
disown
