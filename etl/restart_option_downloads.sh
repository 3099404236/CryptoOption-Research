#!/bin/bash
# Stop and restart the long option downloads with the given worker counts (finished days are skipped).
BTC_WORKERS=${1:-24}
ETH_WORKERS=${2:-16}
pkill -f "fetch_deribit.py options --currency BTC --start 2019-01-01"
pkill -f "fetch_deribit.py options --currency ETH --start 2020-01-01"
sleep 3
export DERIBIT_RAW=/mnt/quant_storage/quant_workspace/deribit_raw
export DERIBIT_PROXY=http://127.0.0.1:7897
cd "$(dirname "$0")"
setsid nohup /home/czf/miniconda3/bin/python fetch_deribit.py options --currency BTC --start 2019-01-01 --end 2026-10-16 --workers "$BTC_WORKERS" \
  >> "$DERIBIT_RAW/logs/job_opt_btc.out" 2>&1 < /dev/null &
setsid nohup /home/czf/miniconda3/bin/python fetch_deribit.py options --currency ETH --start 2020-01-01 --end 2026-10-16 --workers "$ETH_WORKERS" \
  >> "$DERIBIT_RAW/logs/job_opt_eth.out" 2>&1 < /dev/null &
echo "restarted with $BTC_WORKERS / $ETH_WORKERS workers"
