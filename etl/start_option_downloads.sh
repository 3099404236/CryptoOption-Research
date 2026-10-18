#!/bin/bash
# (Re)start the long option-trade downloads with higher concurrency. Finished days are skipped.
export DERIBIT_RAW=/mnt/quant_storage/quant_workspace/deribit_raw
export DERIBIT_PROXY=http://127.0.0.1:7897
cd "$(dirname "$0")"
setsid nohup /home/czf/miniconda3/bin/python fetch_deribit.py options --currency BTC --start 2019-01-01 --end 2026-10-16 --workers 16 \
  > "$DERIBIT_RAW/logs/job_opt_btc.out" 2>&1 < /dev/null &
setsid nohup /home/czf/miniconda3/bin/python fetch_deribit.py options --currency ETH --start 2020-01-01 --end 2026-10-16 --workers 12 \
  > "$DERIBIT_RAW/logs/job_opt_eth.out" 2>&1 < /dev/null &
echo "started"
