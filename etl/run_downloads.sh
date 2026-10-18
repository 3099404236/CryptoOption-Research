#!/bin/bash
# Background download of all raw Deribit data used by the V2 backtests.
# Resumable: finished files are skipped, so it can simply be started again after an interruption.
export DERIBIT_RAW=/mnt/quant_storage/quant_workspace/deribit_raw
export DERIBIT_PROXY=http://127.0.0.1:7897
P=/home/czf/miniconda3/bin/python
cd "$(dirname "$0")"
L=$DERIBIT_RAW/logs
mkdir -p "$L"
( $P fetch_deribit.py delivery --currency BTC && $P fetch_deribit.py delivery --currency ETH \
  && $P fetch_deribit.py perp1m --currency BTC --start 2018-12-01 --end 2026-10-16 \
  && $P fetch_deribit.py perp1m --currency ETH --start 2019-01-01 --end 2026-10-16 \
  && $P fetch_deribit.py perptrades --currency ETH --start 2026-10-10 --end 2026-10-16 --workers 3 \
  && $P fetch_deribit.py perptrades --currency BTC --start 2026-10-10 --end 2026-10-16 --workers 3 \
  && echo PERP_DONE ) > "$L/job_perp.out" 2>&1 &
$P fetch_deribit.py options --currency BTC --start 2019-01-01 --end 2026-10-16 --workers 6 > "$L/job_opt_btc.out" 2>&1 &
$P fetch_deribit.py options --currency ETH --start 2020-01-01 --end 2026-10-16 --workers 4 > "$L/job_opt_eth.out" 2>&1 &
wait
echo ALL_DOWNLOADS_DONE
