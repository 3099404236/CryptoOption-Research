#!/bin/bash
# 5-day high-frequency window: wait for the 2026-10-16 refresh, rebuild the affected files, run both coins.
W=/mnt/quant_storage/quant_workspace/lean_workspace_v2
R=/mnt/quant_storage/quant_workspace/deribit_raw
P=/home/czf/miniconda3/bin/python
cd $W/etl
export DERIBIT_RAW=$R
wait_for() { until grep -q "$1" "$2" 2>/dev/null; do sleep 30; done; }
wait_for "REFRESH_DONE" $R/logs/refresh_20261016.out
wait_for "BUILD_DONE BTC" $R/logs/build_btc_all.out
wait_for "BUILD_DONE ETH" $R/logs/build_eth_all.out
$P build_lean_data.py --currency ETH --start 2020-01-01 --end 2026-10-16 --underlying-only --second-start 2026-10-10 --second-end 2026-10-16 > $R/logs/rebuild_eth_u.out 2>&1
$P build_lean_data.py --currency BTC --start 2019-01-01 --end 2026-10-16 --underlying-only --second-start 2026-10-10 --second-end 2026-10-16 > $R/logs/rebuild_btc_u.out 2>&1
$P build_lean_data.py --currency ETH --start 2026-09-01 --end 2026-10-16 --skip-underlying > $R/logs/rebuild_eth_o.out 2>&1
$P build_lean_data.py --currency BTC --start 2026-09-01 --end 2026-10-16 --skip-underlying > $R/logs/rebuild_btc_o.out 2>&1
cd $W
$P run_suite.py suites/intraday_eth_5d.json results/intraday_eth_5d.json --workers 2 >> results/run_5d.log 2>&1
$P run_suite.py suites/intraday_btc_5d.json results/intraday_btc_5d.json --workers 2 >> results/run_5d.log 2>&1
echo RUN_5D_DONE >> results/run_5d.log
