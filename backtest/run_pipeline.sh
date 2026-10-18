#!/bin/bash
# Main V2 run pipeline (waits for the data builds, then runs the suites; results land in results/).
W=/mnt/quant_storage/quant_workspace/lean_workspace_v2
R=/mnt/quant_storage/quant_workspace/deribit_raw
P=/home/czf/miniconda3/bin/python
cd $W
wait_for() { until grep -q "$1" "$2" 2>/dev/null; do sleep 30; done; }
wait_for "BUILD_DONE BTC" $R/logs/build_btc_all.out
$P run_suite.py suites/weekly_btc.json results/weekly_btc.json --workers 3 >> results/pipeline.log 2>&1
$P run_suite.py suites/sensitivity_btc.json results/sensitivity_btc.json --workers 3 >> results/pipeline.log 2>&1
wait_for "BUILD_DONE ETH" $R/logs/build_eth_all.out
$P run_suite.py suites/intraday_eth_1y.json results/intraday_eth_1y.json --workers 3 >> results/pipeline.log 2>&1
$P run_suite.py suites/weekly_eth.json results/weekly_eth.json --workers 3 >> results/pipeline.log 2>&1
echo PIPELINE_DONE >> results/pipeline.log
