#!/bin/bash
# BTC half of the second-level suite (the ETH half is already running), then the completion marker.
W=/mnt/quant_storage/quant_workspace/lean_workspace_v2
L=/mnt/quant_storage/quant_workspace/logs
P=/opt/qenv/bin/python
export LEAN_BIN=/opt/qenv/bin/lean PATH=/opt/qenv/bin:$PATH
log() { echo "$(date -u "+%F %T") $*" >> $L/seconds_pipeline.log; }
log "running the BTC second-level suite"
cd $W && $P run_suite.py suites/intraday_full_second_btc.json results/intraday_full_second_btc.json --workers 10 >> $L/seconds_pipeline.log 2>&1
$P verify_v2.py results/intraday_full_second_btc.json results/verify_intraday_full_second_btc.json > results/verify_intraday_full_second_btc.txt 2>&1
log "btc second-level suite and verification done"
until grep -q "eth second-level suite and verification done" $L/seconds_pipeline.log; do sleep 60; done
log "SECONDS_PIPELINE_DONE"
