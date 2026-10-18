#!/bin/bash
# Second-level suite in two halves: ETH now (its ticks are complete), BTC after the BTC download finishes.
W=/mnt/quant_storage/quant_workspace/lean_workspace_v2
L=/mnt/quant_storage/quant_workspace/logs
P=/opt/qenv/bin/python
export LEAN_BIN=/opt/qenv/bin/lean PATH=/opt/qenv/bin:$PATH DERIBIT_RAW=/mnt/quant_storage/quant_workspace/deribit_raw
log() { echo "$(date -u '+%F %T') $*" >> $L/seconds_pipeline.log; }
build() {   # $1 = coin, $2 = first year
  pids=""
  for y in $(seq $2 2026); do
    e=$y-12-31; [ $y = 2026 ] && e=2026-10-16
    (cd $W/etl && $P build_seconds.py --currency $1 --start $y-01-01 --end $e > $L/seconds_$1_$y.out 2>&1) &
    pids="$pids $!"
  done
  wait $pids          # only the build jobs (a plain "wait" would also wait for the running ETH suite)
  log "$1 second bars done; days missing/empty: $(cat $L/seconds_$1_*.out | grep -c 'MISSING\|EMPTY')"
}
run() {     # $1 = coin
  cd $W && $P run_suite.py suites/intraday_full_second_$1.json results/intraday_full_second_$1.json --workers 10 >> $L/seconds_pipeline.log 2>&1
  $P verify_v2.py results/intraday_full_second_$1.json results/verify_intraday_full_second_$1.json > results/verify_intraday_full_second_$1.txt 2>&1
  log "$1 second-level suite and verification done"
}
build ETH 2020
log "running the ETH second-level suite"
run eth &
until grep -q PERPTRADES_DONE $L/perptrades.log 2>/dev/null; do sleep 60; done
build BTC 2019
log "running the BTC second-level suite"
run btc &
wait
log "SECONDS_PIPELINE_DONE"
