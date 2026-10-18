#!/bin/bash
# After the tick-trade downloads: build 1-second bars for every day (one process per coin-year), then run the
# intraday strategies with the second-level perpetual over the full history and verify every fill.
W=/mnt/quant_storage/quant_workspace/lean_workspace_v2
L=/mnt/quant_storage/quant_workspace/logs
P=/opt/qenv/bin/python
export LEAN_BIN=/opt/qenv/bin/lean PATH=/opt/qenv/bin:$PATH DERIBIT_RAW=/mnt/quant_storage/quant_workspace/deribit_raw
log() { echo "$(date -u '+%F %T') $*" >> $L/seconds_pipeline.log; }
log "waiting for the tick-trade downloads"
until grep -q PERPTRADES_DONE $L/perptrades.log 2>/dev/null; do sleep 60; done
cd $W/etl
for CUR in BTC ETH; do                         # retry failed days once (days already on disk are skipped)
  if grep -q ERROR $L/perptrades_$CUR.out; then
    Y0=2019; [ $CUR = ETH ] && Y0=2020
    log "retrying failed $CUR days"
    DERIBIT_PROXY=http://127.0.0.1:7890 $P fetch_deribit.py perptrades --currency $CUR --start $Y0-01-01 --end 2026-10-16 \
      --workers 8 > $L/perptrades_${CUR}_retry.out 2>&1
  fi
done
log "building 1-second bars"
for CUR in BTC ETH; do
  Y0=2019; [ $CUR = ETH ] && Y0=2020
  for y in $(seq $Y0 2026); do
    e=$y-12-31; [ $y = 2026 ] && e=2026-10-16
    $P build_seconds.py --currency $CUR --start $y-01-01 --end $e > $L/seconds_${CUR}_$y.out 2>&1 &
  done
done
wait
log "second bars done; days missing/empty: $(cat $L/seconds_*.out | grep -c 'MISSING\|EMPTY')"
until [ -f $L/PIPELINE_DONE ]; do sleep 60; done          # let the minute-level suites finish first
cd $W
log "running the second-level suite"
$P run_suite.py suites/intraday_full_second.json results/intraday_full_second.json --workers 16 >> $L/seconds_pipeline.log 2>&1
$P verify_v2.py results/intraday_full_second.json results/verify_intraday_full_second.json > results/verify_intraday_full_second.txt 2>&1
log "SECONDS_PIPELINE_DONE"
