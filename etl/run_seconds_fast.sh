#!/bin/bash
# Second-level runs with the optimized intraday project (identical logic, ~2x faster) and the quote/spot pairing fix:
# full history for both coins, then the 5-day windows again with the same code; every result is verified.
W=/mnt/quant_storage/quant_workspace/lean_workspace_v2
L=/mnt/quant_storage/quant_workspace/logs
P=/opt/qenv/bin/python
export LEAN_BIN=/opt/qenv/bin/lean PATH=/opt/qenv/bin:$PATH
log() { echo "$(date -u '+%F %T') $*" >> $L/seconds_fast.log; }
cd $W
log "start: full history ETH + BTC (DeribitIntradayFast)"
for c in eth btc; do
  ( $P run_suite.py suites/intraday_full_second_fast_$c.json results/intraday_full_second_fast_$c.json --workers 10 >> $L/seconds_fast.log 2>&1
    $P verify_v2.py results/intraday_full_second_fast_$c.json results/verify_intraday_full_second_fast_$c.json > results/verify_intraday_full_second_fast_$c.txt 2>&1
    log "$c full history done and verified" ) &
done
sleep 600                                    # stagger: the short 5-day windows run alongside the long jobs
for c in eth btc; do
  $P run_suite.py suites/intraday_${c}_5d_fast.json results/intraday_${c}_5d_fast.json --workers 2 >> $L/seconds_fast.log 2>&1
  $P verify_v2.py results/intraday_${c}_5d_fast.json results/verify_intraday_${c}_5d_fast.json > results/verify_intraday_${c}_5d_fast.txt 2>&1
  log "$c 5-day window done and verified"
done
wait
log "SECONDS_FAST_DONE"
