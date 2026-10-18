#!/bin/bash
# Full V2 LEAN data build: underlying (full range, one process) then options in yearly chunks (parallel).
# Each chunk replays the 55 days before its start to rebuild the volatility state, so chunks are independent.
set -e
cd "$(dirname "$0")"
export DERIBIT_RAW=/mnt/quant_storage/quant_workspace/deribit_raw
P=/home/czf/miniconda3/bin/python
END=${END:-2026-10-16}
L=$DERIBIT_RAW/logs
$P calibrate_spreads.py --currency BTC
$P calibrate_spreads.py --currency ETH
$P build_lean_data.py --currency BTC --start 2019-01-01 --end $END --underlying-only --second-start 2026-10-10 --second-end $END > $L/build_btc_underlying.out 2>&1
$P build_lean_data.py --currency ETH --start 2020-01-01 --end $END --underlying-only --second-start 2026-10-10 --second-end $END > $L/build_eth_underlying.out 2>&1
run_chunk() { $P build_lean_data.py --currency $1 --start $2 --end $3 --skip-underlying > $L/build_$1_$2.out 2>&1; }
pids=()
for y in 2019 2020 2021 2022 2023 2024 2025 2026; do
  e="$y-12-31"; [ $y = 2026 ] && e=$END
  run_chunk BTC $y-01-01 $e & pids+=($!)
  if [ $y -ge 2020 ]; then run_chunk ETH $y-01-01 $e & pids+=($!); fi
  while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 5; done
done
wait
echo BUILD_ALL_DONE
