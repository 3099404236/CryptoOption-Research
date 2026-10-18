#!/bin/bash
# Rebuild the LEAN data on the research machine (A409) from the raw Deribit files copied from the
# workstation. Same code and the same spread model file (calibration is NOT re-run), so the output is
# identical to the workstation's data. Underlying and every option year run in parallel.
set -e
cd "$(dirname "$0")"
export DERIBIT_RAW=/mnt/quant_storage/quant_workspace/deribit_raw
P=/opt/qenv/bin/python
L=/mnt/quant_storage/quant_workspace/logs
mkdir -p $L
$P build_lean_data.py --currency BTC --start 2019-01-01 --end 2026-10-16 --underlying-only \
   --second-start 2026-10-10 --second-end 2026-10-16 > $L/build_BTC_underlying.out 2>&1 &
$P build_lean_data.py --currency ETH --start 2020-01-01 --end 2026-10-16 --underlying-only \
   --second-start 2026-10-10 --second-end 2026-10-16 > $L/build_ETH_underlying.out 2>&1 &
for CUR in BTC ETH; do
  Y0=2019; [ $CUR = ETH ] && Y0=2020
  for y in $(seq $Y0 2026); do
    e="$y-12-31"; [ $y = 2026 ] && e=2026-10-16
    $P build_lean_data.py --currency $CUR --start $y-01-01 --end $e --skip-underlying > $L/build_${CUR}_$y.out 2>&1 &
  done
done
wait
echo BUILD_DONE
