#!/bin/bash
# Build one currency: underlying (full range) then options in yearly chunks, at most 3 in parallel.
# usage: build_currency.sh <BTC|ETH> <first-year> <end-date>
set -e
cd "$(dirname "$0")"
export DERIBIT_RAW=/mnt/quant_storage/quant_workspace/deribit_raw
P=/home/czf/miniconda3/bin/python
CUR=$1; Y0=$2; END=$3
L=$DERIBIT_RAW/logs
$P calibrate_spreads.py --currency $CUR > $L/calib_$CUR.out 2>&1
$P build_lean_data.py --currency $CUR --start $Y0-01-01 --end $END --underlying-only \
   --second-start 2026-10-10 --second-end $END > $L/build_${CUR}_underlying.out 2>&1
for y in $(seq $Y0 ${END:0:4}); do
  e="$y-12-31"; [ $y = ${END:0:4} ] && e=$END
  $P build_lean_data.py --currency $CUR --start $y-01-01 --end $e --skip-underlying > $L/build_${CUR}_$y.out 2>&1 &
  while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 5; done
done
wait
echo "BUILD_DONE $CUR"
