#!/bin/bash
# Re-download a day that was only partially available when first fetched (default 2026-10-16).
# Run after that UTC day has ended. Option trades, perpetual tick trades, the perpetual 1-minute month
# file, settlement prices and funding are refreshed.
DAY=${1:-2026-10-16}
D=${DAY//-/}
export DERIBIT_RAW=/mnt/quant_storage/quant_workspace/deribit_raw
export DERIBIT_PROXY=http://127.0.0.1:7897
P=/home/czf/miniconda3/bin/python
cd "$(dirname "$0")"
for c in BTC ETH; do
  rm -f "$DERIBIT_RAW/options/$c/${D:0:4}/$D.parquet" "$DERIBIT_RAW/perp_trades/$c-PERPETUAL/$D.parquet"
  $P fetch_deribit.py options --currency $c --start "$DAY" --end "$DAY" --workers 1
  $P fetch_deribit.py perptrades --currency $c --start "$DAY" --end "$DAY" --workers 1
  rm -f "$DERIBIT_RAW/perp_1m/$c-PERPETUAL/${D:0:6}.parquet"
  $P fetch_deribit.py perp1m --currency $c --start "${DAY:0:7}-01" --end "$DAY"
  $P fetch_deribit.py delivery --currency $c
  $P fetch_deribit.py funding --currency $c --start 2018-12-01 --end "$DAY"
done
echo "REFRESH_DONE $DAY"
