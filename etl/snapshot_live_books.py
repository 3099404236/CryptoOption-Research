"""
Snapshot of Deribit's live option books: median *quoted* half-spread (coin units) per moneyness /
days-to-expiry bucket. Several snapshots taken at different hours anchor the level of the historical
cost model (calibrate_spreads.py measures effective spreads from trades, which understate the cost
of crossing the book).

Usage: python snapshot_live_books.py   (writes $DERIBIT_RAW/calibration/live_quotes_<CUR>_<UTC>.json)
"""
import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from deribit_math import DTE_NAMES, OTM_NAMES, dte_bucket, otm_bucket

RAW = Path(os.environ.get("DERIBIT_RAW", "/mnt/quant_storage/quant_workspace/deribit_raw"))
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def main():
    proxy = os.environ.get("DERIBIT_PROXY")
    sess = requests.Session()
    if proxy:
        sess.proxies.update({"http": proxy, "https": proxy})
    now = datetime.now(timezone.utc)
    for cur in ("BTC", "ETH"):
        for attempt in range(6):
            try:
                rows = sess.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                                params={"currency": cur, "kind": "option"}, timeout=60).json()["result"]
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(3)
        cells = {}
        for r in rows:
            _, e, k, cp = r["instrument_name"].split("-")
            expiry = datetime(2000 + int(e[-2:]), MON[e[-5:-2]], int(e[:-5]), 8, tzinfo=timezone.utc)
            days = (expiry - now).total_seconds() / 86400
            b, a, und = r.get("bid_price"), r.get("ask_price"), r.get("underlying_price")
            if not (b and a and und) or days <= 0:
                continue
            o = int(otm_bucket(np.array([float(k)]), np.array([und]), np.array([cp == "C"]))[0])
            t = int(dte_bucket(np.array([days]))[0])
            cells.setdefault((o, t), []).append((a - b) / 2.0)
        out = {"currency": cur, "utc": now.isoformat(), "otm_buckets": OTM_NAMES, "dte_buckets": DTE_NAMES,
               "cells": {f"{OTM_NAMES[o]}|{DTE_NAMES[t]}": {"quoted_half": float(np.median(v)), "n": len(v)}
                         for (o, t), v in cells.items() if len(v) >= 3}}
        path = RAW / "calibration" / f"live_quotes_{cur}_{now:%Y%m%dT%H%M}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
        print(f"{cur}: {len(out['cells'])} cells -> {path.name}")


if __name__ == "__main__":
    main()
