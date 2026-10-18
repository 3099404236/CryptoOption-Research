"""
Bid/ask cost model for Deribit options.

Free historical *quotes* are not available (Tardis.dev's free samples are behind a bot challenge from
this network). Two trade-based estimators were tried and rejected (documented in the V2 report):
  * effective spread vs Deribit's mark price: after 2021 the mark stored with each trade sits almost on
    the trade price, so the estimate collapses to ~0;
  * bid-ask bounce (opposite-side prints within 60 s): the median gap is 0 (one large order sweeping
    the book prints many fills at the same price).
The model therefore uses the only direct observation of quoted spreads: Deribit's live order books,
snapshotted several times (snapshot_live_books.py). The median quoted half-spread per moneyness /
days-to-expiry bucket, in coin units, is applied to every year. Books were thinner in 2019-2021, so
this understates early-year costs (it favours the strategies); a cost multiplier is exposed for
sensitivity runs.

Usage: python calibrate_spreads.py --currency BTC
Output: $DERIBIT_RAW/calibration/spread_model_<CUR>.json  (half-spreads in coin units)
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from deribit_math import DTE_NAMES, OTM_NAMES

RAW = Path(os.environ.get("DERIBIT_RAW", "/mnt/quant_storage/quant_workspace/deribit_raw"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", required=True)
    a = ap.parse_args()
    snaps = sorted(glob.glob(str(RAW / "calibration" / f"live_quotes_{a.currency}_*.json")))
    if not snaps:
        raise SystemExit("no live book snapshots; run snapshot_live_books.py first")
    vals = {}
    for f in snaps:
        for key, c in json.loads(Path(f).read_text())["cells"].items():
            vals.setdefault(key, []).append(c["quoted_half"])
    table = {}
    for o in OTM_NAMES:
        for t in DTE_NAMES:
            v = vals.get(f"{o}|{t}")
            if v:
                table.setdefault(o, {})[t] = {"half_spread": float(np.median(v)), "snapshots": len(v)}
    # fill empty cells from the nearest expiry bucket of the same moneyness, then nearest moneyness
    for oi, o in enumerate(OTM_NAMES):
        for ti, t in enumerate(DTE_NAMES):
            if table.get(o, {}).get(t):
                continue
            for dt in (1, -1, 2, -2, 3, -3, 4, -4):
                c = table.get(o, {}).get(DTE_NAMES[min(max(ti + dt, 0), len(DTE_NAMES) - 1)])
                if c:
                    table.setdefault(o, {})[t] = {**c, "filled_from_neighbour": True}
                    break
            else:
                for do in (1, -1, 2, -2):
                    c = table.get(OTM_NAMES[min(max(oi + do, 0), len(OTM_NAMES) - 1)], {}).get(t)
                    if c:
                        table.setdefault(o, {})[t] = {**c, "filled_from_neighbour": True}
                        break
    model = {"currency": a.currency, "source": "Deribit live order books (median quoted half-spread)",
             "snapshots": [Path(f).name for f in snaps], "applies_to": "all years (coin units)",
             "otm_buckets": OTM_NAMES, "dte_buckets": DTE_NAMES, "table": {"all": table}}
    out = RAW / "calibration" / f"spread_model_{a.currency}.json"
    out.write_text(json.dumps(model, indent=1))
    print(f"{a.currency}: {len(snaps)} snapshots -> {out}")
    for t in DTE_NAMES:
        print(f"  {t:6s} " + " | ".join(f"{o}: {table.get(o, {}).get(t, {}).get('half_spread', float('nan')):.5f}"
                                        for o in OTM_NAMES))


if __name__ == "__main__":
    main()
