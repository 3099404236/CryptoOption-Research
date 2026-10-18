"""
1-second LEAN bars of the real Deribit perpetual from its tick trades, for any date range
(same bar and quote construction as build_lean_data.py, which only did this for the 5-day window).
Run one process per year in parallel.

Usage: python build_seconds.py --currency BTC --start 2019-01-01 --end 2019-12-31
"""
import argparse
from datetime import datetime, timedelta

import pandas as pd

from build_lean_data import PERP_HALF_SPREAD_BP, RAW, day_ms, sc, write_zip


def build_day(cur, d, base):
    low = cur.lower()
    p = RAW / "perp_trades" / f"{cur}-PERPETUAL" / f"{d:%Y%m%d}.parquet"
    if not p.exists():
        return f"{cur} {d:%Y-%m-%d} MISSING trades"
    t = pd.read_parquet(p, columns=["ts", "price", "amount"]).sort_values("ts", kind="stable")
    if t.empty:
        return f"{cur} {d:%Y-%m-%d} EMPTY"
    t0 = day_ms(d)
    t["sec"] = (t.ts - t0) // 1000
    g = t.groupby("sec").agg(o=("price", "first"), h=("price", "max"), l=("price", "min"),
                             c=("price", "last"), v=("amount", "sum"))
    ms = g.index.values * 1000
    O, H, L, C = sc(g.o), sc(g.h), sc(g.l), sc(g.c)
    half = g.c.values * PERP_HALF_SPREAD_BP / 10000.0
    B, A = sc(g.c.values - half), sc(g.c.values + half)
    ds = f"{d:%Y%m%d}"
    write_zip(base / "second" / low / f"{ds}_trade.zip", {f"{ds}_{low}_second_trade.csv": "\n".join(
        f"{m},{a},{b},{c},{e},{vv:.4f}" for m, a, b, c, e, vv in zip(ms, O, H, L, C, g.v.values)) + "\n"})
    write_zip(base / "second" / low / f"{ds}_quote.zip", {f"{ds}_{low}_second_quote.csv": "\n".join(
        f"{m},{b},{b},{b},{b},10,{a},{a},{a},{a},10" for m, b, a in zip(ms, B, A)) + "\n"})
    return f"{cur} {ds} {len(t)} trades -> {len(g)} seconds"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--ws", default="/mnt/quant_storage/quant_workspace/lean_workspace_v2")
    a = ap.parse_args()
    from pathlib import Path
    base = Path(a.ws) / "data" / "equity" / "usa"
    d, e = datetime.strptime(a.start, "%Y-%m-%d"), datetime.strptime(a.end, "%Y-%m-%d")
    while d <= e:
        print(build_day(a.currency, d, base), flush=True)
        d += timedelta(days=1)
    print("SECONDS_DONE", a.currency, a.start, a.end)
