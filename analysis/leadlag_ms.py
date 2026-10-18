"""
Millisecond lead-lag: how much of the Deribit perpetual's catch-up to Binance can a trader still capture
after a given reaction latency?

Leader: Binance USD-M perpetual (BTCUSDT / ETHUSDT aggTrades, ms timestamps, data.binance.vision).
Follower: Deribit perpetual (our tick files, ms timestamps). Sample fixed in advance: every 20th day from
2021-01-01; 2021-2023 vs 2024-2026 reported separately.
Event: Binance price moves at least X bp within W ms (X = 5, 10, 20; W = 100, 500, 1000), 5-s cool-down.
For latency L (0 ... 1000 ms) after the Binance event, a taker order on Deribit trades at Deribit's last price
at t+L and exits H seconds later (H = 1, 5, 30). Reported: gross capture in the event's direction, the share
of events with a positive capture, and net results at 10 bp (standard taker in and out) and 2 bp round trip.
Usage: python leadlag_ms.py download | analyse
"""
import io
import math
import os
import sys
import zipfile
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests

R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
B = "/mnt/quant_storage/quant_workspace/binance_aggtrades/"
OUT = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/"
DAYS = [datetime(2021, 1, 1) + timedelta(days=20 * i) for i in range(200)]
DAYS = [d for d in DAYS if d <= datetime(2026, 10, 15)]
LAT = (0, 50, 100, 200, 300, 500, 1000)
HOLD = (1, 5, 30)
GRID = [(w, x) for w in (100, 500, 1000) for x in (5, 10, 20)]


def download_one(args):
    sym, d = args
    path = B + f"{sym}-{d:%Y-%m-%d}.npz"
    if os.path.exists(path):
        return path
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{sym}/{sym}-aggTrades-{d:%Y-%m-%d}.zip"
    for _ in range(5):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.open(z.namelist()[0]).read()
            first = raw[:200].split(b"\n")[0]
            df = pd.read_csv(io.BytesIO(raw), header=0 if first[:1].isalpha() else None, usecols=[1, 5])
            df.columns = ["price", "ts"]
            ts = df.ts.values.astype(np.int64)
            ts = np.where(ts > 10 ** 14, ts // 1000, ts)            # microseconds in some newer files
            o = np.argsort(ts, kind="stable")
            np.savez(path, ts=ts[o], price=df.price.values[o].astype(np.float64))
            return path
        except Exception:
            continue
    return None


def deribit(cur, d):
    p = R + f"perp_trades/{cur}-PERPETUAL/{d:%Y%m%d}.parquet"
    if not os.path.exists(p):
        return None
    t = pd.read_parquet(p, columns=["ts", "price"]).sort_values("ts", kind="stable")
    return t.ts.values.astype(np.int64), t.price.values.astype(np.float64)


def day(args):
    cur, d = args
    bp = B + f"{cur}USDT-{d:%Y-%m-%d}.npz"
    if not os.path.exists(bp):
        return None
    z = np.load(bp)
    bts, bpx = z["ts"], z["price"]
    dr = deribit(cur, d)
    if dr is None or len(bts) < 1000:
        return None
    dts, dpx = dr
    day0 = int((d - datetime(1970, 1, 1)).total_seconds() * 1000)
    rows = []
    for (w, x) in GRID:
        j = np.searchsorted(bts, bts - w, side="left")           # first Binance trade inside the window
        r = (bpx / bpx[np.clip(j, 0, len(bpx) - 1)] - 1) * 1e4
        idx = np.flatnonzero((np.abs(r) >= x) & (bts > day0 + 2000) & (bts < day0 + 86_400_000 - 40_000))
        last = -10 ** 18
        for i in idx:
            t = bts[i]
            if t - last < 5000:
                continue
            last = t
            sgn = 1.0 if r[i] > 0 else -1.0
            k0 = np.searchsorted(dts, t, side="right") - 1         # Deribit price at the Binance event
            if k0 < 0:
                continue
            p0 = dpx[k0]
            rec = {"t": t, "w": w, "x": x, "sgn": sgn}
            for L in LAT:
                ke = np.searchsorted(dts, t + L, side="right") - 1
                pe = dpx[ke]
                rec[f"pre_{L}"] = sgn * (pe / p0 - 1) * 1e4            # already moved before we could trade
                for H in HOLD:
                    kx = np.searchsorted(dts, t + L + H * 1000, side="right") - 1
                    rec[f"cap_{L}_{H}"] = sgn * (dpx[kx] / pe - 1) * 1e4
            rows.append(rec)
    return pd.DataFrame(rows) if rows else None


def analyse():
    split = int(datetime(2024, 1, 1).timestamp() * 1000)
    for cur in ("BTC", "ETH"):
        with Pool(24) as p:
            parts = [x for x in p.map(day, [(cur, d) for d in DAYS]) if x is not None]
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(OUT + f"leadlag_ms_{cur}.parquet", index=False)
        print(f"\n=========== {cur}: {len(parts)} days ===========")
        for (w, x) in GRID:
            g0 = df[(df.w == w) & (df.x == x)]
            for lab, g in (("2021-23", g0[g0.t < split]), ("2024-26", g0[g0.t >= split])):
                if len(g) < 20:
                    continue
                print(f"Binance >= {x:2d} bp within {w:4d} ms, {lab}: {len(g):5d} events")
                for H in HOLD:
                    cells = []
                    for L in LAT:
                        c = g[f"cap_{L}_{H}"]
                        cells.append(f"{L:4d}ms {c.mean():+5.1f}")
                    print(f"   hold {H:2d}s gross capture (bp) by latency: " + " | ".join(cells))
                pre = " | ".join(f"{L}ms {g[f'pre_{L}'].mean():+5.1f}" for L in LAT)
                print(f"   Deribit move already done by the time of entry (bp): {pre}")
        print("costs: standard taker in+out ~10 bp; a 2 bp round trip needs top-tier/maker-rebate fees")


if __name__ == "__main__":
    if sys.argv[1] == "download":
        os.makedirs(B, exist_ok=True)
        jobs = [(s, d) for s in ("BTCUSDT", "ETHUSDT") for d in DAYS]
        with Pool(12) as p:
            res = p.map(download_one, jobs)
        print(f"downloaded {sum(r is not None for r in res)}/{len(jobs)}")
    else:
        analyse()
