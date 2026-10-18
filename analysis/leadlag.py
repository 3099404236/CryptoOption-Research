"""
Cross-exchange lead-lag at the 1-second scale: does Binance spot lead the Deribit perpetual by enough, for
long enough, that a retail trader (1 s late) can profit after costs?

Sample fixed in advance: every 10th day from 2021-01-01 to 2026-10-15; in-sample = 2021-2023 days,
out-of-sample = 2024-2026 days. Binance spot 1-second klines (data.binance.vision, close per second) and
the Deribit perpetual's last trade per second (our tick files).
Signal at second t: gap_L = Binance log-return over the last L seconds minus Deribit's over the same seconds
(L = 1, 2, 5). Event when |gap_L| exceeds its in-sample 99.9th percentile (one event per 60 s at most).
Trade: taker on Deribit at the price of second t+1 in the gap's direction, out at t+1+H (H = 5, 15, 30, 60),
0.05 % fee each way (~10 bp round trip).
Usage: python leadlag.py download | analyse
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
B = "/mnt/quant_storage/quant_workspace/binance_1s/"
OUT = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/"
N = 86_400
DAYS = [datetime(2021, 1, 1) + timedelta(days=10 * i) for i in range(10_000)]
DAYS = [d for d in DAYS if d <= datetime(2026, 10, 15)]


def download_one(args):
    sym, d = args
    path = B + f"{sym}-1s-{d:%Y-%m-%d}.npy"
    if os.path.exists(path):
        return path
    url = f"https://data.binance.vision/data/spot/daily/klines/{sym}/1s/{sym}-1s-{d:%Y-%m-%d}.zip"
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4])
            t = df[0].values
            t = np.where(t > 1e14, t // 1000, t)              # newer files use microseconds
            sec = ((t - int((d - datetime(1970, 1, 1)).total_seconds() * 1000)) // 1000).astype(int)
            close = np.full(N, np.nan)
            ok = (sec >= 0) & (sec < N)
            close[sec[ok]] = df[4].values[ok]
            np.save(path, pd.Series(close).ffill().bfill().values.astype(np.float64))
            return path
        except Exception:
            continue
    return None


def deribit_seconds(cur, d):
    p = R + f"perp_trades/{cur}-PERPETUAL/{d:%Y%m%d}.parquet"
    if not os.path.exists(p):
        return None
    t = pd.read_parquet(p, columns=["ts", "price"])
    t0 = int((d - datetime(1970, 1, 1)).total_seconds() * 1000)
    sec = ((t.ts.values - t0) // 1000).clip(0, N - 1)
    o = np.argsort(t.ts.values, kind="stable")
    last = np.full(N, np.nan)
    last[sec[o]] = t.price.values[o]
    return pd.Series(last).ffill().bfill().values


def day_rows(args):
    cur, d = args
    sym = f"{cur}USDT"
    bp = B + f"{sym}-1s-{d:%Y-%m-%d}.npy"
    if not os.path.exists(bp):
        return None
    b = np.log(np.load(bp))
    dd = deribit_seconds(cur, d)
    if dd is None:
        return None
    dl = np.log(dd)
    s = np.arange(10, N - 62)
    out = {"t": int((d - datetime(1970, 1, 1)).total_seconds()) + s}
    for L in (1, 2, 5):
        out[f"gap{L}"] = ((b[s] - b[s - L]) - (dl[s] - dl[s - L])) * 1e4
    for H in (5, 15, 30, 60):
        out[f"f{H}"] = (dl[s + 1 + H] - dl[s + 1]) * 1e4
    out["move_t_to_t1"] = (dl[s + 1] - dl[s]) * 1e4            # what Deribit did before we could act
    return pd.DataFrame({k: (v.astype(np.float32) if k != "t" else v) for k, v in out.items()})


def analyse():
    split = datetime(2024, 1, 1).timestamp()
    for cur in ("BTC", "ETH"):
        with Pool(24) as p:
            parts = [x for x in p.map(day_rows, [(cur, d) for d in DAYS]) if x is not None]
        df = pd.concat(parts, ignore_index=True)
        is_ = df.t < split
        print(f"\n=== {cur}: {len(parts)} days ({is_.sum() // 86000} in-sample, {(~is_).sum() // 86000} out-of-sample)")
        for L in (1, 2, 5):
            g = df[f"gap{L}"]
            thr = np.quantile(g[is_].abs(), 0.999)
            for lab, m in (("in-sample", is_), ("out-of-sample", ~is_)):
                e = df[m & (g.abs() >= thr)]
                e = e[e.t.diff().fillna(999) > 60]
                sgn = np.sign(e[f"gap{L}"].values)
                cells = []
                for H in (5, 15, 30, 60):
                    x = sgn * e[f"f{H}"].values
                    t = x.mean() / (x.std() / math.sqrt(len(x))) if len(x) > 2 else float("nan")
                    cells.append(f"{H}s {x.mean():+6.2f}bp (t {t:+.1f})")
                catch = (sgn * e.move_t_to_t1.values).mean()
                print(f"  L={L}s |gap|>={thr:5.1f}bp {lab:13s} {len(e):5d} events | already caught up by t+1: {catch:+6.2f}bp | after t+1: " + " | ".join(cells))
        print("  a taker round trip costs ~10 bp")


if __name__ == "__main__":
    if sys.argv[1] == "download":
        os.makedirs(B, exist_ok=True)
        jobs = [(s, d) for s in ("BTCUSDT", "ETHUSDT") for d in DAYS]
        with Pool(12) as p:
            res = p.map(download_one, jobs)
        print(f"downloaded {sum(r is not None for r in res)}/{len(jobs)} files")
    else:
        analyse()
