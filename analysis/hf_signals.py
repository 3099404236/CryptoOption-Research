"""
Second-level signals on the Deribit perpetual from its tick trades (no LEAN): can anything predict the next
seconds/minutes by more than the trading cost? Taker fee 0.05 % per side, i.e. a round trip costs ~10 bp
(+ the bid/ask, ~0.1-2 bp); maker fee 0 % but fills are uncertain and adversely selected.

Per UTC day a 1-second grid: last trade price, taker buy / sell USD volume, index price.
Samples every 10 s (from 01:00 to 23:55 so look-backs/forwards stay inside the day):
  r10 r60 r300      past log return over 10 / 60 / 300 s (bp)
  tfi10 tfi60 tfi300 taker-flow imbalance (buy - sell) / (buy + sell) over 10 / 60 / 300 s
  vr60              USD volume of the last 60 s / average 60 s volume of the last hour
  basis             perpetual / index - 1 (bp)
  f10 f60 f300      forward log return over 10 / 60 / 300 s (bp)
Usage: python hf_signals.py build BTC 2019-01-01 2023-12-31   -> results/analysis/hf_BTC_2019-2023.parquet
       python hf_signals.py report <parquet>
"""
import glob
import os
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

R = os.environ.get("DERIBIT_RAW", "/mnt/quant_storage/quant_workspace/deribit_raw/")
OUT = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/"
N = 86_400


def day_samples(args):
    cur, d = args
    p = R + f"perp_trades/{cur}-PERPETUAL/{d:%Y%m%d}.parquet"
    if not os.path.exists(p):
        return None
    t = pd.read_parquet(p)
    if len(t) < 100:
        return None
    t0 = int((d - datetime(1970, 1, 1)).total_seconds() * 1000)
    sec = ((t.ts.values - t0) // 1000).clip(0, N - 1)
    price = np.full(N, np.nan)
    index = np.full(N, np.nan)
    order = np.argsort(t.ts.values, kind="stable")
    price[sec[order]] = t.price.values[order]          # last trade of each second wins
    index[sec[order]] = t["index"].values[order]
    price = pd.Series(price).ffill().bfill().values
    index = pd.Series(index).ffill().bfill().values
    buy = np.bincount(sec, weights=np.where(t.side.values > 0, t.amount.values, 0.0), minlength=N)
    sell = np.bincount(sec, weights=np.where(t.side.values < 0, t.amount.values, 0.0), minlength=N)
    cb, cs = np.concatenate([[0], np.cumsum(buy)]), np.concatenate([[0], np.cumsum(sell)])
    S = np.arange(3600, N - 300, 10)
    lp = np.log(price)
    out = {"t": t0 // 1000 + S}
    for L in (10, 60, 300):
        out[f"r{L}"] = (lp[S] - lp[S - L]) * 1e4
        b, s = cb[S + 1] - cb[S + 1 - L], cs[S + 1] - cs[S + 1 - L]
        tot = b + s
        out[f"tfi{L}"] = np.where(tot > 0, (b - s) / np.where(tot > 0, tot, 1), 0.0)
    v60 = (cb[S + 1] + cs[S + 1]) - (cb[S - 59] + cs[S - 59])
    v3600 = (cb[S + 1] + cs[S + 1]) - (cb[S - 3599] + cs[S - 3599])
    out["vr60"] = np.where(v3600 > 0, v60 / np.where(v3600 > 0, v3600 / 60.0, 1), 0.0)
    out["basis"] = (price[S] / index[S] - 1) * 1e4
    for H in (10, 60, 300):
        out[f"f{H}"] = (lp[S + H] - lp[S]) * 1e4
    return pd.DataFrame({k: (v.astype(np.float32) if k != "t" else v) for k, v in out.items()})


def build(cur, start, end):
    d0, d1 = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
    days = [(cur, d0 + timedelta(days=i)) for i in range((d1 - d0).days + 1)]
    with Pool(int(os.environ.get("WORKERS", "16"))) as pool:
        parts = [x for x in pool.imap(day_samples, days, chunksize=4) if x is not None]
    df = pd.concat(parts, ignore_index=True)
    path = OUT + f"hf_{cur}_{start[:4]}-{end[:4]}.parquet"
    df.to_parquet(path, index=False)
    print(f"{cur} {start}..{end}: {len(parts)} days, {len(df):,} samples -> {path}")


def report(path):
    df = pd.read_parquet(path)
    feats = ["r10", "r60", "r300", "tfi10", "tfi60", "tfi300", "vr60", "basis"]
    print(f"{path}: {len(df):,} samples, {df.t.min()}..{df.t.max()}")
    print("mean forward return (bp) by signal percentile; a taker round trip costs ~10 bp")
    for f in feats:
        x = df[f]
        qs = x.quantile([0.001, 0.01, 0.1, 0.9, 0.99, 0.999]).values
        rows = []
        for name, m in (("<0.1%", x <= qs[0]), ("<1%", x <= qs[1]), ("<10%", x <= qs[2]),
                        (">90%", x >= qs[3]), (">99%", x >= qs[4]), (">99.9%", x >= qs[5])):
            g = df.loc[m, ["f10", "f60", "f300"]]
            rows.append(f"{name}: " + " ".join(f"{g[c].mean():+6.2f}" for c in ("f10", "f60", "f300")))
        print(f"  {f:7s} | " + " | ".join(rows))
    # best linear combination, fitted in-sample (optimistic): how big are the most extreme predictions?
    X = np.column_stack([np.ones(len(df))] + [df[f].values.astype(float) for f in feats])
    for H in ("f10", "f60", "f300"):
        y = df[H].values.astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        r2 = 1 - np.var(y - pred) / np.var(y)
        hi, lo = np.quantile(pred, 0.999), np.quantile(pred, 0.001)
        top, bot = y[pred >= hi].mean(), y[pred <= lo].mean()
        print(f"  OLS {H}: R2 {r2:.4f}; realised mean when prediction in top 0.1% {top:+.2f} bp, bottom 0.1% {bot:+.2f} bp")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        report(sys.argv[2])
