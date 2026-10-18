"""
Second-level perpetual test with MAKER entries (Deribit perpetual maker fee 0 %, taker 0.05 %).
Rule fixed before running it: every 10 s, the OLS forecast of the next-60-s return (same 8 features as
hf_signals.py, coefficients fitted on 2019-2023 only) is computed; when |forecast| is above its in-sample
99th percentile, a limit order is posted at the last trade price in the forecast direction.
  fill:  only if a later trade prints strictly THROUGH the limit within 30 s (no queue priority assumed)
  exit:  60 s after the fill, taker at the last trade price (0.05 % fee); half a tick of spread on the exit
For comparison the same signals traded as taker in and out (0.05 % + 0.05 %).
One position at a time. In-sample 2019-2023 (ETH from 2020), out-of-sample 2024-2026.
Usage: python hf_maker.py BTC
"""
import os
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
OUT = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/"
FEATS = ["r10", "r60", "r300", "tfi10", "tfi60", "tfi300", "vr60", "basis"]
N = 86_400
FEE = 5.0                   # bp, taker
TICK = {"BTC": 0.5, "ETH": 0.05}
G = {}


def day_trades(args):
    cur, d = args
    p = R + f"perp_trades/{cur}-PERPETUAL/{d:%Y%m%d}.parquet"
    if not os.path.exists(p):
        return []
    t = pd.read_parquet(p)
    if len(t) < 100:
        return []
    t0 = int((d - datetime(1970, 1, 1)).total_seconds() * 1000)
    sec = ((t.ts.values - t0) // 1000).clip(0, N - 1)
    order = np.argsort(t.ts.values, kind="stable")
    price = np.full(N, np.nan)
    index = np.full(N, np.nan)
    price[sec[order]] = t.price.values[order]
    index[sec[order]] = t["index"].values[order]
    lo = pd.Series(t.price.values).groupby(sec).min().reindex(range(N)).values      # NaN = no trade that second
    hi = pd.Series(t.price.values).groupby(sec).max().reindex(range(N)).values
    price = pd.Series(price).ffill().bfill().values
    index = pd.Series(index).ffill().bfill().values
    buy = np.bincount(sec, weights=np.where(t.side.values > 0, t.amount.values, 0.0), minlength=N)
    sell = np.bincount(sec, weights=np.where(t.side.values < 0, t.amount.values, 0.0), minlength=N)
    cb, cs = np.concatenate([[0], np.cumsum(buy)]), np.concatenate([[0], np.cumsum(sell)])
    S = np.arange(3600, N - 200, 10)
    lp = np.log(price)
    f = {}
    for L in (10, 60, 300):
        f[f"r{L}"] = (lp[S] - lp[S - L]) * 1e4
        b, s = cb[S + 1] - cb[S + 1 - L], cs[S + 1] - cs[S + 1 - L]
        tot = b + s
        f[f"tfi{L}"] = np.where(tot > 0, (b - s) / np.where(tot > 0, tot, 1), 0.0)
    v60 = (cb[S + 1] + cs[S + 1]) - (cb[S - 59] + cs[S - 59])
    v3600 = (cb[S + 1] + cs[S + 1]) - (cb[S - 3599] + cs[S - 3599])
    f["vr60"] = np.where(v3600 > 0, v60 / np.where(v3600 > 0, v3600 / 60.0, 1), 0.0)
    f["basis"] = (price[S] / index[S] - 1) * 1e4
    X = np.column_stack([np.ones(len(S))] + [f[k] for k in FEATS])
    pred = X @ G["beta"]
    out, busy = [], -1
    half_tick_bp = 0.5 * TICK[cur] / price.mean() * 1e4
    for j in np.flatnonzero(np.abs(pred) >= G["thr"]):
        s0 = S[j]
        if s0 <= busy:
            continue
        side = 1 if pred[j] > 0 else -1
        lim = price[s0]
        # taker version: in at s0, out at s0 + 60
        taker = side * (lp[s0 + 60] - lp[s0]) * 1e4 - 2 * FEE - 2 * half_tick_bp
        # maker version: first second in (s0, s0 + 30] that trades strictly through the limit
        w = lo[s0 + 1:s0 + 31] if side > 0 else hi[s0 + 1:s0 + 31]
        hit = np.flatnonzero((w < lim) if side > 0 else (w > lim))
        if len(hit):
            sf = s0 + 1 + hit[0]
            maker = side * (np.log(price[min(sf + 60, N - 1)]) - np.log(lim)) * 1e4 - FEE - half_tick_bp
            busy = sf + 60
        else:
            maker = np.nan                       # not filled, nothing happens
            busy = s0 + 30
        out.append((d.year, taker, maker))
    return out


def init(beta, thr):
    G["beta"], G["thr"] = beta, thr


def main(cur):
    is_df = pd.read_parquet(OUT + f"hf_{cur}_{'2019' if cur == 'BTC' else '2020'}-2023.parquet")
    X = np.column_stack([np.ones(len(is_df))] + [is_df[k].values.astype(float) for k in FEATS])
    beta, *_ = np.linalg.lstsq(X, is_df.f60.values.astype(float), rcond=None)
    thr = np.quantile(np.abs(X @ beta), 0.99)
    print(f"{cur}: OLS(60 s) fitted on in-sample; trade when |forecast| >= {thr:.2f} bp (in-sample 99th pct)")
    d0 = datetime(2019, 1, 1) if cur == "BTC" else datetime(2020, 1, 1)
    days = [(cur, d0 + timedelta(days=i)) for i in range((datetime(2026, 10, 16) - d0).days + 1)]
    with Pool(24, initializer=init, initargs=(beta, thr)) as pool:
        rows = [r for part in pool.imap(day_trades, days, chunksize=4) for r in part]
    df = pd.DataFrame(rows, columns=["year", "taker", "maker"])
    for lab, g in (("in-sample", df[df.year <= 2023]), ("out-of-sample", df[df.year >= 2024])):
        m = g.maker.dropna()
        tk = g.taker
        t_m = m.mean() / (m.std() / np.sqrt(len(m))) if len(m) > 2 else float("nan")
        t_t = tk.mean() / (tk.std() / np.sqrt(len(tk))) if len(tk) > 2 else float("nan")
        print(f"  {lab:13s}: signals {len(g)}, maker fills {len(m)} ({len(m) / max(len(g), 1):.0%}), "
              f"maker net {m.mean():+.2f} bp/trade (t {t_m:+.2f}) | taker net {tk.mean():+.2f} bp/trade (t {t_t:+.2f})")
        print("     maker net per year (bp/trade): " + " ".join(f"{y}:{v:+.1f}" for y, v in g.groupby("year").maker.mean().items()))


if __name__ == "__main__":
    main(sys.argv[1])
