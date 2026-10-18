"""
Second-level EVENT studies on the Deribit perpetual (tick data), with a retail latency of 1 second.
Rules fixed before looking at results; thresholds come from 2019-2023 (ETH 2020-2023) only, and
2024-2026 is the out-of-sample check.

A) Option-flow hedging pressure
   Each option print gets a Black-76 delta (Deribit trade IV, index as forward). Per second, net TAKER delta
   = sum(side x delta x contracts). A dealer who sold to that taker hedges the opposite way, i.e. in the
   taker's delta direction. Event: |net taker delta in USD| above its in-sample 99.9th percentile (seconds
   with option prints). Response: perpetual move from t+1 s to t+H in the taker's delta direction.
   Trade: taker in at t+1 s, taker out at t+H (0.05 % + 0.05 %).

B) Crash / spike reversal with limit orders ("catch the wick")
   Event: 60-second perpetual return below its in-sample 0.05th percentile (crash) or above the 99.95th
   (spike); 10-minute cool-down. One second later a limit order is posted d below (crash) / above (spike) the
   last price, valid 60 s, filled only if a later trade prints strictly through it. Exit H seconds after the
   fill, taker (0.05 %). Maker entry fee 0 %.

Usage: python hf_events.py build BTC | report BTC
"""
import math
import os
import re
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
OUT = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/"
MON = {m: i + 1 for i, m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
RX = re.compile(r"^(BTC|ETH)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:\.\d+)?)-([CP])$")
N = 86_400
HS = (5, 15, 30, 60, 300)
FEE = 5.0


def perp_seconds(cur, d):
    p = R + f"perp_trades/{cur}-PERPETUAL/{d:%Y%m%d}.parquet"
    if not os.path.exists(p):
        return None
    t = pd.read_parquet(p, columns=["ts", "price"])
    if len(t) < 100:
        return None
    t0 = int((d - datetime(1970, 1, 1)).total_seconds() * 1000)
    sec = ((t.ts.values - t0) // 1000).clip(0, N - 1)
    order = np.argsort(t.ts.values, kind="stable")
    last = np.full(N, np.nan)
    last[sec[order]] = t.price.values[order]
    s = pd.Series(t.price.values)
    lo = s.groupby(sec).min().reindex(range(N)).values
    hi = s.groupby(sec).max().reindex(range(N)).values
    return t0, pd.Series(last).ffill().bfill().values, lo, hi


def option_net_delta(cur, d, t0):
    p = R + f"options/{cur}/{d.year}/{d:%Y%m%d}.parquet"
    if not os.path.exists(p):
        return None
    o = pd.read_parquet(p, columns=["ts", "instrument", "amount", "side", "index", "iv"])
    m = o.instrument.str.extract(RX)
    o = o[m[0].notna()].copy()
    m = m[m[0].notna()]
    exp = pd.to_datetime(m[3].astype(int).add(2000).astype(str) + "-" + m[2].map(MON).astype(str) + "-" + m[1], format="%Y-%m-%d") + pd.Timedelta(hours=8)
    T = (exp.values.astype("datetime64[ms]").astype(np.int64) - o.ts.values) / (365 * 86400e3)
    K = m[4].astype(float).values
    F, sig = o["index"].values, o.iv.values / 100.0
    ok = (T > 0) & (sig > 0) & (F > 0)
    d1 = np.where(ok, (np.log(np.where(ok, F, 1) / K) + 0.5 * sig ** 2 * T) / np.where(ok, sig * np.sqrt(np.clip(T, 1e-12, None)), 1), 0)
    delta = norm.cdf(d1) - (m[5].values == "P")
    net = np.where(ok, o.side.values * delta * o.amount.values * F, 0.0)          # USD delta bought by takers
    sec = ((o.ts.values - t0) // 1000).clip(0, N - 1)
    return np.bincount(sec, weights=net, minlength=N)


def day(args):
    cur, d = args
    ps = perp_seconds(cur, d)
    if ps is None:
        return None
    t0, last, lo, hi = ps
    lp = np.log(last)
    out = {}
    # A) option flow: keep every second with option prints, plus forward responses
    nd = option_net_delta(cur, d, t0)
    if nd is not None:
        s = np.flatnonzero(nd != 0)
        s = s[(s >= 60) & (s < N - 301)]
        rec = {"t": t0 // 1000 + s, "usd": nd[s], "pre60": (lp[s] - lp[s - 60]) * 1e4}
        for H in HS:
            rec[f"f{H}"] = (lp[s + 1 + H] - lp[s + 1]) * 1e4        # from t+1 s (latency) to t+1+H
        out["flow"] = pd.DataFrame(rec)
    # B) per-second 60-s returns (for thresholds) and candidate extreme seconds with fill simulation inputs
    r60 = np.full(N, np.nan)
    r60[60:] = (lp[60:] - lp[:-60]) * 1e4
    out["r60q"] = np.nanpercentile(r60, [0.05, 99.95]) if np.isfinite(r60).any() else None
    cand = np.flatnonzero((r60 <= -60) | (r60 >= 60))           # anything beyond 0.6 % in 60 s, filtered later
    cand = cand[cand < N - 1300]
    rows = []
    for s in cand:
        rows.append((t0 // 1000 + s, r60[s], last[s], lo[s + 1:s + 61], hi[s + 1:s + 61], s))
    if rows:
        ext = []
        for (t, r, p, l60, h60, s) in rows:
            ext.append({"t": t, "r60": r, "p": p, "s": s,
                        "lo": np.nanmin(np.where(np.isnan(l60), np.inf, l60)), "hi": np.nanmax(np.where(np.isnan(h60), -np.inf, h60)),
                        # first second (1..60) at which each depth is traded through, and prices H s later
                        **{f"fill_{k}": first_through(l60, h60, p, r, k) for k in (0, 25, 50, 100)},
                        "path": last[s:s + 1300].astype(np.float32)})
        out["ext"] = ext
    return cur, d, out


def first_through(l60, h60, p, r, k_bp):
    """crash (r < 0): buy limit p*(1-k); spike: sell limit p*(1+k). Returns 1-based second of the first strict trade-through."""
    if r < 0:
        lim = p * (1 - k_bp / 1e4)
        hit = np.flatnonzero(np.nan_to_num(l60, nan=np.inf) < lim)
    else:
        lim = p * (1 + k_bp / 1e4)
        hit = np.flatnonzero(np.nan_to_num(h60, nan=-np.inf) > lim)
    return int(hit[0]) + 1 if len(hit) else -1


def build(cur):
    d0 = datetime(2019, 1, 1) if cur == "BTC" else datetime(2020, 1, 1)
    days = [(cur, d0 + timedelta(days=i)) for i in range((datetime(2026, 10, 16) - d0).days + 1)]
    flows, exts, qs = [], [], []
    with Pool(24) as pool:
        for res in pool.imap(day, days, chunksize=2):
            if res is None:
                continue
            _, d, out = res
            if "flow" in out:
                flows.append(out["flow"])
            exts.extend(out.get("ext", []))
    pd.concat(flows, ignore_index=True).to_parquet(OUT + f"hfev_flow_{cur}.parquet", index=False)
    pd.to_pickle(exts, OUT + f"hfev_ext_{cur}.pkl")
    print(f"{cur}: {sum(len(f) for f in flows):,} option-print seconds, {len(exts):,} extreme-move seconds")


def report(cur):
    split = pd.Timestamp("2024-01-01").timestamp()
    f = pd.read_parquet(OUT + f"hfev_flow_{cur}.parquet")
    is_ = f.t < split
    thr = np.quantile(f.usd[is_].abs(), 0.999)
    print(f"\n=== {cur} A) option-flow hedging pressure: event = |net taker delta| >= ${thr:,.0f} (in-sample 99.9th pct)")
    for lab, m in (("in-sample", is_), ("out-of-sample", ~is_)):
        e = f[m & (f.usd.abs() >= thr)]
        # one event per minute at most (first second), signed by the taker's delta direction
        e = e[e.t.diff().fillna(999) > 60]
        sgn = np.sign(e.usd.values)
        cells = []
        for H in HS:
            x = sgn * e[f"f{H}"].values
            t = x.mean() / (x.std() / math.sqrt(len(x))) if len(x) > 2 else float("nan")
            cells.append(f"{H}s {x.mean():+6.2f}bp (t {t:+.1f})")
        pre = (sgn * e.pre60.values).mean()
        print(f"  {lab:13s} {len(e):5d} events | signed move from t+1: " + " | ".join(cells) + f" | before (t-60..t): {pre:+.2f}bp")
    print("  taker round trip costs ~10 bp; a tradable effect needs a signed move clearly above that")

    ext = pd.DataFrame([{k: v for k, v in e.items() if k != "path"} for e in pd.read_pickle(OUT + f"hfev_ext_{cur}.pkl")])
    paths = [e["path"] for e in pd.read_pickle(OUT + f"hfev_ext_{cur}.pkl")]
    ext["path_i"] = range(len(ext))
    isx = ext.t < split
    # pre-registered threshold: the 0.05th / 99.95th percentile of the 60-s return over ALL in-sample seconds.
    # The candidate set only holds seconds with |r60| >= 60 bp, so take the k-th most extreme candidate, where
    # k = 0.05 % of all in-sample seconds (the tails are far beyond 60 bp, so they lie inside the candidate set).
    n_is_seconds = ext[isx].t.floordiv(86400).nunique() * 0 + len(pd.date_range(
        "2019-01-01" if cur == "BTC" else "2020-01-01", "2023-12-31")) * (86400 - 60)
    k = int(0.0005 * n_is_seconds)
    neg, pos = np.sort(ext.r60[isx & (ext.r60 < 0)].values), np.sort(ext.r60[isx & (ext.r60 > 0)].values)
    lo_q = neg[k - 1] if len(neg) >= k else -np.inf
    hi_q = pos[-k] if len(pos) >= k else np.inf
    print(f"\n=== {cur} B) crash/spike reversal: event = 60-s return <= {lo_q:.0f} bp or >= {hi_q:.0f} bp, 10-min cool-down")
    for lab, m in (("in-sample", isx), ("out-of-sample", ~isx)):
        ev = ext[m & ((ext.r60 <= lo_q) | (ext.r60 >= hi_q))].sort_values("t")
        ev = ev[ev.t.diff().fillna(1e9) > 600]
        print(f"  {lab}: {len(ev)} events ({(ev.r60 < 0).sum()} crashes, {(ev.r60 > 0).sum()} spikes)")
        for k in (0, 25, 50, 100):
            for H in (60, 300, 900):
                nets = []
                for _, row in ev.iterrows():
                    fs = row[f"fill_{k}"]
                    if fs < 0:
                        continue
                    path = paths[int(row.path_i)]
                    lim = row.p * (1 - k / 1e4) if row.r60 < 0 else row.p * (1 + k / 1e4)
                    exit_px = float(path[min(int(fs) + H, len(path) - 1)])
                    side = 1 if row.r60 < 0 else -1
                    nets.append(side * (exit_px / lim - 1) * 1e4 - FEE)
                nets = np.array(nets)
                if len(nets) < 3:
                    continue
                t = nets.mean() / (nets.std() / math.sqrt(len(nets)))
                print(f"     limit {k:3d}bp away, exit {H:4d}s: fills {len(nets):4d}/{len(ev)}, net {nets.mean():+7.2f} bp/trade (t {t:+.2f}), "
                      f"median {np.median(nets):+.1f}, total {nets.sum() / 100:+.1f}% of notional")


if __name__ == "__main__":
    {"build": build, "report": report}[sys.argv[1]](sys.argv[2])
