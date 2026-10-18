"""
Non-option benchmark: perpetual funding carry (long spot, short Deribit perpetual, delta neutral).
The short perpetual receives the hourly funding when it is positive and pays when it is negative.

  always     hold the position the whole time
  trailing   hold only when the trailing 7-day average funding (known at the decision time) is positive;
             re-decided daily at 00:00 UTC
Costs: 0.05 % perpetual taker + 0.10 % spot fee per side at every switch; basis (perpetual vs index)
P&L at each switch from the hourly index in the funding file and the perpetual 1-minute close.
Capital = 1.2 x notional (spot plus a 20 % margin buffer for the short perpetual); no interest on cash.
In-sample = up to 2023, out-of-sample = 2024-2026 (no parameter was tuned).
"""
import glob
import os

import numpy as np
import pandas as pd

R = os.environ.get("DERIBIT_RAW", "/mnt/quant_storage/quant_workspace/deribit_raw/")
COST_SIDE = 0.0005 + 0.0010
CAPITAL = 1.2
RF = {2019: 0.021, 2020: 0.004, 2021: 0.001, 2022: 0.017, 2023: 0.050, 2024: 0.051, 2025: 0.042, 2026: 0.036}

for cur in ["BTC", "ETH"]:
    f = pd.read_csv(R + f"funding/{cur}-PERPETUAL.csv")
    f = f[f.interest_1h.notna()].copy()
    f["t"] = pd.to_datetime(f.ts, unit="ms")
    f = f.set_index("t").sort_index()
    perp = pd.concat([pd.read_parquet(p, columns=["ts", "close"]) for p in sorted(glob.glob(R + f"perp_1m/{cur}-PERPETUAL/*.parquet"))])
    perp = perp.drop_duplicates("ts").set_index("ts")["close"]
    f["perp"] = perp.reindex(f.ts - 60000).values
    f["basis"] = f.perp / f["index"] - 1
    hourly = f.interest_1h
    daily_fund = hourly.resample("D").sum()
    trail = daily_fund.rolling(7).mean().shift(1)           # known at 00:00 of each day
    out = {}
    for name in ("always", "trailing"):
        on = pd.Series(True, index=daily_fund.index) if name == "always" else (trail > 0)
        pnl = daily_fund.where(on, 0.0)
        switches = on.astype(int).diff().abs().fillna(on.iloc[0])
        basis_d = f.basis.resample("D").first().reindex(daily_fund.index).fillna(0.0)
        # entering short perp at basis b and leaving at b' earns (b - b'): entry +b, exit -b
        sign = on.astype(int).diff().fillna(on.iloc[0].astype(int))
        pnl = pnl + sign * basis_d - switches * COST_SIDE
        ret = pnl / CAPITAL
        out[name] = ret
    print(f"\n=== {cur} funding carry (return on capital) ===")
    print(f"{'year':6s} {'always':>8s} {'trailing':>9s} {'days on':>8s} {'T-bill':>7s}")
    for y in sorted(set(out['always'].index.year)):
        a = out["always"][out["always"].index.year == y].sum()
        b = out["trailing"][out["trailing"].index.year == y].sum()
        on_days = int((trail[trail.index.year == y] > 0).sum())
        print(f"{y:<6d} {a:+8.2%} {b:+9.2%} {on_days:8d} {RF.get(y, float('nan')):7.1%}")
    for name, r in out.items():
        for lab, sl in (("in-sample 2019-23", r[r.index < "2024-01-01"]), ("out-of-sample 2024-26", r[r.index >= "2024-01-01"])):
            yrs = len(sl) / 365.0
            ann = sl.sum() / yrs
            vol = sl.std() * np.sqrt(365)
            eq = sl.cumsum()
            dd = (eq.cummax() - eq).max()
            print(f"  {name:8s} {lab:22s}: {ann:+.2%}/yr, vol {vol:.2%}, Sharpe (raw) {ann / vol if vol else 0:.2f}, max drawdown {dd:.2%}")
