"""
Second-level perpetual rules, written down before the full 2019-2023 sample was looked at (the design used
only the first half of 2019). Decisions every 10 s from hf_signals.py samples; one position at a time;
taker in and out at the last trade price, 0.05 % fee per side plus `slip` bp per side.

  JUMP     past-10-s return beyond the trailing 30-day 99.9 % / 0.1 % level -> trade in its direction, 300 s
  BASIS    perpetual/index basis beyond the trailing 30-day 99.9 % / 0.1 % level -> trade towards the index, 300 s
  OLS      OLS forecast of the 300-s return from all features, fitted on 2019-2023 only; trade when the
           forecast exceeds the costs by 5 bp, 300 s
Thresholds use only earlier days (median of the previous 30 days' daily quantiles).
In-sample = 2019-2023 (ETH from 2020), out-of-sample = 2024-2026; the OOS rows are printed once, unchanged rules.
Usage: python hf_rules.py BTC   (needs results/analysis/hf_<CUR>_<years>.parquet from hf_signals.py)
"""
import glob
import sys

import numpy as np
import pandas as pd

OUT = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/"
FEE_BP = 5.0
FEATS = ["r10", "r60", "r300", "tfi10", "tfi60", "tfi300", "vr60", "basis"]


def trailing_thresholds(df, col, lo=0.001, hi=0.999, days=30):
    day = df.t // 86400
    q = df.groupby(day)[col].quantile([lo, hi]).unstack()
    q = q.rolling(days, min_periods=10).median().shift(1)          # previous 30 days only
    return day.map(q[lo]).values, day.map(q[hi]).values


def simulate(df, signal, hold=300, slip=1.0):
    """signal: +1 / -1 / 0 per sample (decided at sample time). Returns per-trade net bp and times."""
    t, f = df.t.values, df[f"f{hold}"].values
    nets, times, busy_until = [], [], -1
    for i in np.flatnonzero(signal != 0):
        if t[i] < busy_until:
            continue
        nets.append(signal[i] * f[i] - 2 * (FEE_BP + slip))
        times.append(t[i])
        busy_until = t[i] + hold
    return np.array(nets), np.array(times)


def summary(label, nets, times):
    if len(nets) == 0:
        return f"{label}: no trades"
    yrs = pd.Series(nets, index=pd.to_datetime(times, unit="s")).groupby(lambda x: x.year).agg(["count", "sum"])
    t = nets.mean() / (nets.std() / np.sqrt(len(nets))) if len(nets) > 1 and nets.std() > 0 else 0.0
    by_year = " ".join(f"{y}:{int(r['count'])}/{r['sum'] / 100:+.1f}%" for y, r in yrs.iterrows())
    return (f"{label}: {len(nets)} trades, mean net {nets.mean():+.2f} bp/trade (t {t:+.2f}), "
            f"total {nets.sum() / 100:+.1f}% of notional | per year trades/total: {by_year}")


def main(cur):
    files = sorted(glob.glob(OUT + f"hf_{cur}_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("t").drop_duplicates("t").reset_index(drop=True)
    split = pd.Timestamp("2024-01-01").timestamp()
    is_ = df.t < split
    print(f"{cur}: {len(df):,} samples from {files}")
    lo, hi = trailing_thresholds(df, "r10")
    jump = np.where(df.r10.values >= hi, 1, np.where(df.r10.values <= lo, -1, 0))
    blo, bhi = trailing_thresholds(df, "basis")
    basis = np.where(df.basis.values <= blo, 1, np.where(df.basis.values >= bhi, -1, 0))
    X = np.column_stack([np.ones(len(df))] + [df[f].values.astype(float) for f in FEATS])
    beta, *_ = np.linalg.lstsq(X[is_.values], df.f300.values[is_.values].astype(float), rcond=None)
    pred = X @ beta
    for slip in (0.0, 1.0, 2.0):
        need = 2 * (FEE_BP + slip) + 5.0
        ols = np.where(pred >= need, 1, np.where(pred <= -need, -1, 0))
        print(f"\n-- slippage {slip:.0f} bp per side (round trip {2 * (FEE_BP + slip):.0f} bp)")
        for name, sig in (("JUMP", jump), ("BASIS", basis), ("OLS", ols)):
            for lab, m in (("in-sample", is_.values), ("out-of-sample", ~is_.values)):
                s = np.where(m, sig, 0)
                nets, times = simulate(df, s, slip=slip)
                print("  " + summary(f"{name:5s} {lab:13s}", nets, times))


if __name__ == "__main__":
    main(sys.argv[1])
