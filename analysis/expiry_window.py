"""Does the perpetual behave differently around Deribit's Friday 08:00 settlement (index TWAP 07:30-08:00)?
Windows: 07:00-07:30, 07:30-08:00, 08:00-08:05, 08:00-08:30. Fridays (weekly expiry; last Friday of the month =
monthly/quarterly) vs Monday-Thursday at the same clock times. Mean return, |return| and t-stat per window."""
import glob, math
import numpy as np, pandas as pd
R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
WIN = [("07:00", "07:30"), ("07:30", "08:00"), ("08:00", "08:05"), ("08:00", "08:30")]
for cur in ("BTC", "ETH"):
    p = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in sorted(glob.glob(R + f"perp_1m/{cur}-PERPETUAL/*.parquet"))])
    p = p.drop_duplicates("ts").set_index(pd.to_datetime(p.drop_duplicates("ts").ts, unit="ms"))["close"].sort_index()
    days = pd.date_range("2019-01-01" if cur == "BTC" else "2020-01-01", "2026-10-15", freq="D")
    rows = []
    for d in days:
        for a, b in WIN:
            ta, tb = pd.Timestamp(f"{d:%Y-%m-%d} {a}") - pd.Timedelta(minutes=1), pd.Timestamp(f"{d:%Y-%m-%d} {b}") - pd.Timedelta(minutes=1)
            if ta in p.index and tb in p.index:
                monthly = d.weekday() == 4 and (d + pd.Timedelta(days=7)).month != d.month
                rows.append((d, f"{a}-{b}", d.weekday(), monthly, math.log(p[tb] / p[ta]) * 1e4))
    df = pd.DataFrame(rows, columns=["day", "win", "wd", "monthly", "r"])
    print(f"\n=== {cur} (bp; Fri = weekly expiry, M = last Friday of month)")
    for w, g in df.groupby("win", sort=False):
        out = []
        for lab, x in (("Mon-Thu", g[g.wd < 4].r), ("Fri", g[g.wd == 4].r), ("Fri-M", g[g.monthly].r)):
            t = x.mean() / (x.std() / math.sqrt(len(x)))
            out.append(f"{lab}: mean {x.mean():+5.2f} (t {t:+.1f}) |r| {x.abs().mean():5.1f} n {len(x)}")
        print(f"  {w}: " + " | ".join(out))
