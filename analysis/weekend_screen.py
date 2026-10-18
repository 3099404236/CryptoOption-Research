"""
Weekend effect test (raw Deribit prints, no LEAN): sell the ATM straddle of the next-Friday expiry on Friday
09:00-11:00 UTC and buy it back on Monday 07:00-11:00 UTC, unhedged or delta-hedged daily at 08:00 with the
perpetual. Leg prices = Black-76 at the perpetual price with the IV implied by Deribit's mark at the leg's
first print in each window; sell at mid - half-spread, buy back at mid + half-spread (live-book model),
option fee 0.03 % of spot per leg per side, perpetual taker fee 0.05 %.
For comparison the same straddle held to the Friday 08:00 settlement is computed on the same weeks.
In-sample 2019-2023, out-of-sample 2024-2026 (no parameters to tune).
"""
import glob
import json
import math
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
MON = {m: i + 1 for i, m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
RX = r"^(BTC|ETH)-\d{1,2}[A-Z]{3}\d{2}-\d+(\.\d+)?-[CP]$"
EPOCH = datetime(1970, 1, 1)
YEAR = 365 * 86400.0
OTM = [(-9, -0.03, "ITM>3%"), (-0.03, -0.01, "ITM1-3%"), (-0.01, 0.01, "ATM"), (0.01, 0.03, "OTM1-3%"), (0.03, 0.06, "OTM3-6%"),
       (0.06, 0.10, "OTM6-10%"), (0.10, 0.20, "OTM10-20%"), (0.20, 9, "OTM>20%")]
DTE = [(0, 1, "<1d"), (1, 3, "1-3d"), (3, 8, "3-8d"), (8, 21, "8-21d"), (21, 9999, ">21d")]
ms = lambda dt: int((dt - EPOCH).total_seconds() * 1000)


def b76(F, K, T, s, call):
    v = s * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    c = F * norm.cdf(d1) - K * norm.cdf(d1 - v)
    return c if call else c - (F - K)


def delta(F, K, T, s, call):
    v = s * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    return norm.cdf(d1) - (0 if call else 1)


def iv_of(px, F, K, T, call):
    lo, hi = b76(F, K, T, 0.02, call), b76(F, K, T, 5.0, call)
    return brentq(lambda s: b76(F, K, T, s, call) - px, 0.02, 5.0) if lo < px < hi else None


def parse(inst):
    _, d, k, cp = inst.split("-")
    m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})$", d)
    return datetime(2000 + int(m.group(3)), MON[m.group(2)], int(m.group(1)), 8), float(k), cp == "C"


def prints(cur, day, t0, t1):
    try:
        t = pd.read_parquet(R + f"options/{cur}/{day.year}/{day:%Y%m%d}.parquet")
    except Exception:
        return None
    t = t[(t.ts >= ms(t0)) & (t.ts < ms(t1)) & (t.combo == 0) & (t.liq == 0)]
    return t[t.instrument.str.match(RX)]


def leg_iv(t, inst, exp):
    g = t[t.instrument == inst]
    if g.empty:
        return None
    f = g.iloc[0]
    _, K, call = parse(inst)
    T = (exp - (EPOCH + timedelta(milliseconds=int(f.ts)))).total_seconds() / YEAR
    v = iv_of(f.mark * f["index"], f["index"], K, T, call)
    return v if v and v > 0.05 else None          # drop broken marks (see the 2019-10-31 glitch)


def run(cur):
    perp = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in sorted(glob.glob(R + f"perp_1m/{cur}-PERPETUAL/*.parquet"))])
    perp = perp.drop_duplicates("ts").set_index("ts")["close"].sort_index()
    spread = json.load(open(R + f"calibration/spread_model_{cur}.json"))["table"]["all"]
    deliv = dict(zip(*pd.read_csv(R + f"delivery/{cur.lower()}_usd.csv")[["date", "price"]].T.values))

    def half(K, S, call, days):
        mny = (K / S - 1) if call else (1 - K / S)
        ob = next(n for lo, hi, n in OTM if lo <= mny < hi)
        db = next(n for lo, hi, n in DTE if lo <= days < hi)
        return spread.get(ob, {}).get(db, {"half_spread": 0.0005})["half_spread"] * S

    rows = []
    d = datetime(2019, 1, 4) if cur == "BTC" else datetime(2020, 1, 3)
    while d <= datetime(2026, 10, 9):
        F, d = d, d + timedelta(days=7)
        t_in, exp = F + timedelta(hours=9), F + timedelta(days=7, hours=8)
        t_out = F + timedelta(days=3, hours=8)                      # Monday 08:00
        s0, s1 = perp.get(ms(t_in) - 60000), perp.get(ms(t_out) - 60000)
        D = deliv.get(f"{exp:%Y-%m-%d}")
        a = prints(cur, F, t_in, t_in + timedelta(hours=2))
        b = prints(cur, t_out, t_out - timedelta(hours=1), t_out + timedelta(hours=3))
        if a is None or b is None or a.empty or b.empty or s0 is None or s1 is None or D is None:
            continue
        info = {i: parse(i) for i in a.instrument.unique()}
        cands = [i for i, v in info.items() if v[0] == exp]
        ks = {}
        for i in cands:
            ks.setdefault(info[i][1], {})[info[i][2]] = i
        pairs = [k for k, v in ks.items() if len(v) == 2 and abs(k / s0 - 1) < 0.03]
        if not pairs:
            continue
        k = min(pairs, key=lambda k: abs(k - s0))
        legs = [ks[k][True], ks[k][False]]
        T0, T1 = (exp - t_in).total_seconds() / YEAR, (exp - t_out).total_seconds() / YEAR
        iv0 = [leg_iv(a, i, exp) for i in legs]
        iv1 = [leg_iv(b, i, exp) for i in legs]
        if None in iv0 or None in iv1:
            continue
        sell = sum(b76(s0, k, T0, v, c) - half(k, s0, c, T0 * 365) for v, c in zip(iv0, (True, False)))
        buy = sum(b76(s1, k, T1, v, c) + half(k, s1, c, T1 * 365) for v, c in zip(iv1, (True, False)))
        fees = 2 * 0.0003 * s0 + 2 * 0.0003 * s1
        # daily hedge at entry, Saturday and Sunday 08:00, closed Monday 08:00
        hp, hfee, h, sp = 0.0, 0.0, 0.0, s0
        for when in (t_in, F + timedelta(days=1, hours=8), F + timedelta(days=2, hours=8)):
            s = perp.get(ms(when) - 60000)
            if s is None:
                continue
            hp += h * (s - sp)
            tau = (exp - when).total_seconds() / YEAR
            target = sum(delta(s, k, tau, v, c) for v, c in zip(iv0, (True, False)))   # short straddle -> hold +delta
            hfee += 0.0005 * abs(target - h) * s
            h, sp = target, s
        hp += h * (s1 - sp)
        hfee += 0.0005 * abs(h) * s1
        # same straddle held to settlement (unhedged), for comparison
        hold = sum(b76(s0, k, T0, v, c) - half(k, s0, c, T0 * 365) for v, c in zip(iv0, (True, False))) \
            - abs(D - k) - 2 * 0.0003 * s0 - min(0.00015 * D, 0.125 * abs(D - k))
        rows.append(dict(cur=cur, friday=F, s0=s0, iv_fri=np.mean(iv0), iv_mon=np.mean(iv1),
                         weekend_unhedged=(sell - buy - fees) / s0, weekend_hedged=(sell - buy - fees + hp - hfee) / s0,
                         hold_to_expiry=hold / s0,
                         rv_weekend=abs(math.log(s1 / s0)) * math.sqrt(YEAR / ((t_out - t_in).total_seconds()))))
    return pd.DataFrame(rows)


def stats(x):
    t = x.mean() / (x.std() / math.sqrt(len(x))) if len(x) > 2 else float("nan")
    return f"{x.mean():+.3%}/wk (t {t:+.2f}, n {len(x)})"


if __name__ == "__main__":
    for cur in ["BTC", "ETH"]:
        df = run(cur)
        df.to_pickle(f"/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/weekend_{cur}.pkl")
        for lab, g in (("in-sample 2019-23", df[df.friday < "2024-01-01"]), ("out-of-sample 2024-26", df[df.friday >= "2024-01-01"])):
            print(f"{cur} {lab}: IV Fri {g.iv_fri.mean():.1%} -> Mon {g.iv_mon.mean():.1%} | short Fri->Mon unhedged "
                  f"{stats(g.weekend_unhedged)}, hedged {stats(g.weekend_hedged)} | held to expiry {stats(g.hold_to_expiry)}")
