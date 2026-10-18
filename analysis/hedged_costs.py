"""Weekly ATM short straddle, delta-hedged daily with the perp: gross P&L at Deribit mark, then costs
(option fee 0.03% of spot per leg, delivery fee 0.015% capped at 12.5% of intrinsic, ATM half-spread from the
live-book model, perp taker 0.05% or maker 0% on every hedge change incl. the final unwind)."""
import glob, math, re
from datetime import datetime, timedelta
import numpy as np, pandas as pd
R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
MON = {m: i + 1 for i, m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
EPOCH = datetime(1970, 1, 1); ms = lambda dt: int((dt - EPOCH).total_seconds() * 1000)
Nf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
RX = r"^(BTC|ETH)-\d{1,2}[A-Z]{3}\d{2}-\d+(\.\d+)?-[CP]$"
HALF = {"BTC": 0.0005, "ETH": 0.0005}       # ATM 3-8 day half-spread in coin (live-book median)
def parse(inst):
    _, d, k, cp = inst.split("-"); m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})$", d)
    return datetime(2000 + int(m.group(3)), MON[m.group(2)], int(m.group(1)), 8), float(k), cp
for cur, start in [("BTC", datetime(2019, 1, 1)), ("ETH", datetime(2020, 1, 1))]:
    perp = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in sorted(glob.glob(R + f"perp_1m/{cur}-PERPETUAL/*.parquet"))]).drop_duplicates("ts").set_index("ts")["close"].sort_index()
    dl = pd.read_csv(R + f"delivery/{cur.lower()}_usd.csv"); deliv = dict(zip(dl.date, dl.price))
    rows = []; d = start
    while d.weekday() != 4: d += timedelta(days=1)
    while d <= datetime(2026, 10, 9):
        F = d; d += timedelta(days=7)
        try: t = pd.read_parquet(R + f"options/{cur}/{F.year}/{F:%Y%m%d}.parquet")
        except Exception: continue
        t0 = F + timedelta(hours=9, minutes=30); exp = F + timedelta(days=7, hours=8)
        t = t[(t.ts >= ms(t0)) & (t.ts < ms(t0) + 7200000) & (t.combo == 0) & (t.liq == 0)]
        t = t[t.instrument.str.match(RX)]
        if t.empty: continue
        info = {i: parse(i) for i in t.instrument.unique()}
        x = t[t.instrument.map(lambda i: info[i][0] == exp)].copy()
        D = deliv.get(f"{exp:%Y-%m-%d}"); s0 = perp.get(ms(t0) - 60000)
        if x.empty or D is None or s0 is None: continue
        x["K"] = x.instrument.map(lambda i: info[i][1]); x["cp"] = x.instrument.map(lambda i: info[i][2])
        both = x.groupby("K").cp.nunique(); ks = both[both == 2].index
        if not len(ks): continue
        k = min(ks, key=lambda k: abs(k - s0))
        if abs(k / s0 - 1) > 0.03: continue
        s = x[x.K == k]
        prem = sum(g.mark.iloc[0] * g["index"].iloc[0] for _, g in s.groupby("cp"))
        iv = np.median(s.iv) / 100
        hedge, turnover, h, sp = 0.0, 0.0, 0.0, s0
        daily = [s0]
        when = t0
        sched = [t0] + [datetime(F.year, F.month, F.day, 8) + timedelta(days=i) for i in range(1, 7)]
        for w in sched:
            sn = perp.get(ms(w) - 60000)
            if sn is None: continue
            hedge += h * (sn - sp)
            tau = (exp - w).total_seconds() / (365 * 86400)
            hn = 2 * Nf((math.log(sn / k) + 0.5 * iv * iv * tau) / (iv * math.sqrt(tau))) - 1
            turnover += abs(hn - h) * sn
            h, sp = hn, sn
            if w != t0: daily.append(sn)
        hedge += h * (D - sp); turnover += abs(h) * D; daily.append(D)
        T = (exp - t0).total_seconds() / (365 * 86400)
        rv_daily = math.sqrt(np.sum(np.diff(np.log(daily)) ** 2) / T)
        intrinsic = abs(D - k)
        fees = 2 * 0.0003 * s0 + min(0.00015 * D, 0.125 * intrinsic)
        spread = 2 * HALF[cur] * s0
        gross = prem - intrinsic + hedge
        rows.append(dict(F=F, iv=iv, rv_daily=rv_daily, prem=prem / s0, gross=gross / s0, fees=fees / s0, spread=spread / s0,
                         hedge_taker=0.0005 * turnover / s0, turnover=turnover / s0))
    df = pd.DataFrame(rows)
    df["net_taker"] = df.gross - df.fees - df.spread - df.hedge_taker
    df["net_maker"] = df.gross - df.fees - df.spread
    t = lambda s: s.mean() / (s.std() / math.sqrt(len(s)))
    print(f"\n{cur}: {len(df)} weeks; ATM IV {df.iv.mean():.1%} vs realized from daily 08:00 closes {df.rv_daily.mean():.1%}")
    print(f"  per week, % of spot: premium {df.prem.mean():.2%} | gross {df.gross.mean():+.3%} (t {t(df.gross):+.2f})")
    print(f"  costs: option fees {df.fees.mean():.3%}, option spread {df.spread.mean():.3%}, hedge taker fees {df.hedge_taker.mean():.3%} (hedge turnover {df.turnover.mean():.2f}x spot)")
    print(f"  net with taker hedges {df.net_taker.mean():+.3%} (t {t(df.net_taker):+.2f}); with maker (0 fee) hedges {df.net_maker.mean():+.3%} (t {t(df.net_maker):+.2f})")
    print(f"  worst week gross {df.gross.min():+.2%}, best {df.gross.max():+.2%}; share of weeks positive {(df.gross > 0).mean():.0%}")
    for y, g in df.groupby(df.F.dt.year):
        print(f"    {y}: n {len(g):2d}  net_taker/wk {g.net_taker.mean():+.3%}  sum {g.net_taker.sum():+.1%}")
