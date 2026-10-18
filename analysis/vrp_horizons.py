"""IV vs realized vol and short ATM straddle P&L (unhedged and daily delta-hedged with the perp) for
~1-week and ~1-month Deribit options, entry Friday 09:30-11:30 UTC (after the 08:00 settlement), prices = Deribit mark."""
import glob, math, re
from datetime import datetime, timedelta
import numpy as np, pandas as pd
R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
MON = {m: i + 1 for i, m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
EPOCH = datetime(1970, 1, 1)
ms = lambda dt: int((dt - EPOCH).total_seconds() * 1000)
Nf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
RX = r"^(BTC|ETH)-\d{1,2}[A-Z]{3}\d{2}-\d+(\.\d+)?-[CP]$"
def parse(inst):
    _, d, k, cp = inst.split("-"); m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})$", d)
    return datetime(2000 + int(m.group(3)), MON[m.group(2)], int(m.group(1)), 8), float(k), cp

def run(cur, start, horizon_days, lo, hi):
    perp = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in sorted(glob.glob(R + f"perp_1m/{cur}-PERPETUAL/*.parquet"))]).drop_duplicates("ts").set_index("ts")["close"].sort_index()
    dl = pd.read_csv(R + f"delivery/{cur.lower()}_usd.csv"); deliv = dict(zip(dl.date, dl.price))
    rows = []; d = start
    while d.weekday() != 4: d += timedelta(days=1)
    while d <= datetime(2026, 10, 9):
        F = d; d += timedelta(days=7)
        try: t = pd.read_parquet(R + f"options/{cur}/{F.year}/{F:%Y%m%d}.parquet")
        except Exception: continue
        t0 = F + timedelta(hours=9, minutes=30)
        t = t[(t.ts >= ms(t0)) & (t.ts < ms(t0) + 7200000) & (t.combo == 0) & (t.liq == 0)]
        t = t[t.instrument.str.match(RX)]
        if t.empty: continue
        info = {i: parse(i) for i in t.instrument.unique()}
        exps = sorted({v[0] for v in info.values() if lo <= (v[0] - t0).days <= hi})
        if not exps: continue
        exp = min(exps, key=lambda e: abs((e - t0).total_seconds() / 86400 - horizon_days))
        D = deliv.get(f"{exp:%Y-%m-%d}"); s0 = perp.get(ms(t0) - 60000)
        if D is None or s0 is None or exp > datetime(2026, 10, 16, 8): continue
        x = t[t.instrument.map(lambda i: info[i][0] == exp)].copy()
        x["K"] = x.instrument.map(lambda i: info[i][1]); x["cp"] = x.instrument.map(lambda i: info[i][2])
        both = x.groupby("K").cp.nunique(); ks = both[both == 2].index
        if not len(ks): continue
        k = min(ks, key=lambda k: abs(k - s0))
        if abs(k / s0 - 1) > 0.03: continue
        s = x[x.K == k]
        prem = sum(g.mark.iloc[0] * g["index"].iloc[0] for _, g in s.groupby("cp"))
        iv = np.median(s.iv) / 100
        T = (exp - t0).total_seconds() / (365 * 86400)
        path = perp.loc[ms(t0) - 60000: ms(exp) - 60000]
        p5 = path.values[::5]
        rv = math.sqrt(np.sum(np.diff(np.log(p5)) ** 2) / T)
        payoff = abs(D - k)
        # daily delta hedge at 08:00 UTC (and at entry) with fixed entry IV
        hedge, h_prev, s_prev, tt = 0.0, None, s0, t0
        times = [t0] + [datetime(tt.year, tt.month, tt.day, 8) + timedelta(days=i) for i in range(1, 400)]
        for when in times:
            if when >= exp: break
            s_now = perp.get(ms(when) - 60000)
            if s_now is None: continue
            if h_prev is not None: hedge += h_prev * (s_now - s_prev)
            tau = (exp - when).total_seconds() / (365 * 86400)
            d1 = (math.log(s_now / k) + 0.5 * iv * iv * tau) / (iv * math.sqrt(tau))
            h_prev, s_prev = 2 * Nf(d1) - 1, s_now          # short straddle delta is -(2N(d1)-1); hold +that in perp
        hedge += h_prev * (D - s_prev)
        rows.append(dict(cur=cur, friday=F, days=(exp - t0).total_seconds() / 86400, iv=iv, rv=rv, prem=prem / s0,
                         unhedged=(prem - payoff) / s0, hedged=(prem - payoff + hedge) / s0))
    return pd.DataFrame(rows)

for cur, start in [("BTC", datetime(2019, 1, 1)), ("ETH", datetime(2020, 1, 1))]:
    for label, h, lo, hi in [("1-week", 7, 5, 9), ("1-month", 28, 21, 35)]:
        df = run(cur, start, h, lo, hi)
        step = 1 if h == 7 else 4                                   # non-overlapping samples for the t-stat
        nz = df.iloc[::step]
        tstat = lambda s: s.mean() / (s.std() / math.sqrt(len(s)))
        print(f"\n{cur} {label} ATM straddle ({len(df)} entries, avg {df.days.mean():.1f} days): IV {df.iv.mean():.1%} vs RV {df.rv.mean():.1%}; IV>RV {(df.iv > df.rv).mean():.0%} of entries; mean IV^2-RV^2 {(df.iv**2 - df.rv**2).mean():+.3f}")
        print(f"   short straddle at mark, % of spot per trade: unhedged {df.unhedged.mean():+.3%} (t {tstat(nz.unhedged):+.2f}), delta-hedged daily {df.hedged.mean():+.3%} (t {tstat(nz.hedged):+.2f}); premium {df.prem.mean():.2%}; hedged capture {df.hedged.sum()/df.prem.sum():+.1%}")
        for y, g in df.groupby(df.friday.dt.year):
            print(f"     {y}: n {len(g):2d} IV {g.iv.mean():.0%} RV {g.rv.mean():.0%} hedged {g.hedged.mean():+.2%} unhedged {g.unhedged.mean():+.2%}")
