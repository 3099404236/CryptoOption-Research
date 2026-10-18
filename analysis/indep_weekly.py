"""Independent (non-LEAN) check: weekly option selling on Deribit using only real trade prints.
Entry Monday 09:30-11:30 UTC on that Friday's expiry, settle at the official 08:00 delivery price.
Prices: 'mark' = Deribit mark price at the print (mid proxy); 'bid' = VWAP of taker-sell prints (what a seller really got).
USD accounting (premium = coin price x index at the print), gross of fees."""
import glob, math, re, sys
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd

R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
MON = {m: i + 1 for i, m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
N = np.vectorize(lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2))))
EPOCH = datetime(1970, 1, 1)

def ms(dt): return int((dt - EPOCH).total_seconds() * 1000)

def parse(inst):
    _, d, k, cp = inst.split("-")
    m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})$", d)
    return datetime(2000 + int(m.group(3)), MON[m.group(2)], int(m.group(1)), 8, 0), float(k), cp

def load_perp(cur):
    fs = sorted(glob.glob(R + f"perp_1m/{cur}-PERPETUAL/*.parquet"))
    p = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in fs]).drop_duplicates("ts").set_index("ts")["close"].sort_index()
    return p

def run(cur, start):
    perp = load_perp(cur)
    dl = pd.read_csv(R + f"delivery/{cur.lower()}_usd.csv")
    deliv = dict(zip(dl["date"], dl["price"]))
    legs = {"C07": ("C", 0.07), "C20": ("C", 0.20), "C35": ("C", 0.35), "P07": ("P", -0.07), "P20": ("P", -0.20), "P35": ("P", -0.35)}
    rows = []
    d = start
    while d.weekday() != 0: d += timedelta(days=1)
    end = datetime(2026, 10, 12)
    while d <= end:
        M = d; d += timedelta(days=7)
        f = R + f"options/{cur}/{M.year}/{M:%Y%m%d}.parquet"
        try:
            t = pd.read_parquet(f)
        except Exception:
            continue
        t0, t1 = ms(M + timedelta(hours=9, minutes=30)), ms(M + timedelta(hours=11, minutes=30))
        t = t[(t.ts >= t0) & (t.ts < t1) & (t.combo == 0) & (t.liq == 0)]
        t = t[t.instrument.str.match(r"^(BTC|ETH)-\d{1,2}[A-Z]{3}\d{2}-\d+(\.\d+)?-[CP]$")]
        exp = M + timedelta(days=4, hours=8)
        D = deliv.get(f"{exp:%Y-%m-%d}")
        if t.empty or D is None:
            continue
        info = {i: parse(i) for i in t.instrument.unique()}
        t = t[t.instrument.map(lambda i: info[i][0] == exp)]
        s0 = perp.get(t0 - 60000)
        if t.empty or s0 is None or np.isnan(s0):
            continue
        T = (exp - (M + timedelta(hours=9, minutes=30))).total_seconds() / (365 * 86400)
        # realized vol over the holding period from 5-minute perp closes
        path = perp.loc[t0 - 60000: ms(exp) - 60000].values[::5]
        rv = math.sqrt(np.sum(np.diff(np.log(path)) ** 2) / T) if len(path) > 100 else np.nan
        g = []
        for inst, x in t.groupby("instrument"):
            _, K, cp = info[inst]
            iv = np.median(x.iv) / 100
            if not iv > 0: continue
            d1 = (math.log(s0 / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
            delta = float(N(d1)) - (1 if cp == "P" else 0)
            first = x.iloc[0]
            sells = x[x.side == -1]
            bid = (sells.price * sells["index"] * sells.amount).sum() / sells.amount.sum() if len(sells) else np.nan
            payoff = max(D - K, 0) if cp == "C" else max(K - D, 0)
            g.append(dict(inst=inst, K=K, cp=cp, iv=iv, delta=delta, mark=first.mark * first["index"], bid=bid, payoff=payoff))
        g = pd.DataFrame(g)
        if g.empty: continue
        atm = g[(g.K / s0 - 1).abs() < 0.03]
        rec = dict(cur=cur, monday=M, s0=s0, D=D, atm_iv=np.median(atm.iv) if len(atm) else np.nan, rv=rv)
        for name, (cp, tgt) in legs.items():
            c = g[(g.cp == cp) & (g.delta.abs() > 0.02) & (g.delta.abs() < 0.6)]
            if c.empty: continue
            r = c.iloc[(c.delta - tgt).abs().argmin()]
            if abs(r.delta - tgt) > 0.08: continue
            rec[name + "_mark"] = (r.mark - r.payoff) / s0
            rec[name + "_bid"] = (r.bid - r.payoff) / s0 if not np.isnan(r.bid) else np.nan
            rec[name + "_prem"] = r.mark / s0
        # ATM straddle: strike closest to spot with both a call and a put print
        both = g.groupby("K").cp.nunique()
        ks = both[both == 2].index
        if len(ks):
            k = min(ks, key=lambda k: abs(k - s0))
            if abs(k / s0 - 1) < 0.03:
                s = g[g.K == k]
                rec["STRADDLE_mark"] = (s.mark.sum() - s.payoff.sum()) / s0
                rec["STRADDLE_bid"] = (s.bid.sum(min_count=len(s)) - s.payoff.sum()) / s0   # NaN unless both legs have taker-sell prints
                rec["STRADDLE_prem"] = s.mark.sum() / s0
        rows.append(rec)
    return pd.DataFrame(rows)

out = pd.concat([run("BTC", datetime(2019, 1, 1)), run("ETH", datetime(2020, 1, 1))])
out.to_pickle("/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/indep_weekly.pkl")

def summ(df, label):
    print(f"\n### {label}: {len(df)} weeks")
    print("ATM IV mean {:.1%}  realized mean {:.1%}  IV>RV in {:.0%} of weeks  mean(IV^2-RV^2) {:+.4f}".format(
        df.atm_iv.mean(), df.rv.mean(), (df.atm_iv > df.rv).mean(), (df.atm_iv ** 2 - df.rv ** 2).mean()))
    print(f"{'leg':10s} {'n':>4s} {'prem%S':>7s} {'pnl@mark%S':>11s} {'t':>6s} {'capture':>8s} {'pnl@bid%S':>10s} {'capture':>8s}")
    for leg in ["P07", "P20", "P35", "C07", "C20", "C35", "STRADDLE"]:
        if leg + "_mark" not in df: continue
        x = df[[leg + "_mark", leg + "_bid", leg + "_prem"]].dropna()
        if x.empty: continue
        m, b, p = x[leg + "_mark"], x[leg + "_bid"], x[leg + "_prem"]
        t = m.mean() / (m.std() / math.sqrt(len(m)))
        print(f"{leg:10s} {len(x):4d} {p.mean():7.3%} {m.mean():11.4%} {t:6.2f} {m.sum()/p.sum():8.1%} {b.mean():10.4%} {b.sum()/p.sum():8.1%}")
    # 0.20-delta strangle = P20 + C20
    x = df[["P20_mark", "C20_mark", "P20_bid", "C20_bid", "P20_prem", "C20_prem"]].dropna()
    if len(x):
        m = x.P20_mark + x.C20_mark; b = x.P20_bid + x.C20_bid; p = x.P20_prem + x.C20_prem
        print(f"{'STRANGLE20':10s} {len(x):4d} {p.mean():7.3%} {m.mean():11.4%} {m.mean()/(m.std()/math.sqrt(len(m))):6.2f} {m.sum()/p.sum():8.1%} {b.mean():10.4%} {b.sum()/p.sum():8.1%}")

for cur in ["BTC", "ETH"]:
    df = out[out.cur == cur]
    summ(df, f"{cur} all years")
    for y, dy in df.groupby(df.monday.dt.year):
        x = dy[["P20_mark", "C20_mark"]].dropna()
        print(f"  {y}: weeks {len(dy):3d}  ATM IV {dy.atm_iv.mean():.0%} RV {dy.rv.mean():.0%}  strangle20 pnl@mark/wk {(x.P20_mark + x.C20_mark).mean():+.3%}  straddle pnl@mark/wk {dy.get('STRADDLE_mark', pd.Series(dtype=float)).mean():+.3%}")
