"""
Fast screen of option-selling variants on raw Deribit data (no LEAN), used to choose what to backtest in LEAN.

Protocol: every variant is computed for every week, but choices are made on the in-sample years only
(<= 2023); the out-of-sample years (2024-2026) are reported separately and only for the finalists.

Per week: enter at t0 on the next weekly expiry, hold to the official 08:00 settlement.
  entry "fri": Friday 09:00 UTC (right after the weekly settlement), 7-day option
  entry "mon": Monday 09:30 UTC, 4-day option (the original project's timing)
Leg mid at t0 = Black-76 at the perpetual price with the IV implied by Deribit's mark price at the leg's
first print in the 2-hour entry window. Sell at mid - half-spread, buy at mid + half-spread (live-book
spread model, coin units x spot). Option fee min(0.03% S, 12.5% price), delivery fee min(0.015% D,
12.5% intrinsic). Hedge with the perpetual (taker 0.05% or maker 0%), hourly funding charged.
P&L per week is per 1-coin unit per leg, as % of spot at entry; equity effect = unit_notional x that.
"""
import glob, json, math, os, re, sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

R = os.environ.get("DERIBIT_RAW", "/mnt/quant_storage/quant_workspace/deribit_raw/")
OUT = os.environ.get("SCREEN_OUT", "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/")
MON = {m: i + 1 for i, m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
RX = r"^(BTC|ETH)-\d{1,2}[A-Z]{3}\d{2}-\d+(\.\d+)?-[CP]$"
EPOCH = datetime(1970, 1, 1)
YEAR = 365 * 86400.0
OTM_B = [("ITM>3%", -9, -0.03), ("ITM1-3%", -0.03, -0.01), ("ATM", -0.01, 0.01), ("OTM1-3%", 0.01, 0.03),
         ("OTM3-6%", 0.03, 0.06), ("OTM6-10%", 0.06, 0.10), ("OTM10-20%", 0.10, 0.20), ("OTM>20%", 0.20, 9)]
DTE_B = [("<1d", 0, 1), ("1-3d", 1, 3), ("3-8d", 3, 8), ("8-21d", 8, 21), (">21d", 21, 9999)]

STRUCTURES = {          # name: list of (right, target |delta| or "atm", qty sign)
    "straddle":      [("C", "atm", -1), ("P", "atm", -1)],
    "strangle25":    [("C", 0.25, -1), ("P", 0.25, -1)],
    "strangle35":    [("C", 0.35, -1), ("P", 0.35, -1)],
    "put30":         [("P", 0.30, -1)],
    "put40":         [("P", 0.40, -1)],
    "ironfly10":     [("C", "atm", -1), ("P", "atm", -1), ("C", 0.10, 1), ("P", 0.10, 1)],
    "ironfly05":     [("C", "atm", -1), ("P", "atm", -1), ("C", 0.05, 1), ("P", 0.05, 1)],
}
HEDGES = ["none", "daily", "h4", "band10", "band25"]


def ms(dt):
    return int((dt - EPOCH).total_seconds() * 1000)


def parse(inst):
    _, d, k, cp = inst.split("-")
    m = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})$", d)
    return datetime(2000 + int(m.group(3)), MON[m.group(2)], int(m.group(1)), 8), float(k), cp


def b76(F, K, T, s, call):
    if T <= 0 or s <= 0:
        return max(F - K, 0.0) if call else max(K - F, 0.0)
    v = s * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    c = F * norm.cdf(d1) - K * norm.cdf(d1 - v)
    return c if call else c - (F - K)


def b76_delta(F, K, T, s, call):
    if T <= 0 or s <= 0:
        return (1.0 if F > K else 0.0) if call else (-1.0 if F < K else 0.0)
    v = s * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    return norm.cdf(d1) if call else norm.cdf(d1) - 1.0


def iv_of(price, F, K, T, call):
    lo, hi = b76(F, K, T, 0.02, call), b76(F, K, T, 5.0, call)
    if not (lo < price < hi):
        return None
    return brentq(lambda s: b76(F, K, T, s, call) - price, 0.02, 5.0, xtol=1e-6)


class Spread:
    def __init__(self, cur):
        self.t = json.load(open(R + f"calibration/spread_model_{cur}.json"))["table"]["all"]   # same model as the LEAN data

    def half(self, K, S, call, days):
        mny = (K / S - 1) if call else (1 - K / S)        # > 0 = out of the money
        ob = next(n for n, lo, hi in OTM_B if lo <= mny < hi)
        db = next(n for n, lo, hi in DTE_B if lo <= days < hi)
        cell = self.t.get(ob, {}).get(db)
        return cell["half_spread"] if cell else 0.0005


def load_common(cur):
    perp = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in sorted(glob.glob(R + f"perp_1m/{cur}-PERPETUAL/*.parquet"))])
    perp = perp.drop_duplicates("ts").set_index("ts")["close"].sort_index()
    dl = pd.read_csv(R + f"delivery/{cur.lower()}_usd.csv")
    deliv = dict(zip(dl.date, dl.price))
    fund = {}
    fp = R + f"funding/{cur}-PERPETUAL.csv"
    if os.path.exists(fp):
        f = pd.read_csv(fp)
        f = f[f.interest_1h.notna()]
        fund = dict(zip((f.ts // 3_600_000).astype(int), f.interest_1h.astype(float)))
    # daily realised variance from 5-minute log returns (UTC days), for the HAR forecast
    p5 = perp[perp.index % 300000 == 0]
    r = np.log(p5).diff().dropna()
    day = pd.to_datetime(r.index, unit="ms").normalize()
    rv = (r ** 2).groupby(day).sum()
    return perp, deliv, fund, rv


def har_forecast(rv, t0):
    """Direct HAR forecast (annualised vol) of the next 7 days' mean daily RV, rolling 365-day OLS, data < t0."""
    x = rv[rv.index < pd.Timestamp(t0.date())].values
    if len(x) < 30 + 7 + 120:
        return np.nan
    X, y = [], []
    for t in range(29, len(x) - 7):
        X.append([1.0, x[t], x[t - 6:t + 1].mean(), x[t - 29:t + 1].mean()])
        y.append(x[t + 1:t + 8].mean())
    X, y = np.array(X[-365:]), np.array(y[-365:])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    t = len(x) - 1
    f = beta @ [1.0, x[t], x[t - 6:t + 1].mean(), x[t - 29:t + 1].mean()]
    m = x[t - 29:t + 1].mean()
    return math.sqrt(365.0 * min(max(f, 0.2 * m), 5.0 * m))


def simulate(legs, s0, t0, exp, D, perp, fund, hedge, perp_fee):
    """legs: list of dict(K, call, q, iv). Returns (hedge_pnl, hedge_fees, funding) in USD."""
    if hedge == "none":
        return 0.0, 0.0, 0.0
    times = [t0]                                    # entry, then every full hour (so 08:00 / 4-hourly checks align)
    t = t0.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while t < exp:
        times.append(t)
        t += timedelta(hours=1)
    h, sp, pnl, fees, fund_paid = 0.0, s0, 0.0, 0.0, 0.0
    for i, t in enumerate(times):
        s = perp.get(ms(t) - 60000)
        if s is None or not s > 0:
            continue
        pnl += h * (s - sp)
        if i > 0 and h != 0.0:
            rate = fund.get(ms(t) // 3_600_000)
            if rate is not None:
                fund_paid += h * s * rate          # long perp pays positive funding
        sp = s
        tau = (exp - t).total_seconds() / YEAR
        net = sum(l["q"] * b76_delta(s, l["K"], tau, l["iv"], l["call"]) for l in legs)
        target = -net
        if hedge == "daily":
            do = (i == 0) or (t.hour == 8 and t.minute == 0)
        elif hedge == "h4":
            do = (i == 0) or (t.hour % 4 == 0 and t.minute == 0)
        else:
            band = 0.10 if hedge == "band10" else 0.25
            do = abs(target - h) > band
        if do:
            fees += perp_fee * abs(target - h) * s
            h = target
    pnl += h * (D - sp)
    fees += perp_fee * abs(h) * D
    return pnl, fees, fund_paid


def week_rows(args):
    cur, entry, t0, exp = args
    perp, deliv, fund, rv, spread = G["perp"], G["deliv"], G["fund"], G["rv"], G["spread"]
    D = deliv.get(f"{exp:%Y-%m-%d}")
    s0 = perp.get(ms(t0) - 60000)
    if D is None or s0 is None:
        return []
    try:
        t = pd.read_parquet(R + f"options/{cur}/{t0.year}/{t0:%Y%m%d}.parquet")
    except Exception:
        return []
    t = t[(t.ts >= ms(t0)) & (t.ts < ms(t0) + 7200000) & (t.combo == 0) & (t.liq == 0)]
    t = t[t.instrument.str.match(RX)]
    if t.empty:
        return []
    info = {i: parse(i) for i in t.instrument.unique()}
    t = t[t.instrument.map(lambda i: info[i][0] == exp)]
    if t.empty:
        return []
    T0 = (exp - t0).total_seconds() / YEAR
    days = T0 * 365
    q = []
    for inst, g in t.groupby("instrument"):
        _, K, cp = info[inst]
        first = g.iloc[0]
        Tp = (exp - (EPOCH + timedelta(milliseconds=int(first.ts)))).total_seconds() / YEAR
        iv = iv_of(first.mark * first["index"], first["index"], K, Tp, cp == "C")
        if iv is None:
            continue
        mid = b76(s0, K, T0, iv, cp == "C")
        q.append(dict(K=K, call=(cp == "C"), iv=iv, mid=mid, delta=b76_delta(s0, K, T0, iv, cp == "C"),
                      half=spread.half(K, s0, cp == "C", days) * s0))
    if not q:
        return []
    q = pd.DataFrame(q)
    atm_k = None
    both = q.groupby("K").call.nunique()
    ks = both[both == 2].index
    if len(ks):
        k = min(ks, key=lambda k: abs(k - s0))
        if abs(k / s0 - 1) < 0.03:
            atm_k = k
    atm_iv = q[(q.K / s0 - 1).abs() < 0.03].iv.median()
    fv = har_forecast(rv, t0)
    rows = []
    for name, spec in STRUCTURES.items():
        legs = []
        for right, tgt, sign in spec:
            c = q[q.call == (right == "C")]
            if tgt == "atm":
                if atm_k is None:
                    legs = None
                    break
                r = c[c.K == atm_k].iloc[0]
            else:
                c = c[(c.delta.abs() > 0.02) & (c.delta.abs() < 0.6)]
                if c.empty:
                    legs = None
                    break
                r = c.iloc[(c.delta.abs() - tgt).abs().argmin()]
                if abs(abs(r.delta) - tgt) > 0.06:
                    legs = None
                    break
            legs.append(dict(K=r.K, call=r.call, q=float(sign), iv=r.iv, mid=r.mid, half=r.half))
        if not legs or len({(l["K"], l["call"]) for l in legs}) < len(legs):
            continue
        prem = 0.0; opt_fee = 0.0; payoff = 0.0; dfee = 0.0; prem_mid = 0.0
        for l in legs:
            px = l["mid"] - l["half"] if l["q"] < 0 else l["mid"] + l["half"]
            px = max(px, 0.0)
            prem += -l["q"] * px
            prem_mid += -l["q"] * l["mid"]
            opt_fee += min(0.0003 * s0, 0.125 * px) * abs(l["q"])
            intr = max(D - l["K"], 0.0) if l["call"] else max(l["K"] - D, 0.0)
            payoff += -l["q"] * intr
            dfee += min(0.00015 * D, 0.125 * intr) * abs(l["q"]) if intr > 0 else 0.0
        for hedge in HEDGES:
            hp_t, hf_t, fu = simulate(legs, s0, t0, exp, D, G["perp"], fund, hedge, 0.0005)
            rows.append(dict(cur=cur, entry=entry, t0=t0, structure=name, hedge=hedge, s0=s0, atm_iv=atm_iv, har=fv,
                             premium=prem / s0, premium_mid=prem_mid / s0, option_pnl=(prem - payoff) / s0,
                             option_costs=(opt_fee + dfee) / s0, spread_cost=(prem_mid - prem) / s0,
                             hedge_pnl=hp_t / s0, hedge_fee_taker=hf_t / s0, funding=fu / s0))
    return rows


G = {}


def init(cur):
    perp, deliv, fund, rv = load_common(cur)
    G.update(perp=perp, deliv=deliv, fund=fund, rv=rv, spread=Spread(cur))


def weeks(cur, entry):
    start = datetime(2019, 1, 1) if cur == "BTC" else datetime(2020, 1, 1)
    d, out = start, []
    wd = 4 if entry == "fri" else 0
    while d.weekday() != wd:
        d += timedelta(days=1)
    while d <= datetime(2026, 10, 12):
        if entry == "fri":
            t0, exp = d + timedelta(hours=9), d + timedelta(days=7, hours=8)
        else:
            t0, exp = d + timedelta(hours=9, minutes=30), d + timedelta(days=4, hours=8)
        if exp <= datetime(2026, 10, 16, 8):
            out.append((cur, entry, t0, exp))
        d += timedelta(days=7)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    frames = []
    for cur in ["BTC", "ETH"]:
        jobs = weeks(cur, "fri") + weeks(cur, "mon")
        if os.environ.get("LIMIT"):                      # quick test on a few weeks
            jobs = jobs[200:200 + int(os.environ["LIMIT"])]
        with Pool(int(os.environ.get("WORKERS", "24")), initializer=init, initargs=(cur,)) as pool:
            for rows in pool.imap_unordered(week_rows, jobs, chunksize=4):
                frames.extend(rows)
        print(cur, "done", len(frames), flush=True)
    df = pd.DataFrame(frames)
    df.to_pickle(OUT + "screen_volsell.pkl")
    print("rows", len(df))
