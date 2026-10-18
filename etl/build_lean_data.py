"""
Build the LEAN data set for the V2 Deribit backtests from raw Deribit downloads.

What changed versus the original ETL (and why):
  * Time stamps are UTC and the workspace registers BTC/ETH as 24/7 UTC instruments, so nothing is
    shifted by the New York clock and nothing outside 09:30-16:00 is dropped.
  * The underlying is the real Deribit perpetual (1-minute OHLCV; 1-second bars from tick trades for
    the short high-frequency window), not the index price copied from option trades and forward-filled.
  * Option quotes are no longer "bid = ask = last trade" (BTC) or "trade +/- 1%" (ETH):
      mid  = Black-76 value from the current perpetual price and the contract's latest
             mark-implied volatility (Deribit mark price inverted; sticky-strike between trades)
      half = historical effective half-spread for that year / moneyness / days-to-expiry,
             measured from Deribit trades (calibrate_spreads.py)
    A contract is only quoted after its first real trade and while its volatility is <= 72 h old.
  * Options are European and expire at 08:00 UTC; no quote exists after that.
  * Real trade prints are kept as trade bars (volume in coins) next to the quotes.
  * Chain files list real contracts with their first-trade time, so strategies cannot pick a contract
    before it existed; signed order-flow files carry Deribit's aggressor side for the flow strategy.

Usage: python build_lean_data.py --currency BTC --start 2019-01-01 --end 2026-10-15 --ws <workspace>
                                 [--second-start 2026-10-11 --second-end 2026-10-15]
"""
import argparse
import io
import json
import os
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from deribit_math import (DTE_NAMES, is_option, OTM_NAMES, YEAR_SECONDS, black76, dte_bucket, implied_vol, otm_bucket,
                          parse_instrument, vega)

RAW = Path(os.environ.get("DERIBIT_RAW", "/mnt/quant_storage/quant_workspace/deribit_raw"))
MAX_DTE_DAYS = 50          # quote contracts up to 50 days to expiry (calendar spreads need ~35)
MAX_MONEYNESS = 0.30       # |K/S - 1| <= 30 %
MAX_IV_AGE_MS = 72 * 3600 * 1000
PERP_HALF_SPREAD_BP = 0.5  # perpetual: half-spread 0.5 bp (tick is 0.5 USD on BTC, 0.05 USD on ETH)


def write_zip(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in entries.items():
            z.writestr(name, text)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.rename(path)


def sc(x):
    """LEAN equity/option price scaling (deci-cents)."""
    return np.round(np.asarray(x, dtype=float) * 10000.0).astype(np.int64)


class SpreadModel:
    """Quoted half-spread (coin units) per moneyness / days-to-expiry bucket from live Deribit books
    (calibrate_spreads.py), applied to every year, times an optional sensitivity multiplier."""

    def __init__(self, cur, mult=1.0):
        m = json.loads((RAW / "calibration" / f"spread_model_{cur}.json").read_text())
        t = m["table"]["all"]
        self.tab = np.array([[t[o][d]["half_spread"] for d in DTE_NAMES] for o in OTM_NAMES]) * mult

    def table(self, year):
        return self.tab


def load_perp_minutes(cur, start, end):
    frames = []
    d = datetime(start.year, start.month, 1)
    while d <= end:
        p = RAW / "perp_1m" / f"{cur}-PERPETUAL" / f"{d:%Y%m}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        d = datetime(d.year + (d.month == 12), d.month % 12 + 1, 1)
    df = pd.concat(frames).drop_duplicates("ts").sort_values("ts")
    return df.set_index("ts")


def day_ms(d):
    return int(d.replace(tzinfo=timezone.utc).timestamp() * 1000)


def build_underlying(cur, perp, days, ws, second_days):
    low = cur.lower()
    base = ws / "data" / "equity" / "usa"
    hour_rows, daily_rows = [], []
    for d in days:
        t0 = day_ms(d)
        idx = np.arange(t0, t0 + 86_400_000, 60_000)
        bars = perp.reindex(idx)
        close = bars["close"].ffill().bfill()
        o = bars["open"].fillna(close)
        h = bars["high"].fillna(close)
        lo = bars["low"].fillna(close)
        v = bars["volume"].fillna(0.0)
        ms = (idx - t0)
        half = close.values * PERP_HALF_SPREAD_BP / 10000.0
        O, H, L, C = sc(o), sc(h), sc(lo), sc(close)
        B, A = sc(close.values - half), sc(close.values + half)
        trade = "\n".join(f"{m},{a},{b},{c},{e},{vv:.4f}" for m, a, b, c, e, vv in zip(ms, O, H, L, C, v.values))
        quote = "\n".join(f"{m},{b},{b},{b},{b},10,{a},{a},{a},{a},10" for m, b, a in zip(ms, B, A))
        ds = f"{d:%Y%m%d}"
        write_zip(base / "minute" / low / f"{ds}_trade.zip", {f"{ds}_{low}_minute_trade.csv": trade + "\n"})
        write_zip(base / "minute" / low / f"{ds}_quote.zip", {f"{ds}_{low}_minute_quote.csv": quote + "\n"})
        for hh in range(24):
            sl = slice(hh * 60, hh * 60 + 60)
            hour_rows.append(f"{ds} {hh:02d}:00,{O[sl][0]},{H[sl].max()},{L[sl].min()},{C[sl][-1]},{int(v.values[sl].sum())}")
        daily_rows.append(f"{ds} 00:00,{O[0]},{H.max()},{L.min()},{C[-1]},{int(v.values.sum())}")
    write_zip(base / "hour" / f"{low}.zip", {f"{low}.csv": "\n".join(hour_rows) + "\n"})
    write_zip(base / "daily" / f"{low}.zip", {f"{low}.csv": "\n".join(daily_rows) + "\n"})

    # 1-second bars from tick trades for the high-frequency window (seconds without trades are filled
    # forward by LEAN; quotes are last trade +/- half-spread)
    for d in second_days:
        p = RAW / "perp_trades" / f"{cur}-PERPETUAL" / f"{d:%Y%m%d}.parquet"
        if not p.exists():
            print(f"  missing perp trades for {d:%Y-%m-%d}, second bars skipped")
            continue
        t = pd.read_parquet(p).sort_values("ts")
        t0 = day_ms(d)
        t["sec"] = (t.ts - t0) // 1000
        g = t.groupby("sec").agg(o=("price", "first"), h=("price", "max"), l=("price", "min"),
                                 c=("price", "last"), v=("amount", "sum"))
        ms = g.index.values * 1000
        O, H, L, C = sc(g.o), sc(g.h), sc(g.l), sc(g.c)
        half = g.c.values * PERP_HALF_SPREAD_BP / 10000.0
        B, A = sc(g.c.values - half), sc(g.c.values + half)
        ds = f"{d:%Y%m%d}"
        write_zip(base / "second" / low / f"{ds}_trade.zip", {f"{ds}_{low}_second_trade.csv": "\n".join(
            f"{m},{a},{b},{c},{e},{vv:.4f}" for m, a, b, c, e, vv in zip(ms, O, H, L, C, g.v.values)) + "\n"})
        write_zip(base / "second" / low / f"{ds}_quote.zip", {f"{ds}_{low}_second_quote.csv": "\n".join(
            f"{m},{b},{b},{b},{b},10,{a},{a},{a},{a},10" for m, b, a in zip(ms, B, A)) + "\n"})
        print(f"  second bars {ds}: {len(g)} seconds with trades")


def build_options(cur, perp, days, ws, spreads):
    low = cur.lower()
    opt_dir = ws / "data" / "option" / "usa" / "minute" / low
    alt = ws / "data" / "alternative" / "deribit"
    meta, first_ts, iv_state = {}, {}, {}           # iv_state[inst] = (ts, iv)
    warm = days[0] - timedelta(days=MAX_DTE_DAYS + 5)
    # warm-up: replay earlier days so contracts listed before the start have a volatility state
    d = warm
    while d < days[0]:
        _update_state(cur, d, meta, first_ts, iv_state)
        d += timedelta(days=1)
    stats = []
    for d in days:
        t0 = day_ms(d)
        trades = _update_state(cur, d, meta, first_ts, iv_state, keep=True)
        idx_end = t0 + np.arange(1, 1441) * 60_000            # end time of each minute bar
        S = perp["close"].reindex(t0 + np.arange(0, 1440) * 60_000).ffill().bfill().values
        year_tab = spreads.table(d.year)

        # contracts eligible today
        cands = [n for n, (exp, k, is_call) in meta.items()
                 if exp > t0 and first_ts.get(n, 1 << 62) < t0 + 86_400_000
                 and (exp - t0) <= MAX_DTE_DAYS * 86_400_000 and abs(k / S[0] - 1) <= MAX_MONEYNESS]
        q_entries, t_entries, chain_rows = {}, {}, []
        by_inst = dict(tuple(trades.groupby("instrument"))) if len(trades) else {}
        for n in cands:
            exp, k, is_call = meta[n]
            # volatility path: state from previous days, updated by today's trades at minute ends
            iv_path = np.full(1440, np.nan)
            age_ref = np.full(1440, np.nan)
            prev = iv_state.get(("prev", n))
            if prev:
                iv_path[:] = prev[1]
                age_ref[:] = prev[0]
            g = by_inst.get(n)
            if g is not None:
                g = g[g.iv_mark.notna()]
                if len(g):
                    m = ((g.ts.values - t0) // 60_000).astype(int)
                    last = pd.Series(g.iv_mark.values, index=m).groupby(level=0).last()
                    lts = pd.Series(g.ts.values, index=m).groupby(level=0).last()
                    upd = np.full(1440, np.nan)
                    upd_t = np.full(1440, np.nan)
                    upd[last.index.values] = last.values
                    upd_t[lts.index.values] = lts.values
                    s1 = pd.Series(upd).ffill().values
                    s2 = pd.Series(upd_t).ffill().values
                    has = ~np.isnan(s1)
                    iv_path[has] = s1[has]
                    age_ref[has] = s2[has]
            live = ~np.isnan(iv_path) & (idx_end <= exp) & ((idx_end - age_ref) <= MAX_IV_AGE_MS)
            if first_ts.get(n, 1 << 62) >= t0:
                live &= idx_end > first_ts[n]
            if not live.any():
                continue
            mi = np.nonzero(live)[0]
            T = (exp - idx_end[mi]) / 1000.0 / YEAR_SECONDS
            F = S[mi]
            mid = black76(F, k, T, iv_path[mi], is_call)
            ob = otm_bucket(k, F, is_call)
            db = dte_bucket((exp - idx_end[mi]) / 86_400_000.0)
            half = year_tab[ob, db] * F
            bid = np.maximum(mid - half, 0.0)
            ask = mid + half
            B, A = sc(bid), sc(ask)
            right = "call" if is_call else "put"
            ks = int(round(k * 10000))
            exp_s = datetime.fromtimestamp(exp / 1000, tz=timezone.utc).strftime("%Y%m%d")
            name = f"{d:%Y%m%d}_{low}_minute_quote_european_{right}_{ks}_{exp_s}.csv"
            q_entries[name] = "\n".join(f"{mm*60000},{b},{b},{b},{b},10,{a},{a},{a},{a},10"
                                        for mm, b, a in zip(mi, B, A)) + "\n"
            if g is not None and len(by_inst[n]):
                tt = by_inst[n]
                tm = (tt.ts.values - t0) // 60_000
                usd = tt.price.values * tt["index"].values
                bars = pd.DataFrame({"m": tm, "p": usd, "v": tt.amount.values}).groupby("m").agg(
                    o=("p", "first"), h=("p", "max"), l=("p", "min"), c=("p", "last"), v=("v", "sum"))
                bars = bars[(t0 + (bars.index.values + 1) * 60_000) <= exp]
                if len(bars):
                    tname = name.replace("_quote_", "_trade_")
                    t_entries[tname] = "\n".join(
                        f"{mm*60000},{o},{h},{l},{c},{v:.4f}" for mm, o, h, l, c, v in
                        zip(bars.index.values, sc(bars.o), sc(bars.h), sc(bars.l), sc(bars.c), bars.v.values)) + "\n"
            chain_rows.append(f"{n},{exp_s},{k:g},{right},{first_ts[n]},{mi[0]*60000 + t0}")
        ds = f"{d:%Y%m%d}"
        if q_entries:
            write_zip(opt_dir / f"{ds}_quote_european.zip", q_entries)
        if t_entries:
            write_zip(opt_dir / f"{ds}_trade_european.zip", t_entries)
        (alt / "chain" / low).mkdir(parents=True, exist_ok=True)
        (alt / "chain" / low / f"{ds}.csv").write_text(
            "instrument,expiry,strike,right,first_trade_ms,first_quote_ms\n" + "\n".join(chain_rows) + "\n")
        _write_flow(alt / "flow" / low / f"{ds}.csv", trades, t0)
        # roll volatility state to the end of the day; forget expired contracts
        for n in list(meta):
            if meta[n][0] < t0:
                meta.pop(n)
                first_ts.pop(n, None)
                iv_state.pop(n, None)
                iv_state.pop(("prev", n), None)
                continue
            st = iv_state.get(n)
            if st:
                iv_state[("prev", n)] = st
        stats.append((ds, len(trades), len(q_entries), len(t_entries)))
        if d.day == 1 or d == days[-1]:
            print(f"  {cur} {ds}: {len(trades)} trades, {len(q_entries)} quoted contracts, {len(t_entries)} traded", flush=True)
    return stats


def _update_state(cur, d, meta, first_ts, iv_state, keep=False):
    p = RAW / "options" / cur / f"{d:%Y}" / f"{d:%Y%m%d}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["ts", "instrument", "price", "amount", "side", "index", "mark", "iv", "iv_mark", "block"])
    t = pd.read_parquet(p).sort_values("ts")
    t = t[t.instrument.map(is_option)]
    new = [n for n in t.instrument.unique() if n not in meta]
    if new:
        meta.update(parse_instrument(new))
    first = t.groupby("instrument").ts.min()
    for n, ts in first.items():
        if n not in first_ts:
            first_ts[n] = int(ts)
    exp = t.instrument.map(lambda n: meta[n][0]).values
    k = t.instrument.map(lambda n: meta[n][1]).values
    is_call = t.instrument.map(lambda n: meta[n][2]).values
    T = (exp - t.ts.values) / 1000.0 / YEAR_SECONDS
    F = t["index"].values
    mark_usd = t["mark"].values * F
    iv = implied_vol(mark_usd, F, k, T, is_call)
    fallback = t["iv"].values / 100.0
    iv = np.where(np.isnan(iv), fallback, iv)
    iv = np.where((iv > 0.05) & (iv < 5.0), iv, np.nan)
    t = t.assign(iv_mark=iv, exp=exp, strike=k, is_call=is_call, T=T)
    last = t[t.iv_mark.notna()].groupby("instrument").agg(ts=("ts", "last"), iv=("iv_mark", "last"))
    for n, row in last.iterrows():
        iv_state[n] = (int(row.ts), float(row.iv))
    if not keep:
        for n in last.index:
            iv_state[("prev", n)] = iv_state[n]
    return t


def _write_flow(path, t, t0):
    """Signed option order flow per minute (Deribit aggressor side), vega-weighted, block trades excluded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not len(t):
        path.write_text("minute_ms,nbp_vega,nbp_atm_vega,nbp_contracts,trades,volume\n")
        return
    t = t[(t.block == 0) & t.iv_mark.notna()]
    v = vega(t["index"].values, t.strike.values, t["T"].values, t.iv_mark.values) / 100.0  # USD per vol point
    otm = np.abs(t.strike.values / t["index"].values - 1.0)
    atm = (otm < 0.05) & (t["T"].values * 365.0 <= 30.0)
    df = pd.DataFrame({"m": (t.ts.values - t0) // 60_000, "nv": t.side.values * t.amount.values * v,
                       "na": np.where(atm, t.side.values * t.amount.values * v, 0.0),
                       "nc": t.side.values * t.amount.values, "n": 1, "vol": t.amount.values})
    g = df.groupby("m").agg(nv=("nv", "sum"), na=("na", "sum"), nc=("nc", "sum"), n=("n", "sum"), vol=("vol", "sum"))
    path.write_text("minute_ms,nbp_vega,nbp_atm_vega,nbp_contracts,trades,volume\n" + "\n".join(
        f"{m*60000},{a:.2f},{b:.2f},{c:.4f},{n},{vv:.4f}" for m, a, b, c, n, vv in
        zip(g.index.values, g.nv.values, g.na.values, g.nc.values, g.n.values, g.vol.values)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--ws", default="/mnt/quant_storage/quant_workspace/lean_workspace_v2")
    ap.add_argument("--second-start")
    ap.add_argument("--second-end")
    ap.add_argument("--skip-underlying", action="store_true")
    ap.add_argument("--underlying-only", action="store_true")
    ap.add_argument("--spread-mult", type=float, default=1.0, help="sensitivity: scale quoted spreads")
    a = ap.parse_args()
    ws = Path(a.ws)
    s = datetime.strptime(a.start, "%Y-%m-%d")
    e = datetime.strptime(a.end, "%Y-%m-%d")
    days = [s + timedelta(days=i) for i in range((e - s).days + 1)]
    sec_days = []
    if a.second_start:
        ss = datetime.strptime(a.second_start, "%Y-%m-%d")
        se = datetime.strptime(a.second_end, "%Y-%m-%d")
        sec_days = [ss + timedelta(days=i) for i in range((se - ss).days + 1)]
    perp = load_perp_minutes(a.currency, s - timedelta(days=2), e + timedelta(days=1))
    if not a.skip_underlying:
        build_underlying(a.currency, perp, days, ws, sec_days)
        print(f"{a.currency}: underlying files written for {len(days)} days")
    if a.underlying_only:
        return
    stats = build_options(a.currency, perp, days, ws, SpreadModel(a.currency, a.spread_mult))
    dl = RAW / "delivery" / f"{a.currency.lower()}_usd.csv"
    out = ws / "data" / "alternative" / "deribit" / "delivery"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.currency.lower()}.csv").write_text(dl.read_text())
    fund = RAW / "funding" / f"{a.currency}-PERPETUAL.csv"
    if fund.exists():
        fdir = ws / "data" / "alternative" / "deribit" / "funding"
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / fund.name).write_text(fund.read_text())
    print(f"{a.currency}: options done, {sum(x[2] for x in stats)} contract-days quoted")


if __name__ == "__main__":
    main()
