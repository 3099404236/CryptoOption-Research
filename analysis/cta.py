"""
Classic futures (CTA-style) strategies on the Deribit BTC / ETH perpetuals, at three bar sizes (1h, 4h, 1d).

Engine (vectorised, no look-ahead): the signal is computed on a bar's close and the position is traded at
the next bar's open; P&L = position x perpetual return; costs = 0.05 % taker fee + 1 bp slippage per unit of
turnover; the perpetual's hourly funding is paid/received on the position (longs pay positive funding).
Position = direction (-1 / 0 / +1) x volatility scaling (target 40 % a year from the last 30 days, capped at
2x), one unit = equity. Idle cash earns nothing (compare with T-bills).

Families (parameters on the in-sample years only, 2019-2023 / ETH 2020-2023; 2024-2026 is out-of-sample):
  TSMOM_L     time-series momentum (Moskowitz, Ooi & Pedersen 2012): sign of the last L days' return
  MA_f_s      moving-average crossover: long when MA(f) > MA(s), short (or flat) otherwise
  DONCHIAN_n  breakout: long above the n-bar high, short below the n-bar low, exit at the opposite n/2 channel
  RSI2        short-term mean reversion (Connors): long when RSI(2) < 10 and close > MA(200), exit when RSI(2) > 70;
              short when RSI(2) > 90 and close < MA(200), exit when RSI(2) < 30
  each trend rule in a long/short and a long/flat version; benchmarks: buy & hold, vol-targeted buy & hold
Usage: python cta.py
"""
import glob
import math

import numpy as np
import pandas as pd

R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
OUT = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/"
FEE = 0.0005 + 0.0001
SPLIT = pd.Timestamp("2024-01-01")
RF = {2019: 0.021, 2020: 0.004, 2021: 0.001, 2022: 0.017, 2023: 0.050, 2024: 0.051, 2025: 0.042, 2026: 0.036}


def load(cur):
    p = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(R + f"perp_1m/{cur}-PERPETUAL/*.parquet"))])
    p = p.drop_duplicates("ts").sort_values("ts")
    p.index = pd.to_datetime(p.ts, unit="ms")
    f = pd.read_csv(R + f"funding/{cur}-PERPETUAL.csv")
    f = f[f.interest_1h.notna()]
    fund = pd.Series(f.interest_1h.values, index=pd.to_datetime(f.ts, unit="ms")).sort_index()
    return p, fund


def bars(p, fund, rule):
    b = p.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    fb = fund.resample(rule).sum().reindex(b.index).fillna(0.0)          # funding paid during the bar (per unit long)
    b["fund"] = fb
    return b


def vol_scale(b, per_year):
    r = np.log(b.close).diff()
    n = int(round(30 * per_year / 365))
    vol = r.rolling(n, min_periods=n // 2).std() * math.sqrt(per_year)
    return (0.40 / vol).clip(upper=2.0)


def rsi(c, n=2):
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def run(b, direction, per_year, scale=True):
    """direction: Series in {-1,0,1} decided at the bar close -> held over the next bar."""
    pos = direction.fillna(0.0) * (vol_scale(b, per_year) if scale else 1.0)
    pos = pos.fillna(0.0).shift(1).fillna(0.0)                         # trade at next bar open
    ret = b.open.shift(-1) / b.open - 1                                 # open-to-open return of the bar held
    turnover = pos.diff().abs().fillna(pos.abs())
    pnl = pos * ret - turnover * FEE - pos * b.fund
    return pnl.fillna(0.0), pos


def stats(pnl, per_year):
    if len(pnl) < 10:
        return None
    eq = (1 + pnl).cumprod()
    yrs = len(pnl) / per_year
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = pnl.std() * math.sqrt(per_year)
    mdd = (1 - eq / eq.cummax()).max()
    rf = np.mean([RF.get(y, 0.03) for y in pnl.index.year])
    return dict(cagr=cagr, vol=vol, sharpe=(cagr - rf) / vol if vol > 0 else 0.0, mdd=mdd, rf=rf)


def strategies(b, per_bar_day):
    """yield (name, direction series). Lookbacks are in days and converted to bars."""
    c = b.close
    d2b = lambda days: max(1, int(round(days * per_bar_day)))
    yield "BUY_HOLD", pd.Series(1.0, index=c.index)
    for L in (7, 14, 28, 56, 112, 224):
        s = np.sign(c / c.shift(d2b(L)) - 1)
        yield f"TSMOM_{L}d", s
        yield f"TSMOM_{L}d_longflat", s.clip(lower=0)
    for f, sl in ((5, 20), (10, 50), (20, 100), (50, 200)):
        s = np.sign(c.rolling(d2b(f)).mean() - c.rolling(d2b(sl)).mean())
        yield f"MA_{f}_{sl}d", s
        yield f"MA_{f}_{sl}d_longflat", s.clip(lower=0)
    for n in (20, 55):
        hi, lo = b.high.rolling(d2b(n)).max().shift(1), b.low.rolling(d2b(n)).min().shift(1)
        hx, lx = b.high.rolling(d2b(n / 2)).max().shift(1), b.low.rolling(d2b(n / 2)).min().shift(1)
        state, out = 0.0, []
        for cl, h, l, h2, l2 in zip(c.values, hi.values, lo.values, hx.values, lx.values):
            if state == 0 and cl > h:
                state = 1.0
            elif state == 0 and cl < l:
                state = -1.0
            elif state == 1 and cl < l2:
                state = 0.0
            elif state == -1 and cl > h2:
                state = 0.0
            out.append(state)
        s = pd.Series(out, index=c.index)
        yield f"DONCHIAN_{n}d", s
        yield f"DONCHIAN_{n}d_longflat", s.clip(lower=0)
    r2, ma = rsi(c, 2), c.rolling(200).mean()
    state, out = 0.0, []
    for x, cl, m in zip(r2.values, c.values, ma.values):
        if state == 0 and x < 10 and cl > m:
            state = 1.0
        elif state == 0 and x > 90 and cl < m:
            state = -1.0
        elif state == 1 and x > 70:
            state = 0.0
        elif state == -1 and x < 30:
            state = 0.0
        out.append(state)
    yield "RSI2_meanrev", pd.Series(out, index=c.index)


def main():
    rows = []
    for cur in ("BTC", "ETH"):
        p, fund = load(cur)
        start = pd.Timestamp("2019-01-01" if cur == "BTC" else "2020-01-01")
        for rule, per_year, per_day in (("1h", 24 * 365, 24), ("4h", 6 * 365, 6), ("1D", 365, 1)):
            b = bars(p, fund, rule)
            for name, direction in strategies(b, per_day):
                for scaled in (True, False):
                    if name == "BUY_HOLD" and not scaled:
                        pass
                    pnl, pos = run(b, direction, per_year, scale=scaled)
                    pnl = pnl[pnl.index >= start]
                    a, o = stats(pnl[pnl.index < SPLIT], per_year), stats(pnl[pnl.index >= SPLIT], per_year)
                    if a and o:
                        trades = (pos[pos.index >= start].diff().abs() > 1e-9).sum()
                        rows.append(dict(cur=cur, bar=rule, strategy=name, voltarget=scaled, trades_per_year=trades / (len(pnl) / per_year),
                                         **{"is_" + k: v for k, v in a.items()}, **{"oos_" + k: v for k, v in o.items()}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "cta_results.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    cols = ["bar", "strategy", "voltarget", "trades_per_year", "is_cagr", "is_sharpe", "is_mdd", "oos_cagr", "oos_sharpe", "oos_mdd"]
    for cur, g in df.groupby("cur"):
        print(f"\n==================== {cur} ====================")
        bh = g[(g.strategy == "BUY_HOLD") & (~g.voltarget) & (g.bar == "1D")].iloc[0]
        print(f"buy & hold: in-sample CAGR {bh.is_cagr:+.1%} Sharpe {bh.is_sharpe:.2f} maxDD {bh.is_mdd:.0%} | out-of-sample CAGR {bh.oos_cagr:+.1%} Sharpe {bh.oos_sharpe:.2f} maxDD {bh.oos_mdd:.0%}")
        for bar, gb in g.groupby("bar"):
            t = gb.sort_values("is_sharpe", ascending=False)
            print(f"\n-- {bar} bars: top 8 by IN-SAMPLE Sharpe (of {len(t)})")
            print(t[cols].head(8).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
            best = t.iloc[0]
            print(f"   chosen on in-sample: {best.strategy} voltarget={best.voltarget} -> OUT-OF-SAMPLE CAGR {best.oos_cagr:+.1%}, "
                  f"Sharpe {best.oos_sharpe:.2f}, maxDD {best.oos_mdd:.0%}; median OOS Sharpe of all {len(t)} variants {t.oos_sharpe.median():.2f}")


if __name__ == "__main__":
    main()
