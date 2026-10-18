"""
Cross-sectional momentum across coins (Liu, Tsyvinski & Wu 2022 style) on Binance USD-M perpetuals.
Universe fixed in advance: 20 large coins that had Binance perpetuals by 2020 (survivorship caveat: they are
coins that are still large today). Daily klines and 8-hourly funding from data.binance.vision.
Every Monday 00:00 UTC: rank coins with >= 60 days of history by their past L-day return (L = 7, 14, 28, 56);
long the top 20 % and short the bottom 20 %, equal weight, 0.5x long + 0.5x short of equity (dollar neutral);
'reversal' = the opposite; 'long-top' = only the long leg at 1x. Costs: 0.05 % taker + 1 bp slippage on
turnover, funding paid by longs / received by shorts. In-sample 2020-07..2023, out-of-sample 2024-2026.
Usage: python xsmom.py download | analyse
"""
import io
import math
import os
import sys
import zipfile
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests

B = "/mnt/quant_storage/quant_workspace/binance_um/"
OUT = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/analysis/"
COINS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "LTC", "LINK", "BCH", "TRX", "AVAX", "ATOM",
         "ETC", "XLM", "FIL", "UNI", "EOS", "XTZ"]
MONTHS = pd.period_range("2019-09", "2026-09", freq="M")
FEE = 0.0005 + 0.0001
RF = {2020: 0.004, 2021: 0.001, 2022: 0.017, 2023: 0.050, 2024: 0.051, 2025: 0.042, 2026: 0.036}


def fetch(args):
    kind, coin, m = args
    sym = f"{coin}USDT"
    path = B + f"{kind}_{sym}_{m}.csv"
    if os.path.exists(path):
        return path
    url = (f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1d/{sym}-1d-{m}.zip" if kind == "k" else
           f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip")
    for _ in range(4):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.open(z.namelist()[0]).read()
            open(path, "wb").write(raw)
            return path
        except Exception:
            continue
    return None


def read(kind, coin):
    frames = []
    for m in MONTHS:
        p = B + f"{kind}_{coin}USDT_{m}.csv"
        if not os.path.exists(p):
            continue
        raw = open(p, "rb").read()
        hdr = 0 if raw[:1].isalpha() else None
        df = pd.read_csv(io.BytesIO(raw), header=hdr)
        if kind == "k":
            df = df.iloc[:, [0, 4]]
            df.columns = ["t", "close"]
        else:
            df = df.iloc[:, [0, 2]] if df.shape[1] >= 3 else df.iloc[:, [0, 1]]
            df.columns = ["t", "rate"]
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames)
    t = df.t.astype(np.int64)
    df.index = pd.to_datetime(np.where(t > 10 ** 14, t // 1000, t), unit="ms")
    return df.drop(columns="t").sort_index()


def analyse():
    close = pd.DataFrame({c: read("k", c).close for c in COINS if read("k", c) is not None})
    close.index = close.index.normalize()
    close = close[~close.index.duplicated()].sort_index()
    fund = pd.DataFrame({c: f.rate.resample("D").sum() for c in COINS if (f := read("f", c)) is not None}).reindex(close.index).fillna(0.0)
    ret = close.pct_change()
    mondays = close.index[(close.index.weekday == 0) & (close.index >= "2020-07-01")]
    out = []
    for L in (7, 14, 28, 56):
        for kind in ("momentum", "reversal", "long-top"):
            w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
            for d in mondays:
                hist = close.loc[:d].iloc[:-1]                          # data before the Monday open (no look-ahead)
                ok = hist.notna().sum() >= 60
                past = (hist.iloc[-1] / hist.iloc[-1 - L] - 1)[ok].dropna()
                if len(past) < 8:
                    continue
                k = max(1, int(round(0.2 * len(past))))
                top, bot = past.nlargest(k).index, past.nsmallest(k).index
                if kind == "reversal":
                    top, bot = bot, top
                row = pd.Series(0.0, index=close.columns)
                if kind == "long-top":
                    row[top] = 1.0 / k
                else:
                    row[top], row[bot] = 0.5 / k, -0.5 / k
                w.loc[d] = row
            w.loc[~w.index.isin(mondays)] = np.nan              # weights change only on rebalance Mondays
            w = w.ffill().fillna(0.0)
            held = w.shift(1).fillna(0.0)                            # weights set at Monday 00:00 earn that day's return
            pnl = (held * ret.fillna(0.0)).sum(axis=1) - (held.diff().abs().sum(axis=1)) * FEE - (held * fund).sum(axis=1)
            pnl = pnl[pnl.index >= "2020-07-01"]
            for lab, x in (("is", pnl[pnl.index < "2024-01-01"]), ("oos", pnl[pnl.index >= "2024-01-01"])):
                eq = (1 + x).cumprod()
                yrs = len(x) / 365
                cagr = eq.iloc[-1] ** (1 / yrs) - 1
                vol = x.std() * math.sqrt(365)
                rf = np.mean([RF.get(y, 0.03) for y in x.index.year])
                out.append(dict(L=L, kind=kind, period=lab, cagr=cagr, vol=vol, sharpe=(cagr - rf) / vol, mdd=(1 - eq / eq.cummax()).max()))
    # benchmark: equal-weight long all coins
    ew = ret.where(close.notna().cumsum() >= 60).mean(axis=1).fillna(0.0)
    ew = ew[ew.index >= "2020-07-01"]
    for lab, x in (("is", ew[ew.index < "2024-01-01"]), ("oos", ew[ew.index >= "2024-01-01"])):
        eq = (1 + x).cumprod()
        cagr = eq.iloc[-1] ** (365 / len(x)) - 1
        out.append(dict(L=0, kind="equal-weight long all", period=lab, cagr=cagr, vol=x.std() * math.sqrt(365),
                        sharpe=(cagr - np.mean([RF.get(y, 0.03) for y in x.index.year])) / (x.std() * math.sqrt(365)),
                        mdd=(1 - eq / eq.cummax()).max()))
    df = pd.DataFrame(out)
    df.to_csv(OUT + "xsmom_results.csv", index=False)
    piv = df.pivot_table(index=["kind", "L"], columns="period", values=["cagr", "sharpe", "mdd"]).round(3)
    pd.set_option("display.width", 200)
    print(f"coins with data: {list(close.columns)}  ({close.index.min().date()} .. {close.index.max().date()})")
    print(piv.to_string())


if __name__ == "__main__":
    if sys.argv[1] == "download":
        os.makedirs(B, exist_ok=True)
        jobs = [(k, c, str(m)) for k in ("k", "f") for c in COINS for m in MONTHS]
        with Pool(16) as p:
            res = p.map(fetch, jobs)
        print(f"files: {sum(r is not None for r in res)}/{len(jobs)}")
    else:
        analyse()
