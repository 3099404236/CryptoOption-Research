"""Equal-capital combination of the option-selling strategies that made money in both periods plus funding carry.
Sleeves: BTC straddle/fri/daily, BTC put30/mon/HAR, ETH strangle25/mon/HAR (LEAN daily equity, 25 % or 100 %
notional per leg) and BTC funding carry (always on, return on 1.2x capital, from funding_carry.py's definition).
Weights fixed in advance (equal capital); no idle-cash interest anywhere."""
import json, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/quant_storage/quant_workspace/lean_workspace_v2")
from run_suite import equity_series
W = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/"
R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
PICK = {("BTC", "straddle", "fri", "daily", "none"): "BTC straddle", ("BTC", "put30", "mon", "none", "har1.10"): "BTC put",
        ("ETH", "strangle25", "mon", "none", "har1.10"): "ETH strangle"}

def sleeves(suite):
    out = {}
    for r in json.load(open(W + suite + ".json")):
        key = (r["coin"], r["structure"], r["entry"], r["hedge"], r["filter"])
        if key not in PICK or not r.get("ok"):
            continue
        main = [f for f in __import__("glob").glob(r["folder"] + "/*.json") if not re.search(r"(order-events|summary|data-monitor)", f)][0]
        s = pd.Series(dict(equity_series(json.load(open(main)))))
        s.index = pd.to_datetime(s.index)
        out.setdefault(PICK[key], []).append(s.pct_change().dropna())
    return {k: pd.concat(v).sort_index().groupby(level=0).last() for k, v in out.items()}   # IS run ends at 00:00 of the OOS start

f = pd.read_csv(R + "funding/BTC-PERPETUAL.csv")
f = f[f.interest_1h.notna()]
carry = pd.Series(f.interest_1h.values, index=pd.to_datetime(f.ts, unit="ms")).resample("D").sum() / 1.2
carry.index = carry.index.normalize()

def stats(r):
    yrs = len(r) / 365.0
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(365)
    eq = (1 + r).cumprod()
    mdd = (1 - eq / eq.cummax()).max()
    return f"CAGR {cagr:+6.2%}  vol {vol:5.2%}  raw Sharpe {cagr / vol if vol else 0:5.2f}  maxDD {mdd:6.2%}"

for suite, label in (("volsell_finalists", "25 % notional"), ("volsell_scaled", "100 % notional")):
    sl = sleeves(suite)
    df = pd.DataFrame(sl)
    df["BTC carry"] = carry.reindex(df.index).fillna(0.0)
    print(f"\n=== option sleeves at {label} (+ BTC funding carry), equal capital ===")
    print("daily-return correlation:\n" + df.corr().round(2).to_string())
    for lab, sub in (("2019/20-2023", df[df.index < "2024-01-01"].dropna()), ("2024-2026", df[df.index >= "2024-01-01"].dropna())):
        print(f"-- {lab}")
        for c in df.columns:
            print(f"   {c:13s} {stats(sub[c])}")
        print(f"   {'3 options':13s} {stats(sub[list(sl)].mean(axis=1))}")
        print(f"   {'3 opt+carry':13s} {stats(sub.mean(axis=1))}")
