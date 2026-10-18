"""Week-by-week reconciliation: LEAN SHORT_STRANGLE (BTC) gross P&L vs the independent print-based strangle."""
import json, re, glob
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np, pandas as pd
WS = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/"
OPT_RE = re.compile(r"^(BTC|ETH)\s+(\d{2})(\d{2})(\d{2})([CP])(\d{8,9})$")
ind = pd.read_pickle(WS + "results/indep_weekly.pkl")
for cur, suite in [("BTC", "weekly_btc"), ("ETH", "weekly_eth")]:
    run = [r for r in json.load(open(WS + f"results/{suite}.json")) if r["strategy"] == "SHORT_STRANGLE"][0]
    ev = json.load(open(glob.glob(run["folder"] + "/*-order-events.json")[0]))
    perp = {}
    wk = defaultdict(lambda: dict(cash=0.0, coins=0.0, s=None, entry=None))
    for e in ev:
        if e.get("status") != "filled": continue
        sym = (e.get("symbolValue") or "").strip(); m = OPT_RE.match(sym)
        if not m: continue
        exp = datetime(2000 + int(m.group(2)), int(m.group(3)), int(m.group(4)))
        q, px = float(e["fillQuantity"]), float(e["fillPrice"])
        w = wk[exp]
        w["cash"] += -q * px
        if not (e.get("message") or "").startswith("Deribit cash settlement"):
            w["coins"] += abs(q) / 2
            w["entry"] = datetime.fromtimestamp(e["time"], tz=timezone.utc).replace(tzinfo=None)
    rows = []
    for exp, w in wk.items():
        rows.append(dict(friday=exp, monday=exp - timedelta(days=4), lean_cash=w["cash"], coins=w["coins"], entry=w["entry"]))
    L = pd.DataFrame(rows)
    I = ind[ind.cur == cur].copy()
    I["mondayd"] = pd.to_datetime(I.monday).dt.normalize()
    L["mondayd"] = pd.to_datetime(L.monday).dt.normalize()
    J = L.merge(I[["mondayd", "s0", "P20_bid", "C20_bid", "P20_mark", "C20_mark"]], on="mondayd", how="left")
    J["lean_pct"] = J.lean_cash / (J.coins * J.s0)          # gross P&L per 1-coin strangle, % of spot
    J["ind_pct"] = J.P20_bid + J.C20_bid
    both = J.dropna(subset=["lean_pct", "ind_pct"])
    only = J[J.ind_pct.isna()]
    print(f"\n=== {cur} SHORT_STRANGLE: LEAN weeks {len(L)}, gross cash ${L.lean_cash.sum():,.0f}")
    print(f"common weeks {len(both)}: LEAN mean {both.lean_pct.mean():+.3%}/wk  independent(bid) mean {both.ind_pct.mean():+.3%}/wk  corr {both.lean_pct.corr(both.ind_pct):.2f}")
    print(f"   LEAN gross $ on common weeks {both.lean_cash.sum():,.0f}; independent at LEAN size ${(both.ind_pct * both.coins * both.s0).sum():,.0f}")
    print(f"LEAN-only weeks (no clean 0.20-delta prints Mon 09:30-11:30) {len(only)}: LEAN gross ${only.lean_cash.sum():,.0f}")
    J["year"] = J.friday.dt.year
    for y, g in J.groupby("year"):
        b = g.dropna(subset=["lean_pct", "ind_pct"])
        print(f"  {y}: LEAN gross ${g.lean_cash.sum():>10,.0f}  avg coins {g.coins.mean():5.1f}  common {len(b):2d} wks LEAN {b.lean_pct.mean():+.3%} vs indep {b.ind_pct.mean():+.3%}")
