"""Re-price every LEAN option market fill with real Deribit prints of the same contract (+-60 min):
sells vs taker-sell prints (real bids), buys vs taker-buy prints (real asks). USD = coin price x index."""
import json, re, glob
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np, pandas as pd

WS = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/"
R = "/mnt/quant_storage/quant_workspace/deribit_raw/"
OPT_RE = re.compile(r"^(BTC|ETH)\s+(\d{2})(\d{2})(\d{2})([CP])(\d{8,9})$")
MONS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
_day = {}

def day_trades(cur, d):
    key = (cur, d)
    if key not in _day:
        if len(_day) > 8: _day.clear()
        try:
            t = pd.read_parquet(R + f"options/{cur}/{d[:4]}/{d}.parquet", columns=["ts","instrument","price","amount","side","index","mark","combo","liq"])
            _day[key] = t[(t.combo == 0) & (t.liq == 0)]
        except Exception:
            _day[key] = None
    return _day[key]

def name(cur, sym):
    m = OPT_RE.match(sym)
    yy, mm, dd, cp, k = m.group(2), m.group(3), m.group(4), m.group(5), int(m.group(6))
    ks = k // 1000 if k % 1000 == 0 else k / 1000
    return f"{cur}-{int(dd)}{MONS[int(mm)-1]}{yy}-{ks:g}-{cp}"

def check(run):
    folder, cur = run["folder"], run["coin"]
    ev = json.load(open(glob.glob(folder + "/*-order-events.json")[0]))
    out = defaultdict(float); n = defaultdict(int); rel = []
    for e in ev:
        if e.get("status") != "filled": continue
        sym = (e.get("symbolValue") or "").strip()
        if not OPT_RE.match(sym) or (e.get("message") or "").startswith("Deribit cash settlement"): continue
        q, px, ts = float(e["fillQuantity"]), float(e["fillPrice"]), int(e["time"] * 1000)
        d = datetime.fromtimestamp(e["time"], tz=timezone.utc).strftime("%Y%m%d")
        t = day_trades(cur, d)
        n["fills"] += 1
        if t is None: continue
        x = t[(t.instrument == name(cur, sym)) & (t.ts >= ts - 3600000) & (t.ts <= ts + 3600000)]
        side = x[x.side == (-1 if q < 0 else 1)]
        if len(x): # Deribit mark (mid proxy) at the print closest in time
            j = (x.ts - ts).abs().argmin()
            out["lean_minus_mark_signed"] += -q * (px - x.mark.iloc[j] * x["index"].iloc[j])
            n["with_mark"] += 1
        if side.empty: continue
        real = (side.price * side["index"] * side.amount).sum() / side.amount.sum()
        # cash effect of using LEAN's price instead of the real print: positive = LEAN was more favourable
        out["lean_minus_real"] += -q * (px - real)
        out["abs_notional_prem"] += abs(q) * real
        n["matched"] += 1
        rel.append((px - real) / real if real > 0 else np.nan)
    rel = np.array([r for r in rel if np.isfinite(r)])
    return n, out, rel

for suite in ["weekly_btc", "weekly_eth", "intraday_eth_1y"]:
    print(f"\n=== {suite}")
    print(f"{'strategy':28s} {'fills':>6s} {'matched':>7s} {'LEAN-real $':>12s} {'LEAN-mark $':>12s} {'median rel':>10s} {'mean rel':>9s} {'LEAN pnl $':>11s}")
    for run in json.load(open(WS + f"results/{suite}.json")):
        if not run.get("ok"): continue
        n, out, rel = check(run)
        pnl = run["metrics"]["end_equity"] - run["metrics"]["start_equity"]
        print(f"{run['strategy']:28s} {n['fills']:6d} {n['matched']:7d} {out['lean_minus_real']:12,.0f} {out['lean_minus_mark_signed']:12,.0f} "
              f"{np.median(rel) if len(rel) else float('nan'):10.2%} {np.mean(rel) if len(rel) else float('nan'):9.2%} {pnl:11,.0f}")
