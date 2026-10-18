"""Did any LEAN option market fill happen on a broken model quote (e.g. after a Deribit mark-price glitch)?
A quote is suspicious when its mid at the fill minute is < 1/3 or > 3x the mids 10 minutes before AND after.
Impact = cash difference between the fill and the average neighbour mid (positive = flattering the strategy)."""
import glob, json, re, sys, zipfile
from datetime import datetime, timezone, timedelta
from collections import defaultdict
sys.path.insert(0, "/mnt/quant_storage/quant_workspace/lean_workspace_v2")
from verify_v2 import OPT_RE, DATA, zentry

def mid_at(coin, sym, t):
    m = OPT_RE.match(sym)
    yy, mm, dd, cp, k = m.group(2), m.group(3), m.group(4), m.group(5), int(m.group(6))
    start = t - timedelta(minutes=1)
    day = start.strftime("%Y%m%d")
    name = f"{day}_{coin.lower()}_minute_quote_european_{'call' if cp == 'C' else 'put'}_{int(round(k / 1000.0 * 10000))}_20{yy}{mm}{dd}.csv"
    text = zentry(str(DATA / "option/usa/minute" / coin.lower() / f"{day}_quote_european.zip"), name)
    if not text:
        return None
    ms = int((start - datetime(start.year, start.month, start.day)).total_seconds() * 1000)
    last = None
    for line in text.splitlines():
        p = line.split(",")
        if int(p[0]) > ms:
            break
        last = p
    if not last:
        return None
    b, a = int(last[4]) / 1e4, int(last[9]) / 1e4
    return 0.5 * (b + a) if a > 0 else None

W = "/mnt/quant_storage/quant_workspace/lean_workspace_v2/results/"
for suite in ["intraday_full", "volsell_finalists", "volsell_scaled"]:
    flagged, impact, n = 0, 0.0, 0
    per = defaultdict(float)
    for run in json.load(open(W + suite + ".json")):
        if not run.get("ok"):
            continue
        evf = glob.glob(run["folder"] + "/*-order-events.json")
        if not evf:
            continue
        orders = json.load(open([f for f in glob.glob(run["folder"] + "/*.json") if not re.search(r"(order-events|summary|data-monitor)", f)][0])).get("orders", {})
        tags = {int(k): (o.get("tag") or "") for k, o in orders.items()}
        for e in json.load(open(evf[0])):
            if e.get("status") != "filled":
                continue
            sym = (e.get("symbolValue") or "").strip()
            if not OPT_RE.match(sym) or (e.get("message") or "").startswith("Deribit cash settlement"):
                continue
            if tags.get(int(e["orderId"]), "").startswith("MMFILL@"):
                continue
            n += 1
            t = datetime.fromtimestamp(e["time"], tz=timezone.utc).replace(tzinfo=None)
            m0 = mid_at(run["coin"], sym, t)
            mb, ma = mid_at(run["coin"], sym, t - timedelta(minutes=10)), mid_at(run["coin"], sym, t + timedelta(minutes=10))
            if m0 is None or mb is None or ma is None or min(mb, ma) <= 0:
                continue
            if m0 < min(mb, ma) / 3 or m0 > 3 * max(mb, ma):
                flagged += 1
                q, px = float(e["fillQuantity"]), float(e["fillPrice"])
                fair = 0.5 * (mb + ma)
                d = -q * (px - fair)
                impact += d
                per[run["strategy"] + " " + run["coin"]] += d
    print(f"{suite}: {n} market fills, {flagged} on suspicious quotes, net impact ${impact:,.0f}  {dict(per)}")
