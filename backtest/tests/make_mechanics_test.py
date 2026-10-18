"""Synthetic 3-day dataset (Sat 2026-10-10 .. Mon 2026-10-12) used only to verify LEAN mechanics:
24/7 UTC sessions, European option file naming, multiplier/lot size, fee model, settlement fill.
Prices are deterministic so every expected number can be checked by hand."""
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ws = Path(sys.argv[1])
data = ws / "data"
days = ["20261010", "20261011", "20261012"]
zips = defaultdict(dict)  # zip path -> {entry name: lines}


def spot(day_idx, minute):
    return 80000.0 + 10.0 * (day_idx * 1440 + minute) / 60.0  # +10 USD per hour


hour_lines = []
for i, d in enumerate(days):
    tr, qt = [], []
    for m in range(1440):
        s = int(round(spot(i, m) * 10000))
        tr.append(f"{m*60000},{s},{s},{s},{s},10")
        qt.append(f"{m*60000},{s-5000},{s-5000},{s-5000},{s-5000},5,{s+5000},{s+5000},{s+5000},{s+5000},5")
        if m % 60 == 0:
            hour_lines.append(f"{d} {m//60:02d}:00,{s},{s},{s},{s},600")
    zips[data / "equity/usa/minute/btc" / f"{d}_trade.zip"][f"{d}_btc_minute_trade.csv"] = tr
    zips[data / "equity/usa/minute/btc" / f"{d}_quote.zip"][f"{d}_btc_minute_quote.csv"] = qt
zips[data / "equity/usa/hour/btc.zip"]["btc.csv"] = hour_lines

# two European options expiring Sunday 2026-10-11 08:00 UTC; quotes only until 08:00 on expiry day
for i, d in enumerate(days[:2]):
    last_minute = 1440 if d == "20261010" else 8 * 60
    for right, strike in (("call", 80000), ("put", 81000)):
        q, t = [], []
        for m in range(last_minute):
            s = spot(i, m)
            intrinsic = max(0.0, s - strike) if right == "call" else max(0.0, strike - s)
            mid = intrinsic + 300.0
            b, a, c = (int(round(x * 10000)) for x in (mid - 20, mid + 20, mid))
            q.append(f"{m*60000},{b},{b},{b},{b},3,{a},{a},{a},{a},3")
            t.append(f"{m*60000},{c},{c},{c},{c},1")
        k = strike * 10000
        base = data / "option/usa/minute/btc"
        zips[base / f"{d}_quote_european.zip"][f"{d}_btc_minute_quote_european_{right}_{k}_20261011.csv"] = q
        zips[base / f"{d}_trade_european.zip"][f"{d}_btc_minute_trade_european_{right}_{k}_20261011.csv"] = t

for path, entries in zips.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, lines in entries.items():
            z.writestr(name, "\n".join(lines) + "\n")
    path.write_bytes(buf.getvalue())
print(f"synthetic mechanics data written: {len(zips)} zip files")
