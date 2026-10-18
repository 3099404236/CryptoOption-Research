"""Plausibility checks of the built LEAN option data (sampled days).

For each sampled day: quoted contracts, share of quote rows with zero bid, median relative spread,
and how far the model mid is from the real Deribit trade prints of the same minute (median absolute
deviation in % of the trade price) - a direct test of the mark-to-model quotes.
Usage: python check_built_data.py <ws> <coin> <every_n_days>
"""
import sys
import zipfile
from pathlib import Path

import numpy as np

ws, coin, every = Path(sys.argv[1]), sys.argv[2].lower(), int(sys.argv[3])
d = ws / "data" / "option" / "usa" / "minute" / coin
files = sorted(d.glob("*_quote_european.zip"))[::every]
print(f"{'day':9s} {'contracts':>9s} {'zero-bid%':>9s} {'rel.spread':>10s} {'|mid-trade|/trade':>18s} {'pairs':>6s}")
for qf in files:
    day = qf.name[:8]
    tf = d / f"{day}_trade_european.zip"
    with zipfile.ZipFile(qf) as zq:
        names = zq.namelist()
        zero = rows = 0
        rel = []
        dev = []
        trades = zipfile.ZipFile(tf) if tf.exists() else None
        tnames = set(trades.namelist()) if trades else set()
        for n in names:
            q = np.array([[int(x) for x in l.split(",")] for l in zq.read(n).decode().split()])
            bid, ask = q[:, 4] / 1e4, q[:, 9] / 1e4
            mid = 0.5 * (bid + ask)
            rows += len(q)
            zero += int((bid <= 0).sum())
            ok = mid > 0
            rel += list(((ask - bid) / mid)[ok])
            tn = n.replace("_quote_", "_trade_")
            if tn in tnames:
                t = np.array([[float(x) for x in l.split(",")] for l in trades.read(tn).decode().split()])
                qm = dict(zip(q[:, 0], mid))
                for ms, close in zip(t[:, 0], t[:, 4] / 1e4):
                    m = qm.get(int(ms))
                    if m and close > 0:
                        dev.append(abs(m - close) / close)
    print(f"{day:9s} {len(names):9d} {100*zero/max(rows,1):8.1f}% {np.median(rel)*100:9.1f}% "
          f"{(np.median(dev)*100 if dev else float('nan')):17.1f}% {len(dev):6d}")
