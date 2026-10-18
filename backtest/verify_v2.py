"""
Independent checks of V2 backtest outputs against the raw data (run after run_suite.py).

For every run folder in the results file:
  1. no simulated early assignment / physical delivery events
  2. every settlement fill equals the option's intrinsic value at Deribit's official delivery price
  3. every option market fill equals the ask (buys) or bid (sells) of that contract's quote bar for the
     minute that ended at the fill time (spread multiplier taken into account)
  4. no option order before the contract's first real Deribit trade (chain files)
  5. option fees equal min(0.03% x perpetual price, 12.5% x fill price) x quantity

Usage: python verify_v2.py <results.json> [<report.json>]
"""
import csv
import glob
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

WS = Path("/mnt/quant_storage/quant_workspace/lean_workspace_v2")
DATA = WS / "data"
OPT_RE = re.compile(r"^(BTC|ETH)\s+(\d{2})(\d{2})(\d{2})([CP])(\d{8,9})$")   # 9 digits for strikes >= 100,000
_zip_cache = {}


def zentry(path, entry):
    key = (path, entry)
    if key not in _zip_cache:
        if len(_zip_cache) > 64:
            _zip_cache.clear()
        try:
            with zipfile.ZipFile(path) as z:
                _zip_cache[key] = z.read(entry).decode() if entry in z.namelist() else None
        except FileNotFoundError:
            _zip_cache[key] = None
    return _zip_cache[key]


def quote_at(coin, sym, t):
    """bid/ask of the quote bar that ended at t (UTC naive)."""
    m = OPT_RE.match(sym)
    yy, mm, dd, cp, k = m.group(2), m.group(3), m.group(4), m.group(5), int(m.group(6))
    start = t - timedelta(minutes=1)
    day = start.strftime("%Y%m%d")
    right = "call" if cp == "C" else "put"
    strike = int(round(k / 1000.0 * 10000))
    name = f"{day}_{coin.lower()}_minute_quote_european_{right}_{strike}_20{yy}{mm}{dd}.csv"
    text = zentry(str(DATA / "option/usa/minute" / coin.lower() / f"{day}_quote_european.zip"), name)
    if not text:
        return None
    ms = str(int((start - datetime(start.year, start.month, start.day)).total_seconds() * 1000))
    last = None
    for line in text.splitlines():
        p = line.split(",")
        if int(p[0]) > int(ms):
            break
        last = p
    if not last:
        return None
    return int(last[4]) / 10000.0, int(last[9]) / 10000.0


def trade_bar_at(coin, sym, t):
    """(open, high, low, close, volume) of the real trade bar that ended at t, or None."""
    m = OPT_RE.match(sym)
    yy, mm, dd, cp, k = m.group(2), m.group(3), m.group(4), m.group(5), int(m.group(6))
    start = t - timedelta(minutes=1)
    day = start.strftime("%Y%m%d")
    right = "call" if cp == "C" else "put"
    name = f"{day}_{coin.lower()}_minute_trade_european_{right}_{int(round(k / 1000.0 * 10000))}_20{yy}{mm}{dd}.csv"
    text = zentry(str(DATA / "option/usa/minute" / coin.lower() / f"{day}_trade_european.zip"), name)
    if not text:
        return None
    ms = int((start - datetime(start.year, start.month, start.day)).total_seconds() * 1000)
    for line in text.splitlines():
        p = line.split(",")
        if int(p[0]) == ms:
            return tuple(int(x) / 1e4 for x in p[1:5]) + (float(p[5]),)
    return None


def perp_close(coin, t):
    start = t - timedelta(minutes=1)
    day = start.strftime("%Y%m%d")
    text = zentry(str(DATA / "equity/usa/minute" / coin.lower() / f"{day}_trade.zip"), f"{day}_{coin.lower()}_minute_trade.csv")
    if not text:
        return None
    idx = (start.hour * 60 + start.minute)
    p = text.splitlines()[idx].split(",")
    return int(p[4]) / 10000.0


def first_trades(coin):
    out = {}
    for f in sorted(glob.glob(str(DATA / "alternative/deribit/chain" / coin.lower() / "*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                out.setdefault(r["instrument"], int(r["first_trade_ms"]))
    return out


def deribit_name(coin, sym):
    m = OPT_RE.match(sym)
    yy, mm, dd, cp, k = m.group(2), m.group(3), m.group(4), m.group(5), int(m.group(6))
    mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"][int(mm) - 1]
    return f"{coin}-{int(dd)}{mon}{yy}-{k // 1000 if k % 1000 == 0 else k / 1000:g}-{cp}"


def check(run):
    folder, coin = Path(run["folder"]), run["coin"]
    mult = float(run.get("spread_mult", 1.0))
    evf = list(folder.glob("*-order-events.json"))
    ev = json.loads(evf[0].read_text()) if evf else []      # LEAN writes no file when there were no orders
    main = [f for f in folder.glob("*.json") if not re.search(r"(order-events|summary|data-monitor)", f.name)][0]
    orders = json.loads(main.read_text()).get("orders", {})
    order_type = {int(k): o.get("type") for k, o in orders.items()}
    order_tag = {int(k): (o.get("tag") or "") for k, o in orders.items()}
    delivery = {r["date"]: float(r["price"]) for r in csv.DictReader(open(DATA / "alternative/deribit/delivery" / f"{coin.lower()}.csv"))}
    firsts = first_trades(coin)
    res = defaultdict(int)
    bad = defaultdict(list)
    pos = defaultdict(float)                         # running positions from fills
    for e in sorted(ev, key=lambda x: (x["time"], x["orderId"])):
        if e.get("status") != "filled":
            continue
        sym0 = (e.get("symbolValue") or "").strip()
        t0 = datetime.fromtimestamp(e["time"], tz=timezone.utc).replace(tzinfo=None)
        tag0 = order_tag.get(int(e["orderId"]), "")
        if tag0.startswith(("flatten after settlement", "flatten hedge")):
            live = [s_ for s_, q_ in pos.items() if abs(q_) > 1e-9 and OPT_RE.match(s_) and
                    datetime(2000 + int(OPT_RE.match(s_).group(2)), int(OPT_RE.match(s_).group(3)),
                             int(OPT_RE.match(s_).group(4))).date() > t0.date()]
            if live:
                bad["hedge_flattened_with_open_options"].append((str(t0), live[:2]))
        pos[sym0] += float(e["fillQuantity"])
    for e in ev:
        if e.get("status") != "filled":
            continue
        sym = (e.get("symbolValue") or "").strip()
        msg = e.get("message") or ""
        t = datetime.fromtimestamp(e["time"], tz=timezone.utc).replace(tzinfo=None)
        if "Assign" in msg or "Exercise" in msg:
            res["assignment_or_exercise_events"] += 1
            bad["assignment"].append((str(t), sym, msg[:80]))
        if not OPT_RE.match(sym):
            continue
        qty, px = float(e["fillQuantity"]), float(e["fillPrice"])
        m = OPT_RE.match(sym)
        k = int(m.group(6)) / 1000.0
        if msg.startswith("Deribit cash settlement"):
            res["settlements"] += 1
            d = delivery.get(f"20{m.group(2)}-{m.group(3)}-{m.group(4)}")
            intrinsic = max(d - k, 0.0) if m.group(5) == "C" else max(k - d, 0.0)
            if abs(intrinsic - px) > 1e-4 or t != datetime(t.year, t.month, t.day, 8, 0):
                bad["settlement"].append((str(t), sym, px, intrinsic))
            continue
        res["option_fills"] += 1
        first = firsts.get(deribit_name(coin, sym))
        if first is None or first > e["time"] * 1000:
            bad["before_first_trade"].append((str(t), sym, first))
        tag = order_tag.get(int(e["orderId"]), "")
        if tag.startswith("MMFILL@"):                # market-making fill: at our quote, through a real print
            res["mm_fills"] += 1
            tb = trade_bar_at(coin, sym, t)
            quoted = float(tag.split("@")[1].split()[0])
            ok = tb is not None and abs(px - quoted) < 1e-4 and                 ((qty < 0 and tb[1] >= quoted) or (qty > 0 and tb[2] <= quoted)) and abs(qty) <= 0.25 * tb[4] + 1e-9
            if not ok:
                bad["mm_fill_not_supported_by_print"].append((str(t), sym, qty, px, tb))
            continue
        q = quote_at(coin, sym, t)
        if q is None:
            res["fills_without_quote_bar"] += 1
            continue
        if e.get("orderFeeAmount") is not None and qty != 0:
            s = perp_close(coin, t)
            if s:
                fee = min(0.0003 * s, 0.125 * px) * abs(qty)
                if abs(fee - float(e["orderFeeAmount"])) > max(0.02 * fee, 0.02):
                    res["fee_mismatch"] += 1
        bid, ask = q
        mid = 0.5 * (bid + ask)
        expect = mid + mult * ((ask if qty > 0 else bid) - mid)
        if order_type.get(int(e["orderId"])) == 1:   # limit orders (market making) fill at their limit price
            res["limit_fills"] += 1
            continue
        res["market_fills_checked"] += 1
        if abs(px - expect) > max(1e-4, 1e-6 * expect):
            bad["fill_not_at_quote"].append((str(t), sym, qty, px, round(expect, 4)))
    return {"run": f"{run['project']}/{run['strategy']}/{coin}/{run['start']}..{run['end']}/x{mult}",
            "counts": dict(res), "violations": {k: len(v) for k, v in bad.items()},
            "examples": {k: v[:3] for k, v in bad.items()}}


def main():
    runs = [r for r in json.loads(Path(sys.argv[1]).read_text()) if r.get("ok")]
    report = [check(r) for r in runs]
    for r in report:
        print(r["run"], r["counts"], "VIOLATIONS:", r["violations"] or "none")
        for k, ex in r["examples"].items():
            print("   e.g.", k, ex)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
