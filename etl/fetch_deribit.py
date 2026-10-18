"""
Deribit raw data downloader (V2).

Keeps every field the backtest needs and never invents data:
  options     all option trades of a currency, one parquet file per UTC day, with
              coin price, USD index at trade time, Deribit mark price, trade IV,
              aggressor direction, amount, block-trade / liquidation flags
  perp1m      1-minute OHLCV of the {CUR}-PERPETUAL (24/7, UTC) -> the hedge instrument
  perptrades  tick trades of the perpetual -> 1-second bars for the second-level test
  delivery    official daily 08:00 UTC delivery (settlement) prices of the index

Usage (on the workstation, miniconda python):
  python fetch_deribit.py options    --currency BTC --start 2019-01-01 --end 2026-10-16
  python fetch_deribit.py perp1m     --currency BTC --start 2018-12-20 --end 2026-10-17
  python fetch_deribit.py perptrades --currency ETH --start 2026-10-10 --end 2026-10-16
  python fetch_deribit.py delivery   --currency BTC
"""
import argparse
import concurrent.futures
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

RAW = Path(os.environ.get("DERIBIT_RAW", "/mnt/quant_storage/quant_workspace/deribit_raw"))
HISTORY = "https://history.deribit.com/api/v2/public/"
LIVE = "https://www.deribit.com/api/v2/public/"
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "deribit-backtest-etl/2.0"
        proxy = os.environ.get("DERIBIT_PROXY")  # e.g. http://127.0.0.1:7897 on the workstation
        if proxy:
            s.proxies.update({"http": proxy, "https": proxy})
        _local.s = s
    return _local.s


def call(base, method, **params):
    """GET with retries; raises after repeated failures instead of silently skipping data."""
    last = None
    for attempt in range(10):
        try:
            # the history endpoint answers in ~1 s but occasionally hangs; fail fast and retry
            r = session().get(base + method, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()["result"]
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # network hiccup
            last = repr(e)
        time.sleep(min(15, 2 ** attempt))
    raise RuntimeError(f"{method} {params} failed: {last}")


def ms(d):
    return int(d.replace(tzinfo=timezone.utc).timestamp() * 1000)


def days(start, end):
    d = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while d <= e:
        yield d
        d += timedelta(days=1)


def paginate_trades(method, key, value, t0, t1, **extra):
    """All trades in [t0, t1] (ms). Restarts each page at the last timestamp (inclusive) and
    de-duplicates on trade_id, so trades sharing a millisecond across a page boundary are kept."""
    out, seen, start = [], set(), t0
    while True:
        res = call(HISTORY, method, **{key: value}, **extra, start_timestamp=start, end_timestamp=t1,
                   count=10000, sorting="asc", include_old="true")
        page = res.get("trades", [])
        new = [t for t in page if t["trade_id"] not in seen]
        for t in new:
            seen.add(t["trade_id"])
        out += new
        if not res.get("has_more") or not page:
            return out
        last = page[-1]["timestamp"]
        start = last if (new and last > start) else start + 1


# ---------------------------------------------------------------- option trades
def fetch_option_day(cur, day):
    path = RAW / "options" / cur / f"{day:%Y}" / f"{day:%Y%m%d}.parquet"
    if path.exists():
        return f"{cur} {day:%Y-%m-%d} skip"
    t0 = ms(day)
    trades = paginate_trades("get_last_trades_by_currency_and_time", "currency", cur, t0, t0 + 86_400_000 - 1,
                             kind="option")
    rows = [{
        "ts": t["timestamp"],
        "trade_id": t["trade_id"],
        "instrument": t["instrument_name"],
        "price": t["price"],                      # option price in coin units
        "amount": t["amount"],                    # contracts (1 contract = 1 coin)
        "side": 1 if t["direction"] == "buy" else -1,   # aggressor
        "index": t.get("index_price"),
        "mark": t.get("mark_price"),
        "iv": t.get("iv"),
        "block": 1 if t.get("block_trade_id") else 0,
        "liq": 1 if t.get("liquidation") else 0,
        "combo": 1 if t.get("combo_id") else 0,
    } for t in trades if t.get("instrument_name", "").count("-") == 3]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.rename(path)
    return f"{cur} {day:%Y-%m-%d} {len(rows)} trades"


# ---------------------------------------------------------------- perpetual 1m bars
def fetch_perp_1m(cur, start, end):
    inst = f"{cur}-PERPETUAL"
    out_dir = RAW / "perp_1m" / inst
    out_dir.mkdir(parents=True, exist_ok=True)
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    month = datetime(s.year, s.month, 1)
    while month <= e:
        nxt = datetime(month.year + (month.month == 12), month.month % 12 + 1, 1)
        path = out_dir / f"{month:%Y%m}.parquet"
        if not path.exists() or nxt > datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1):
            frames, t = [], ms(month)
            t_end = ms(min(nxt, e + timedelta(days=1)))
            while t < t_end:
                chunk_end = min(t + 4800 * 60_000, t_end)
                r = call(LIVE, "get_tradingview_chart_data", instrument_name=inst,
                         start_timestamp=t, end_timestamp=chunk_end - 1, resolution=1)
                if r.get("ticks"):
                    frames.append(pd.DataFrame({"ts": r["ticks"], "open": r["open"], "high": r["high"],
                                                "low": r["low"], "close": r["close"], "volume": r["volume"]}))
                t = chunk_end
            if frames:
                df = pd.concat(frames).drop_duplicates("ts").sort_values("ts")
                df = df[(df.ts >= ms(month)) & (df.ts < ms(nxt))]
                df.to_parquet(path, index=False)
                print(f"{inst} {month:%Y-%m}: {len(df)} bars", flush=True)
        month = nxt


# ---------------------------------------------------------------- perpetual tick trades
def fetch_perp_trades_day(cur, day):
    inst = f"{cur}-PERPETUAL"
    path = RAW / "perp_trades" / inst / f"{day:%Y%m%d}.parquet"
    if path.exists():
        return f"{inst} {day:%Y-%m-%d} skip"
    t0 = ms(day)
    trades = paginate_trades("get_last_trades_by_instrument_and_time", "instrument_name", inst, t0, t0 + 86_400_000 - 1)
    df = pd.DataFrame({"ts": [t["timestamp"] for t in trades], "price": [t["price"] for t in trades],
                       "amount": [t["amount"] for t in trades],
                       "side": [1 if t["direction"] == "buy" else -1 for t in trades],
                       "index": [t.get("index_price") for t in trades]})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return f"{inst} {day:%Y-%m-%d} {len(df)} trades"


# ---------------------------------------------------------------- perpetual funding
def fetch_funding(cur, start, end):
    """Hourly funding of the perpetual (interest_1h): a long perp position pays it, a short receives it."""
    inst = f"{cur}-PERPETUAL"
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
    rows, t = [], s
    while t < e:
        t2 = min(t + timedelta(days=30), e)
        r = call(LIVE, "get_funding_rate_history", instrument_name=inst, start_timestamp=ms(t), end_timestamp=ms(t2))
        rows += [{"ts": x["timestamp"], "interest_1h": x.get("interest_1h"), "interest_8h": x.get("interest_8h"),
                  "index": x.get("index_price")} for x in r]
        t = t2
    df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")
    out = RAW / "funding" / f"{inst}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{inst}: {len(df)} hourly funding records -> {out}")


# ---------------------------------------------------------------- settlement prices
def fetch_delivery(cur):
    idx = f"{cur.lower()}_usd"
    rows, off = [], 0
    while True:
        r = call(LIVE, "get_delivery_prices", index_name=idx, offset=off, count=100)
        rows += r["data"]
        if len(r["data"]) < 100:
            break
        off += 100
    df = pd.DataFrame(rows).rename(columns={"delivery_price": "price"}).drop_duplicates("date").sort_values("date")
    out = RAW / "delivery" / f"{idx}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{idx}: {len(df)} settlement prices {df.date.iloc[0]} .. {df.date.iloc[-1]} -> {out}")


def run_parallel(fn, cur, start, end, workers):
    log = RAW / "logs" / f"{fn.__name__}_{cur}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(workers) as ex, open(log, "a") as fh:
        futs = {ex.submit(fn, cur, d): d for d in days(start, end)}
        for f in concurrent.futures.as_completed(futs):
            try:
                msg = f.result()
            except Exception as e:
                msg = f"{cur} {futs[f]:%Y-%m-%d} ERROR {e}"
            fh.write(msg + "\n")
            fh.flush()
            print(msg, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["options", "perp1m", "perptrades", "delivery", "funding"])
    p.add_argument("--currency", required=True, choices=["BTC", "ETH"])
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.what == "options":
        run_parallel(fetch_option_day, a.currency, a.start, a.end, a.workers)
    elif a.what == "perptrades":
        run_parallel(fetch_perp_trades_day, a.currency, a.start, a.end, a.workers)
    elif a.what == "perp1m":
        fetch_perp_1m(a.currency, a.start, a.end)
    elif a.what == "funding":
        fetch_funding(a.currency, a.start, a.end)
    else:
        fetch_delivery(a.currency)
