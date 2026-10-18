#!/usr/bin/env python3
"""
CryptoOption V2 paper trader: the strategies selected in the V2 research, run forward in time on LIVE Deribit
public market data. No account, no API key, no real orders: every fill is simulated at the live best bid/ask at
the decision time, with Deribit fees, the perpetual's real hourly funding and Deribit's official 08:00 UTC
delivery price for option settlement. All state is in SQLite, so a restart resumes without trading twice.

Two processes:
  run   the trading loop. Every cycle it also writes a read-only snapshot for the dashboard (public/state.json)
        and, every 5 minutes, CSV exports (public/*.csv). It opens no network port.
  web   a small read-only web server for the dashboard. It serves the page below plus the files in public/ and
        nothing else, so it can run as an unprivileged user and be exposed to the internet.

Python standard library only.  Usage:  python3 papertrader.py run | web | export | selftest
"""
import gzip
import json
import math
import os
import sqlite3
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API = "https://www.deribit.com/api/v2/public/"
DB = os.environ.get("PAPER_DB", "/opt/cryptopaper/paper.db")
PUBLIC = os.environ.get("PAPER_PUBLIC", os.path.join(os.path.dirname(DB) or ".", "public"))
PORT = int(os.environ.get("PAPER_PORT", "18090"))
HOST = os.environ.get("PAPER_HOST", "127.0.0.1")
GITHUB = os.environ.get("PAPER_GITHUB", "https://github.com/3099404236/CryptoOption-Research")
UTC = timezone.utc
START_EQUITY = 1_000_000.0
OPTION_FEE, DELIVERY_FEE, PERP_FEE, SPOT_FEE, FEE_CAP = 0.0003, 0.00015, 0.0005, 0.0010, 0.125
LOT = {"BTC": (0.1, 0.001), "ETH": (1.0, 0.01)}          # option lot, perpetual lot (coins)
CAP = {"BTC": 25.0, "ETH": 400.0}                        # max coins per option leg (as in the backtests)
UNIT = 1.0                                               # option notional per leg = 100 % of equity ("4x" runs)
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}

ACCOUNTS = [
    ("BTC_TSMOM28", "BTC 趋势跟踪：28 天动量，只做多/空仓，波动率目标 40%（日线，每天 00:05 调仓）"),
    ("ETH_TSMOM112", "ETH 趋势跟踪：112 天动量，只做多/空仓，波动率目标 40%（日线，每天 00:05 调仓）"),
    ("BTC_CARRY", "BTC 资金费率套利：买现货（按指数价）+ 空永续，收资金费"),
    ("ETH_CARRY", "ETH 资金费率套利：买现货（按指数价）+ 空永续，收资金费"),
    ("BTC_STRADDLE_DH", "BTC 卖平值跨式：周五 09:00 卖下周五到期，每天 08:00 用永续对冲 Delta"),
    ("BTC_PUT30_HAR", "BTC 卖 0.30 Delta 看跌：周一 09:30，仅当平值 IV ≥ 1.1 × HAR 预测波动率"),
    ("ETH_STRANGLE25_HAR", "ETH 卖 0.25 Delta 宽跨：周一 09:30，仅当平值 IV ≥ 1.1 × HAR 预测波动率"),
    ("BTC_HOLD", "对照：BTC 买入持有"),
    ("ETH_HOLD", "对照：ETH 买入持有"),
]
COMBO = ["BTC_STRADDLE_DH", "BTC_PUT30_HAR", "ETH_STRANGLE25_HAR", "BTC_CARRY"]


# ------------------------------------------------------------------------------------------ utilities
def now_utc():
    return datetime.now(UTC)


def iso(t):
    return t.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def api(method, **params):
    url = API + method + ("?" + urllib.parse.urlencode(params) if params else "")
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cryptopaper/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())["result"]
        except Exception as e:                       # network hiccup / rate limit: back off and retry
            last = e
            time.sleep(1 + 2 * attempt)
    raise RuntimeError(f"{method} {params} failed: {last}")


def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def b76_delta(F, K, T, sigma, call):
    if T <= 0 or sigma <= 0:
        return (1.0 if F > K else 0.0) if call else (-1.0 if F < K else 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    return ncdf(d1) - (0.0 if call else 1.0)


def parse_option(name):
    """BTC-24OCT26-80000-C -> (expiry datetime 08:00 UTC, strike, is_call)"""
    p = name.split("-")
    if len(p) != 4 or p[3] not in ("C", "P"):
        return None
    d = p[1]
    try:
        day, mon, yy = int(d[:-5]), MONTHS[d[-5:-2]], int(d[-2:])
        return datetime(2000 + yy, mon, day, 8, tzinfo=UTC), float(p[2]), p[3] == "C"
    except (ValueError, KeyError):
        return None


def floor_lot(q, lot):
    return math.floor(q / lot + 1e-9) * lot


# ------------------------------------------------------------------------------------------ storage
class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS accounts(name TEXT PRIMARY KEY, label TEXT, cash REAL, created TEXT);
            CREATE TABLE IF NOT EXISTS positions(account TEXT, instrument TEXT, qty REAL, avg REAL, PRIMARY KEY(account, instrument));
            CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, account TEXT, instrument TEXT,
                                              qty REAL, price REAL, fee REAL, tag TEXT);
            CREATE TABLE IF NOT EXISTS equity(ts TEXT, account TEXT, equity REAL, PRIMARY KEY(ts, account));
            CREATE TABLE IF NOT EXISTS events(key TEXT PRIMARY KEY, ts TEXT, info TEXT);
            CREATE TABLE IF NOT EXISTS funding(ts TEXT, account TEXT, instrument TEXT, rate REAL, amount REAL);
            CREATE TABLE IF NOT EXISTS rv(coin TEXT, day TEXT, rv REAL, PRIMARY KEY(coin, day));
            CREATE TABLE IF NOT EXISTS log(ts TEXT, msg TEXT);
            """)
            if "quote" not in [r[1] for r in self.db.execute("PRAGMA table_info(trades)")]:
                self.db.execute("ALTER TABLE trades ADD COLUMN quote TEXT")     # the live quote each fill used
            for name, label in ACCOUNTS:
                self.db.execute("INSERT OR IGNORE INTO accounts VALUES (?,?,?,?)", (name, label, START_EQUITY, iso(now_utc())))
            self.db.commit()

    def q(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def x(self, sql, args=()):
        with self.lock:
            self.db.execute(sql, args)
            self.db.commit()

    def log(self, msg):
        print(f"{iso(now_utc())} {msg}", flush=True)
        self.x("INSERT INTO log VALUES (?,?)", (iso(now_utc()), msg[:2000]))

    def done(self, key):
        return bool(self.q("SELECT 1 FROM events WHERE key=?", (key,)))

    def mark_done(self, key, info=""):
        self.x("INSERT OR REPLACE INTO events VALUES (?,?,?)", (key, iso(now_utc()), info))


# ------------------------------------------------------------------------------------------ market data
class Market:
    def __init__(self):
        self.perp, self.options, self.options_ts = {}, {}, {}

    def refresh_perp(self):
        for c in ("BTC", "ETH"):
            t = api("ticker", instrument_name=f"{c}-PERPETUAL")
            st = t.get("stats") or {}
            self.perp[c] = dict(bid=t["best_bid_price"], ask=t["best_ask_price"], mark=t["mark_price"], index=t["index_price"],
                                bid_size=t.get("best_bid_amount"), ask_size=t.get("best_ask_amount"),
                                funding_8h=t.get("funding_8h"), change_24h=st.get("price_change"), volume_usd=st.get("volume_usd"),
                                ts=t.get("timestamp"))

    def refresh_options(self, c, max_age=240):
        if c in self.options_ts and time.time() - self.options_ts[c] < max_age:
            return
        out = {}
        for r in api("get_book_summary_by_currency", currency=c, kind="option"):
            info = parse_option(r["instrument_name"])
            if info is None:
                continue
            exp, k, call = info
            out[r["instrument_name"]] = dict(exp=exp, K=k, call=call, bid=r.get("bid_price") or 0.0, ask=r.get("ask_price") or 0.0,
                                             mark=r.get("mark_price") or 0.0, iv=(r.get("mark_iv") or 0.0) / 100.0,
                                             F=r.get("underlying_price") or self.perp[c]["index"],
                                             index=r.get("estimated_delivery_price") or self.perp[c]["index"])
        self.options[c], self.options_ts[c] = out, time.time()

    def option(self, name):
        c = name.split("-")[0]
        return self.options.get(c, {}).get(name)

    def daily_closes(self, c, days):
        end = int(now_utc().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        start = end - (days + 5) * 86_400_000
        r = api("get_tradingview_chart_data", instrument_name=f"{c}-PERPETUAL", start_timestamp=start, end_timestamp=end, resolution="1D")
        rows = sorted((t, cl) for t, cl in zip(r["ticks"], r["close"]) if t < end)
        return [cl for _, cl in rows]

    def delivery_price(self, c, day):
        r = api("get_delivery_prices", index_name=f"{c.lower()}_usd", offset=0, count=5)
        for x in r.get("data", []):
            if x["date"] == day.strftime("%Y-%m-%d"):
                return float(x["delivery_price"])
        return None

    def funding_last_hour(self, c, hour_end):
        end = int(hour_end.timestamp() * 1000)
        r = api("get_funding_rate_history", instrument_name=f"{c}-PERPETUAL", start_timestamp=end - 3_600_000, end_timestamp=end)
        rows = [x for x in r if x["timestamp"] == end] or r[-1:]
        return (float(rows[-1]["interest_1h"]), float(rows[-1]["index_price"])) if rows else (None, None)


# ------------------------------------------------------------------------------------------ HAR forecast
def har_forecast(store, market, c):
    """Direct HAR forecast (annualised vol) of the next 7 days' mean daily RV; rolling 365-day OLS on daily RV from
    5-minute perpetual returns (same definition as the LEAN backtests)."""
    today = now_utc().replace(hour=0, minute=0, second=0, microsecond=0)
    have = {d for (d,) in store.q("SELECT day FROM rv WHERE coin=?", (c,))}
    need = [today - timedelta(days=i) for i in range(1, 420)]
    missing = [d for d in need if d.strftime("%Y-%m-%d") not in have]
    if missing:
        lo, hi = min(missing), max(missing) + timedelta(days=1)
        t = lo
        while t < hi:
            u = min(t + timedelta(days=10), hi)
            r = api("get_tradingview_chart_data", instrument_name=f"{c}-PERPETUAL", start_timestamp=int(t.timestamp() * 1000),
                    end_timestamp=int(u.timestamp() * 1000), resolution="5")
            pts = sorted(zip(r["ticks"], r["close"]))
            acc = {}
            for (t0, c0), (t1, c1) in zip(pts, pts[1:]):
                if t1 - t0 == 300_000 and c0 > 0 and c1 > 0:
                    day = datetime.fromtimestamp(t1 / 1000 - 1, UTC).strftime("%Y-%m-%d")
                    acc[day] = acc.get(day, 0.0) + math.log(c1 / c0) ** 2
            for day, v in acc.items():
                if datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC) < today:
                    store.x("INSERT OR REPLACE INTO rv VALUES (?,?,?)", (c, day, v))
            t = u
    rv = [v for _, v in store.q("SELECT day, rv FROM rv WHERE coin=? AND day < ? ORDER BY day", (c, today.strftime("%Y-%m-%d")))]
    if len(rv) < 30 + 7 + 120:
        return None
    X, y = [], []
    for t in range(29, len(rv) - 7):
        X.append([1.0, rv[t], sum(rv[t - 6:t + 1]) / 7, sum(rv[t - 29:t + 1]) / 30])
        y.append(sum(rv[t + 1:t + 8]) / 7)
    X, y = X[-365:], y[-365:]
    xtx = [[sum(a[i] * a[j] for a in X) for j in range(4)] for i in range(4)]
    xty = [sum(a[i] * b for a, b in zip(X, y)) for i in range(4)]
    beta = solve(xtx, xty)
    t = len(rv) - 1
    f = sum(b * v for b, v in zip(beta, [1.0, rv[t], sum(rv[t - 6:t + 1]) / 7, sum(rv[t - 29:t + 1]) / 30]))
    m = sum(rv[t - 29:t + 1]) / 30
    return math.sqrt(365.0 * min(max(f, 0.2 * m), 5.0 * m))


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        if abs(M[i][i]) < 1e-30:
            return [0.0] * n
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [a - f * c for a, c in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]


# ------------------------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, store, market):
        self.s, self.m = store, market

    # ---- accounting
    def cash(self, acct):
        return self.s.q("SELECT cash FROM accounts WHERE name=?", (acct,))[0][0]

    def positions(self, acct):
        return {i: (q, a) for i, q, a in self.s.q("SELECT instrument, qty, avg FROM positions WHERE account=?", (acct,)) if abs(q) > 1e-12}

    def price(self, inst):
        """mark-to-market price in USD"""
        c = inst.split("-")[0]
        if inst.endswith("-PERPETUAL"):
            return self.m.perp[c]["mark"]
        if inst.endswith("-SPOT"):
            return self.m.perp[c]["index"]
        o = self.m.option(inst)
        if o is None:
            self.m.refresh_options(c, max_age=60)
            o = self.m.option(inst)
        if o is None:                                  # expired, waiting for the delivery price: intrinsic at the index
            info = parse_option(inst)
            if info is None:
                return 0.0
            _, k, call = info
            S = self.m.perp[c]["index"]
            return max(S - k, 0.0) if call else max(k - S, 0.0)
        return o["mark"] * o["index"]

    def equity(self, acct):
        return self.cash(acct) + sum(q * self.price(i) for i, (q, _) in self.positions(acct).items())

    def book(self, acct, inst, qty, price, fee, tag, quote=""):
        with self.s.lock:
            db = self.s.db
            db.execute("UPDATE accounts SET cash = cash - ? - ? WHERE name=?", (qty * price, fee, acct))
            row = db.execute("SELECT qty, avg FROM positions WHERE account=? AND instrument=?", (acct, inst)).fetchone()
            q0, a0 = row if row else (0.0, 0.0)
            q1 = q0 + qty
            if abs(q1) < 1e-12:
                a1 = 0.0                                   # flat
            elif abs(q0) < 1e-12 or (q0 > 0) != (q1 > 0):
                a1 = price                                 # new position or flipped side
            elif (q0 > 0) == (qty > 0):
                a1 = (a0 * q0 + price * qty) / q1          # added to the position
            else:
                a1 = a0                                    # reduced: entry price unchanged
            db.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?)", (acct, inst, q1, a1))
            db.execute("INSERT INTO trades(ts, account, instrument, qty, price, fee, tag, quote) VALUES (?,?,?,?,?,?,?,?)",
                       (iso(now_utc()), acct, inst, qty, price, fee, tag, quote))
            db.commit()
        self.s.log(f"[{acct}] {'BUY' if qty > 0 else 'SELL'} {abs(qty):g} {inst} @ {price:,.2f} fee {fee:,.2f} ({tag})")

    # ---- fills at the live best bid/ask
    def trade_perp(self, acct, c, qty, tag):
        qty = round(qty / LOT[c][1]) * LOT[c][1]
        if abs(qty) < LOT[c][1] / 2:
            return
        bid, ask = self.m.perp[c]["bid"], self.m.perp[c]["ask"]
        p = ask if qty > 0 else bid
        self.book(acct, f"{c}-PERPETUAL", qty, p, PERP_FEE * abs(qty) * p, tag, f"买一 {bid:,.2f} / 卖一 {ask:,.2f}")

    def trade_spot(self, acct, c, qty, tag):
        p = self.m.perp[c]["index"]
        self.book(acct, f"{c}-SPOT", qty, p, SPOT_FEE * abs(qty) * p, tag, f"指数价 {p:,.2f}")

    def trade_option(self, acct, name, qty, tag):
        o = self.m.option(name)
        px_coin = o["ask"] if qty > 0 else o["bid"]
        if not px_coin or px_coin <= 0:
            raise RuntimeError(f"no {'ask' if qty > 0 else 'bid'} for {name}")
        p = px_coin * o["index"]
        fee = min(OPTION_FEE * o["index"], FEE_CAP * p) * abs(qty)
        c = name.split("-")[0]
        self.book(acct, name, qty, p, fee, tag,
                  f"买一 {o['bid']:g} / 卖一 {o['ask']:g} {c}，× 指数 {o['index']:,.2f}，IV {o['iv']:.1%}")

    def settle_options(self, acct, c, expiry):
        D = self.m.delivery_price(c, expiry)
        if D is None:
            return False
        for inst, (q, _) in self.positions(acct).items():
            info = parse_option(inst)
            if info and info[0] == expiry:
                _, k, call = info
                intrinsic = max(D - k, 0.0) if call else max(k - D, 0.0)
                fee = min(DELIVERY_FEE * D, FEE_CAP * intrinsic) * abs(q) if intrinsic > 0 else 0.0
                self.book(acct, inst, -q, intrinsic, fee, f"Deribit 结算价 {D:,.2f}", f"官方结算价 {D:,.2f}")
        return True

    def option_delta(self, acct, t):
        total = 0.0
        for inst, (q, _) in self.positions(acct).items():
            info = parse_option(inst)
            o = self.m.option(inst)
            if info and o and o["iv"] > 0:
                T = (info[0] - t).total_seconds() / (365 * 86400)
                total += q * b76_delta(o["F"], info[1], T, o["iv"], info[2])
        return total

    def funding(self, hour_end):
        for acct, _ in ACCOUNTS:
            for inst, (q, _) in self.positions(acct).items():
                if inst.endswith("-PERPETUAL"):
                    c = inst.split("-")[0]
                    rate, idx = self.m.funding_last_hour(c, hour_end)
                    if rate is None:
                        continue
                    amount = -q * idx * rate                      # longs pay positive funding, shorts receive it
                    self.s.x("UPDATE accounts SET cash = cash + ? WHERE name=?", (amount, acct))
                    self.s.x("INSERT INTO funding VALUES (?,?,?,?,?)", (iso(hour_end), acct, inst, rate, amount))

    def snapshot(self, t):
        for acct, _ in ACCOUNTS:
            self.s.x("INSERT OR REPLACE INTO equity VALUES (?,?,?)", (iso(t), acct, self.equity(acct)))


# ------------------------------------------------------------------------------------------ strategies
def weekly_expiry(market, c, target):
    """the listed Friday expiry closest to target (datetime)"""
    exps = sorted({o["exp"] for o in market.options.get(c, {}).values() if o["exp"].weekday() == 4})
    good = [e for e in exps if abs((e - target).total_seconds()) < 2 * 86400]
    return min(good, key=lambda e: abs((e - target).total_seconds())) if good else None


def atm_pair(market, c, exp):
    rows = [(n, o) for n, o in market.options[c].items() if o["exp"] == exp and o["bid"] > 0]
    F = next((o["F"] for _, o in rows), None)
    strikes = {}
    for n, o in rows:
        strikes.setdefault(o["K"], {})[o["call"]] = n
    pairs = [k for k, v in strikes.items() if len(v) == 2]
    if not pairs or F is None:
        return None
    k = min(pairs, key=lambda k: abs(k - F))
    return (strikes[k][True], strikes[k][False]) if abs(k / F - 1) < 0.03 else None


def by_delta(market, c, exp, call, target, t):
    best = None
    for n, o in market.options[c].items():
        if o["exp"] != exp or o["call"] != call or o["bid"] <= 0 or o["iv"] <= 0:
            continue
        T = (exp - t).total_seconds() / (365 * 86400)
        d = abs(b76_delta(o["F"], o["K"], T, o["iv"], call))
        if 0.02 < d < 0.6 and (best is None or abs(d - target) < best[0]):
            best = (abs(d - target), n)
    return best[1] if best and best[0] <= 0.06 else None


class Trader:
    def __init__(self, store, market):
        self.s, self.m, self.e = store, market, Engine(store, market)

    def size(self, acct, c):
        idx = self.m.perp[c]["index"]
        return floor_lot(min(self.e.equity(acct) * UNIT / idx, CAP[c]), LOT[c][0])

    # ---- trend following (daily)
    def trend(self, acct, c, lookback, t):
        key = f"{acct}:rebalance:{t:%Y-%m-%d}"
        if self.s.done(key) or not (t.hour > 0 or t.minute >= 5) and self.s.done(f"{acct}:started"):
            return
        closes = self.m.daily_closes(c, lookback + 40)
        if len(closes) < lookback + 31:
            return
        mom = closes[-1] / closes[-1 - lookback] - 1
        signal = 1.0 if mom > 0 else 0.0
        rets = [math.log(b / a) for a, b in zip(closes[-31:-1], closes[-30:])]
        mu = sum(rets) / len(rets)
        vol = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) * math.sqrt(365)
        scale = min(0.40 / vol, 2.0) if vol > 0 else 0.0
        price = self.m.perp[c]["mark"]
        target = signal * scale * self.e.equity(acct) / price
        cur = self.e.positions(acct).get(f"{c}-PERPETUAL", (0.0, 0.0))[0]
        if abs(target - cur) * price > 100:
            self.e.trade_perp(acct, c, target - cur, f"调仓：{lookback} 天涨跌 {mom:+.1%}，{'看多' if signal else '空仓'}，"
                                                     f"波动率 {vol:.0%}，仓位系数 {scale * signal:.2f}")
        self.s.mark_done(key, f"signal={signal} vol={vol:.3f} scale={scale:.2f} target={target:.4f} mom={mom:.4f}")
        self.s.mark_done(f"{acct}:started")

    # ---- funding carry and benchmarks (enter once)
    def carry(self, acct, c):
        if self.s.done(f"{acct}:entered"):
            return
        idx = self.m.perp[c]["index"]
        q = round(START_EQUITY / 1.2 / idx / LOT[c][1]) * LOT[c][1]
        self.e.trade_spot(acct, c, q, "建仓：买现货")
        self.e.trade_perp(acct, c, -q, "建仓：空永续")
        self.s.mark_done(f"{acct}:entered")

    def hold(self, acct, c):
        if self.s.done(f"{acct}:entered"):
            return
        idx = self.m.perp[c]["index"]
        self.e.trade_spot(acct, c, START_EQUITY * 0.998 / idx, "建仓：买入持有")
        self.s.mark_done(f"{acct}:entered")

    # ---- option sellers
    def open_options(self, acct):
        return {i: q for i, (q, _) in self.e.positions(acct).items() if parse_option(i)}

    def straddle(self, t):
        acct, c = "BTC_STRADDLE_DH", "BTC"
        self.settle_and_unhedge(acct, c, t, hedged=True)
        if t.weekday() == 4 and 9 <= t.hour < 12 and not self.s.done(f"{acct}:entry:{t:%Y-%m-%d}") and not self.open_options(acct):
            self.m.refresh_options(c, max_age=60)
            exp = weekly_expiry(self.m, c, (t + timedelta(days=7)).replace(hour=8, minute=0, second=0, microsecond=0))
            pair = atm_pair(self.m, c, exp) if exp else None
            q = self.size(acct, c)
            if pair and q > 0:
                for n in pair:
                    self.e.trade_option(acct, n, -q, "开仓：卖平值跨式")
                self.s.mark_done(f"{acct}:entry:{t:%Y-%m-%d}", f"{pair} q={q}")
                self.s.mark_done(f"{acct}:hedge_due", iso(t))
            elif t.hour == 11 and t.minute >= 30:
                self.s.mark_done(f"{acct}:entry:{t:%Y-%m-%d}", "跳过：没有带买价的平值期权")
        # daily 08:00 hedge (and once right after entry)
        if self.open_options(acct):
            due = self.s.q("SELECT ts FROM events WHERE key=?", (f"{acct}:hedge_due",))
            key = f"{acct}:hedge:{t:%Y-%m-%d}"
            if due or (t.hour == 8 and not self.s.done(key) and not self._expiring_today(acct, t)):
                self.m.refresh_options(c, max_age=60)
                target = -self.e.option_delta(acct, t)
                cur = self.e.positions(acct).get(f"{c}-PERPETUAL", (0.0, 0.0))[0]
                self.e.trade_perp(acct, c, target - cur, f"Delta 对冲：期权 Delta {-target:+.3f}")
                self.s.x("DELETE FROM events WHERE key=?", (f"{acct}:hedge_due",))
                if t.hour == 8:
                    self.s.mark_done(key)

    def _expiring_today(self, acct, t):
        return any(parse_option(i)[0].date() == t.date() for i in self.open_options(acct))

    def settle_and_unhedge(self, acct, c, t, hedged):
        for inst in list(self.open_options(acct)):
            exp = parse_option(inst)[0]
            if t >= exp + timedelta(seconds=30) and not self.s.done(f"{acct}:settled:{exp:%Y-%m-%d}"):
                if self.e.settle_options(acct, c, exp):
                    self.s.mark_done(f"{acct}:settled:{exp:%Y-%m-%d}")
        if hedged and not self.open_options(acct):
            cur = self.e.positions(acct).get(f"{c}-PERPETUAL", (0.0, 0.0))[0]
            if abs(cur) > 0:
                self.e.trade_perp(acct, c, -cur, "到期结算后平掉对冲")

    def monday_seller(self, acct, c, legs, t):
        self.settle_and_unhedge(acct, c, t, hedged=False)
        key = f"{acct}:entry:{t:%Y-%m-%d}"
        if not (t.weekday() == 0 and (t.hour, t.minute) >= (9, 30) and t.hour < 12) or self.s.done(key) or self.open_options(acct):
            return
        self.m.refresh_options(c, max_age=60)
        exp = weekly_expiry(self.m, c, (t + timedelta(days=4)).replace(hour=8, minute=0, second=0, microsecond=0))
        pair = atm_pair(self.m, c, exp) if exp else None
        if not pair:
            if t.hour == 11 and t.minute >= 30:
                self.s.mark_done(key, "跳过：没有平值期权")
            return
        atm_iv = sum(self.m.option(n)["iv"] for n in pair) / 2
        fv = har_forecast(self.s, self.m, c)
        if fv is None or atm_iv < 1.10 * fv:
            har_txt = "无" if fv is None else f"{fv:.1%}"
            self.s.mark_done(key, f"本周不交易：平值 IV {atm_iv:.1%} < 1.1 × HAR 预测 {har_txt}")
            self.s.log(f"[{acct}] 本周不交易：平值 IV {atm_iv:.1%}，HAR 预测 {har_txt}")
            return
        names = [by_delta(self.m, c, exp, call, d, t) for call, d in legs]
        q = self.size(acct, c)
        if None in names or q <= 0:
            if t.hour == 11 and t.minute >= 30:
                self.s.mark_done(key, "跳过：找不到合适的行权价")
            return
        for n in names:
            self.e.trade_option(acct, n, -q, f"开仓：平值 IV {atm_iv:.1%} ≥ 1.1 × HAR {fv:.1%}")
        self.s.mark_done(key, f"开仓：平值 IV {atm_iv:.1%} ≥ 1.1 × HAR 预测 {fv:.1%}，卖出 {', '.join(names)} 各 {q:g}")

    # ---- one cycle
    def cycle(self, t):
        self.m.refresh_perp()
        for fn in (lambda: self.trend("BTC_TSMOM28", "BTC", 28, t), lambda: self.trend("ETH_TSMOM112", "ETH", 112, t),
                   lambda: self.carry("BTC_CARRY", "BTC"), lambda: self.carry("ETH_CARRY", "ETH"),
                   lambda: self.hold("BTC_HOLD", "BTC"), lambda: self.hold("ETH_HOLD", "ETH"),
                   lambda: self.straddle(t),
                   lambda: self.monday_seller("BTC_PUT30_HAR", "BTC", [(False, 0.30)], t),
                   lambda: self.monday_seller("ETH_STRANGLE25_HAR", "ETH", [(True, 0.25), (False, 0.25)], t)):
            try:
                fn()
            except Exception as ex:
                self.s.log(f"strategy error: {ex!r}\n{traceback.format_exc()[-1500:]}")
        hour_end = t.replace(minute=0, second=0, microsecond=0)
        if t.minute >= 1 and not self.s.done(f"funding:{iso(hour_end)}"):
            self.e.funding(hour_end)
            self.s.mark_done(f"funding:{iso(hour_end)}")
        for c in ("BTC", "ETH"):                        # option marks for the equity snapshot
            if any(parse_option(i) and i.startswith(c) for a, _ in ACCOUNTS for i in self.e.positions(a)):
                self.m.refresh_options(c, max_age=240)
        slot = t.replace(minute=t.minute - t.minute % 5, second=0, microsecond=0)
        if not self.s.done(f"snap:{iso(slot)}"):
            self.e.snapshot(slot)
            self.s.mark_done(f"snap:{iso(slot)}")


# ------------------------------------------------------------------------------------------ dashboard snapshot
CSVS = [
    ("equity.csv", "SELECT ts, account, equity FROM equity ORDER BY ts, account", ["ts_utc", "account", "equity_usd"]),
    ("trades.csv", "SELECT ts, account, instrument, qty, price, fee, tag, quote FROM trades ORDER BY id",
     ["ts_utc", "account", "instrument", "qty", "price_usd", "fee_usd", "note", "live_quote_used"]),
    ("positions.csv", "SELECT account, instrument, qty, avg FROM positions WHERE abs(qty) > 1e-12", ["account", "instrument", "qty", "avg_price_usd"]),
    ("funding.csv", "SELECT ts, account, instrument, rate, amount FROM funding ORDER BY ts",
     ["hour_end_utc", "account", "instrument", "interest_1h", "amount_usd"]),
]


def csv_of(store, sql, header):
    out = [",".join(header)]
    for r in store.q(sql):
        out.append(",".join('"' + x.replace('"', '""') + '"' if isinstance(x, str) else "" if x is None else f"{x}" for x in r))
    return "\n".join(out) + "\n"


def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def clean(x):
    """JSON-safe: NaN / inf -> None, floats rounded"""
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else round(x, 6)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def next_at(t, weekday=None, hour=0, minute=0):
    """first time after t at hh:mm UTC (on the given weekday, Monday = 0, if set)"""
    c = t.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if weekday is not None:
        c += timedelta(days=(weekday - t.weekday()) % 7)
    while c <= t:
        c += timedelta(days=7 if weekday is not None else 1)
    return c


def describe(inst):
    c = inst.split("-")[0]
    if inst.endswith("-PERPETUAL"):
        return f"{c} 永续合约"
    if inst.endswith("-SPOT"):
        return f"{c} 现货（按指数价模拟）"
    info = parse_option(inst)
    if info:
        exp, k, call = info
        return f"{c} {'看涨' if call else '看跌'}期权，行权价 {k:,.0f}，{exp:%m-%d} 08:00 UTC 到期"
    return inst


class Publisher:
    """Builds public/state.json for the dashboard after every trading cycle (and the CSV exports every 5 minutes).
    Runs in the trading thread; it only reads the database and the market data already fetched for trading, plus
    a few cached extra calls (option chains every 5 minutes, HAR forecast every hour, daily closes once a day)."""

    def __init__(self, trader):
        self.tr, self.s, self.m, self.e = trader, trader.s, trader.m, trader.e
        self.har, self.mom, self.eq_cache, self.csv_slot, self.last_err = {}, {}, (None, None), None, None
        os.makedirs(PUBLIC, exist_ok=True)

    def publish(self, t):
        try:
            state = self.build(t)
            write_atomic(os.path.join(PUBLIC, "state.json"),
                         json.dumps(clean(state), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            slot = t.replace(minute=t.minute - t.minute % 5, second=0, microsecond=0)
            if slot != self.csv_slot:
                for fn, sql, header in CSVS:
                    write_atomic(os.path.join(PUBLIC, fn), csv_of(self.s, sql, header).encode("utf-8-sig"))
                self.csv_slot = slot
            self.last_err = None
        except Exception as ex:                       # never let the dashboard disturb trading; log each new error once
            msg = repr(ex)
            if msg != self.last_err:
                self.s.log(f"dashboard error: {msg}\n{traceback.format_exc()[-1200:]}")
                self.last_err = msg

    # ---- cached extras
    def har_cached(self, c, t):
        key = t.strftime("%Y-%m-%d %H")
        if self.har.get(c, (None, None))[0] != key:
            try:
                self.har[c] = (key, har_forecast(self.s, self.m, c))
            except Exception:
                self.har[c] = (key, self.har.get(c, (None, None))[1])
        return self.har[c][1]

    def momentum(self, c, L, t):
        key = (c, L, t.strftime("%Y-%m-%d"))
        if key not in self.mom:
            closes = self.m.daily_closes(c, L + 5)
            self.mom = {k: v for k, v in self.mom.items() if k[2] == key[2]}
            self.mom[key] = closes[-1] / closes[-1 - L] - 1 if len(closes) > L else None
        return self.mom[key]

    def equity_series(self):
        last = self.s.q("SELECT MAX(ts), COUNT(*) FROM equity")[0]
        if self.eq_cache[0] == last:
            return self.eq_cache[1]
        rows = self.s.q("SELECT ts, account, equity FROM equity ORDER BY ts")
        times = sorted({ts for ts, _, _ in rows})
        ix = {ts: i for i, ts in enumerate(times)}
        full = {a: [None] * len(times) for a, _ in ACCOUNTS}
        for ts, a, v in rows:
            if a in full:
                full[a][ix[ts]] = v
        n = len(times)
        step = max(1, math.ceil(n / 1500))
        keep = list(range(0, n, step))
        if keep and keep[-1] != n - 1:
            keep.append(n - 1)
        out = dict(full=full, times=times,
                   t=[times[i][:16] for i in keep],
                   s={a: [None if full[a][i] is None else round(full[a][i], 2) for i in keep] for a in full})
        self.eq_cache = (last, out)
        return out

    # ---- market panel
    def market_info(self, c, t):
        p = dict(self.m.perp[c])
        mid = (p["bid"] + p["ask"]) / 2
        p["spread_bp"] = (p["ask"] - p["bid"]) / mid * 1e4 if mid else None
        p["funding_ann"] = p["funding_8h"] * 3 * 365 if p.get("funding_8h") is not None else None
        p["har"] = self.har_cached(c, t)
        p["quotes"], p["atm_iv"], p["exp"] = [], None, None
        try:
            self.m.refresh_options(c, max_age=300)
        except Exception:
            pass
        chain = self.m.options.get(c, {})
        p["n_options"] = len(chain)
        exps = sorted({o["exp"] for o in chain.values() if o["exp"].weekday() == 4 and o["exp"] - t > timedelta(days=1)})
        if not exps:
            return p
        exp = exps[0]
        p["exp"] = iso(exp)
        pair = atm_pair(self.m, c, exp)
        rows = []
        if pair:
            p["atm_iv"] = sum(self.m.option(n)["iv"] for n in pair) / 2
            rows += [(pair[0], "平值看涨"), (pair[1], "平值看跌")]
        legs = [(False, 0.30, "0.30 Delta 看跌（BTC 卖看跌策略用）")] if c == "BTC" else \
               [(True, 0.25, "0.25 Delta 看涨（ETH 宽跨策略用）"), (False, 0.25, "0.25 Delta 看跌（ETH 宽跨策略用）")]
        for call, d, label in legs:
            n = by_delta(self.m, c, exp, call, d, t)
            if n:
                rows.append((n, label))
        T = (exp - t).total_seconds() / (365 * 86400)
        for n, label in rows:
            o = self.m.option(n)
            rows_d = abs(b76_delta(o["F"], o["K"], T, o["iv"], o["call"])) if o["iv"] > 0 else None
            p["quotes"].append(dict(name=n, label=label, K=o["K"], bid=o["bid"], ask=o["ask"], index=o["index"], iv=o["iv"],
                                    delta=rows_d, bid_usd=o["bid"] * o["index"], ask_usd=o["ask"] * o["index"]))
        p["quotes_ts"] = iso(datetime.fromtimestamp(self.m.options_ts.get(c, time.time()), UTC))
        return p

    # ---- accounts
    def status(self, acct, t, pos, eq):
        """(what the account holds now, (next action time, what it will do))"""
        c = acct[:3]
        if "TSMOM" in acct:
            L = 28 if acct == "BTC_TSMOM28" else 112
            q = pos.get(f"{c}-PERPETUAL", (0.0, 0.0))[0]
            ev = self.s.q("SELECT ts, info FROM events WHERE key LIKE ? ORDER BY ts DESC LIMIT 1", (f"{acct}:rebalance:%",))
            info = dict(kv.split("=", 1) for kv in ev[0][1].split()) if ev else {}
            try:
                mom = float(info["mom"]) if "mom" in info else self.momentum(c, L, t)
            except Exception:
                mom = None
            mom_txt = "—" if mom is None else f"{mom:+.1%}"
            last = f"最近一次调仓 {ev[0][0][5:16]} UTC：过去 {L} 天涨跌 {mom_txt}，30 天年化波动率 {float(info.get('vol', 'nan')):.0%}。" if ev else ""
            if q > 0:
                now = f"持有 {c} 永续多头 {q:,.3f} {c}（约 {q * self.m.perp[c]['mark'] / eq:.2f} 倍净值）。"
            else:
                now = "空仓（动量不为正时不持有）。"
            return now + last, (next_at(t, None, 0, 5), "每日调仓：按前一天收盘价重新计算动量和波动率，调整仓位")
        if "CARRY" in acct:
            spot = pos.get(f"{c}-SPOT", (0.0, 0.0))[0]
            perp = pos.get(f"{c}-PERPETUAL", (0.0, 0.0))[0]
            fs, fn = self.s.q("SELECT COALESCE(SUM(amount), 0), COUNT(*) FROM funding WHERE account=?", (acct,))[0]
            now = (f"持有现货 {spot:+,.3f} {c}、永续 {perp:+,.3f} {c}，两边涨跌抵消。"
                   f"已记 {fn} 小时资金费，合计 {'+' if fs >= 0 else '-'}${abs(fs):,.0f}。")
            return now, (t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=1), "整点记一小时资金费")
        if "HOLD" in acct:
            spot = pos.get(f"{c}-SPOT", (0.0, 0.0))[0]
            return f"持有 {spot:,.4f} {c} 现货，一直不动（对照组）。", None
        opts = {i: q for i, (q, _) in pos.items() if parse_option(i)}
        if opts:
            exp = min(parse_option(i)[0] for i in opts)
            legs = "、".join(f"{i}（{q:+g}）" for i, q in opts.items())
            if acct == "BTC_STRADDLE_DH":
                h = pos.get("BTC-PERPETUAL", (0.0, 0.0))[0]
                od = self.e.option_delta(acct, t)
                now = f"已卖出 {legs}；永续对冲 {h:+.3f} BTC，期权 Delta {od:+.3f}，合计 {h + od:+.3f}。"
                nxt = next_at(t, None, 8, 0)
                if nxt.date() == exp.date() or nxt > exp:
                    return now, (exp, "到期：按 Deribit 官方结算价结算，然后平掉永续对冲")
                return now, (nxt, "每日 Delta 对冲：把组合 Delta 调回 0")
            ev = self.s.q("SELECT info FROM events WHERE key LIKE ? ORDER BY ts DESC LIMIT 1", (f"{acct}:entry:%",))
            why = f"（{ev[0][0].split('，卖出')[0]}）" if ev and ev[0][0].startswith("开仓") else ""
            return f"已卖出 {legs}{why}，不对冲。", (exp, "到期：按 Deribit 官方结算价结算")
        if acct == "BTC_STRADDLE_DH":
            return "空仓，等待周五开仓。", (next_at(t, 4, 9, 0), "卖出下周五到期的 BTC 平值看涨 + 看跌，并立即对冲 Delta")
        ev = self.s.q("SELECT ts, info FROM events WHERE key LIKE ? ORDER BY ts DESC LIMIT 1", (f"{acct}:entry:%",))
        monday = (t - timedelta(days=t.weekday())).replace(hour=9, minute=30, second=0, microsecond=0)
        if ev and ev[0][0] >= iso(monday) and not ev[0][1].startswith("开仓"):
            now = f"{ev[0][1]}。"
        else:
            now = "空仓，等待周一判断。"
        return now, (next_at(t, 0, 9, 30), "比较平值 IV 和 HAR 预测：IV ≥ 1.1 × HAR 才卖出当周五到期的期权")

    def build(self, t):
        eqs = self.equity_series()
        day_ago = iso(t - timedelta(days=1))
        fees = dict(self.s.q("SELECT account, SUM(fee) FROM trades GROUP BY account"))
        ntr = dict(self.s.q("SELECT account, COUNT(*) FROM trades GROUP BY account"))
        fund = {a: (s, n) for a, s, n in self.s.q("SELECT account, SUM(amount), COUNT(*) FROM funding GROUP BY account")}
        accounts = []
        for name, label in ACCOUNTS:
            pos = self.e.positions(name)
            eq = self.e.equity(name)
            ser = [v for v in eqs["full"][name] if v is not None]
            peak, mdd = START_EQUITY, 0.0
            for v in ser + [eq]:
                peak = max(peak, v)
                mdd = max(mdd, 1 - v / peak)
            prev = next((v for ts, v in zip(reversed(eqs["times"]), reversed(eqs["full"][name])) if v is not None and ts <= day_ago),
                        START_EQUITY)
            plist = []
            for inst, (q, avg) in sorted(pos.items()):
                px = self.e.price(inst)
                plist.append(dict(inst=inst, desc=describe(inst), qty=q, avg=avg, price=px, value=q * px, upnl=q * (px - avg)))
            delta = sum(q for i, (q, _) in pos.items() if not parse_option(i)) + self.e.option_delta(name, t)
            now, nxt = self.status(name, t, pos, eq)
            accounts.append(dict(name=name, label=label, equity=eq, ret=eq / START_EQUITY - 1, ret24h=eq / prev - 1, mdd=mdd,
                                 trades=ntr.get(name, 0), fees=fees.get(name) or 0.0, funding=(fund.get(name) or (0.0, 0))[0],
                                 funding_hours=(fund.get(name) or (0.0, 0))[1], delta=delta, positions=plist, now=now,
                                 next_ts=iso(nxt[0]) if nxt else None, next_what=nxt[1] if nxt else None))
        trades = [dict(id=i, ts=ts, acct=a, inst=inst, desc=describe(inst), qty=q, price=p, fee=f, tag=tag, quote=quote or "")
                  for i, ts, a, inst, q, p, f, tag, quote in
                  self.s.q("SELECT id, ts, account, instrument, qty, price, fee, tag, quote FROM trades ORDER BY id DESC LIMIT 300")]
        funding = [dict(ts=ts, acct=a, inst=i, rate=r, amount=x) for ts, a, i, r, x in
                   self.s.q("SELECT ts, account, instrument, rate, amount FROM funding ORDER BY ts DESC, account LIMIT 60")]
        log = [dict(ts=ts, msg=m.split("\n")[0][:300]) for ts, m in self.s.q("SELECT ts, msg FROM log ORDER BY rowid DESC LIMIT 40")]
        market = {}
        for c in ("BTC", "ETH"):
            try:
                market[c] = self.market_info(c, t)
            except Exception as ex:
                market[c] = dict(self.m.perp.get(c, {}), error=repr(ex))
        return dict(v=2, generated=iso(t), started=self.s.q("SELECT MIN(created) FROM accounts")[0][0], github=GITHUB,
                    start_equity=START_EQUITY, combo=COMBO, accounts=accounts, eq=dict(t=eqs["t"], s=eqs["s"]),
                    n_trades=sum(ntr.values()), trades=trades, funding=funding, log=log, market=market)


# ------------------------------------------------------------------------------------------ public web server
class PublicHandler(BaseHTTPRequestHandler):
    """Read-only: the page itself and a fixed list of files from PUBLIC. No database access, nothing else."""
    server_version, sys_version, timeout = "cryptopaper", "", 20
    FILES = {"/state.json": ("state.json", "application/json; charset=utf-8"),
             "/equity.csv": ("equity.csv", "text/csv; charset=utf-8"), "/trades.csv": ("trades.csv", "text/csv; charset=utf-8"),
             "/positions.csv": ("positions.csv", "text/csv; charset=utf-8"), "/funding.csv": ("funding.csv", "text/csv; charset=utf-8")}
    cache, lock = {}, threading.Lock()

    def do_GET(self):
        self.serve(True)

    def do_HEAD(self):
        self.serve(False)

    def load(self, path):
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        with open(path, "rb") as f:
            raw = f.read()
        val = (raw, gzip.compress(raw, 6))
        with self.lock:
            for k in [k for k in self.cache if k[0] == path]:
                del self.cache[k]
            self.cache[key] = val
        return val

    def serve(self, body):
        path = urllib.parse.urlsplit(self.path).path
        extra = {}
        if path in ("/", "/index.html"):
            raw, gz = PAGE_BYTES
            ctype, cache = "text/html; charset=utf-8", "no-cache"
        elif path in self.FILES:
            fn, ctype = self.FILES[path]
            try:
                raw, gz = self.load(os.path.join(PUBLIC, fn))
            except OSError:
                return self.send_error(503, "snapshot not ready yet")
            cache = "no-store"
            if fn.endswith(".csv"):
                extra["Content-Disposition"] = f'attachment; filename="paper_{fn}"'
        elif path == "/robots.txt":
            raw, ctype, cache = b"User-agent: *\nAllow: /\n", "text/plain", "max-age=86400"
            gz = None
        else:
            return self.send_error(404)
        use_gz = gz is not None and "gzip" in self.headers.get("Accept-Encoding", "") and len(raw) > 1024
        data = gz if use_gz else raw
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if use_gz:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(data)

    def log_message(self, *a):
        pass


def serve_public():
    srv = ThreadingHTTPServer((HOST, PORT), PublicHandler)
    srv.daemon_threads = True
    print(f"{iso(now_utc())} dashboard on http://{HOST}:{PORT} (files from {PUBLIC})", flush=True)
    srv.serve_forever()


# ------------------------------------------------------------------------------------------ the page
PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>加密货币策略模拟盘</title>
<meta name="description" content="第一阶段回测选出的加密货币策略，在 Deribit 实时行情上自动模拟交易">
<style>
:root{--bg:#f4f6f9;--card:#fff;--text:#0f172a;--muted:#64748b;--line:#e2e8f0;--soft:#f1f5f9;--accent:#2563eb;--accent-soft:#eff6ff;
--pos:#059669;--neg:#dc2626;--warn:#b45309;--warn-soft:#fffbeb;--ok-soft:#ecfdf5;--shadow:0 1px 2px rgba(15,23,42,.05),0 2px 8px rgba(15,23,42,.05);--combo:#0f172a}
@media (prefers-color-scheme:dark){:root{--bg:#0b1120;--card:#111a2e;--text:#e2e8f0;--muted:#94a3b8;--line:#1e2a44;--soft:#16213a;--accent:#60a5fa;
--accent-soft:#132245;--pos:#34d399;--neg:#f87171;--warn:#fbbf24;--warn-soft:#2a2210;--ok-soft:#0f2a22;--shadow:none;--combo:#f8fafc}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1200px;margin:0 auto;padding:22px 16px 48px}
header{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:flex-end;justify-content:space-between;margin-bottom:14px}
h1{font-size:24px;line-height:1.3;margin:0 0 4px;letter-spacing:.2px}
.sub{color:var(--muted);margin:0}
h2{font-size:18px;margin:34px 0 4px}
h2+.hint{margin:0 0 12px}
h3{font-size:15px;margin:0 0 8px}
.hint,.muted{color:var(--muted);font-size:13px}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;background:var(--ok-soft);font-size:13px;white-space:nowrap}
.pill i{width:8px;height:8px;border-radius:50%;background:var(--pos);display:inline-block;box-shadow:0 0 0 3px rgba(5,150,105,.15)}
.pill.bad{background:var(--warn-soft)}.pill.bad i{background:var(--warn)}
.links{display:flex;flex-wrap:wrap;gap:8px}
.btn{display:inline-block;padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);font-size:13px;color:var(--text)}
.btn:hover{border-color:var(--accent);text-decoration:none}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:16px}
.intro{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.intro .card b{display:block;font-size:15px;margin-bottom:4px}
.intro p{margin:0;color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:12px}
.kpi .lbl{color:var(--muted);font-size:12px}
.kpi .val{font-size:22px;font-weight:650;line-height:1.3;margin-top:2px;font-variant-numeric:tabular-nums}
.kpi .val.small{font-size:16px}
.kpi .foot{color:var(--muted);font-size:12px;margin-top:2px}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.num{font-variant-numeric:tabular-nums}
.chart-head{display:flex;flex-wrap:wrap;gap:8px;justify-content:space-between;align-items:center;margin-bottom:8px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;background:var(--card);color:var(--text);padding:5px 12px;font:inherit;font-size:13px;cursor:pointer}
.seg button.on{background:var(--accent);color:#fff}
#chart{position:relative;width:100%}
#chart svg{display:block;width:100%;height:auto;touch-action:pan-y}
.legend{display:flex;flex-wrap:wrap;gap:6px 8px;margin-top:10px}
.legend button{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:var(--card);color:var(--text);
border-radius:999px;padding:3px 10px;font:inherit;font-size:12px;cursor:pointer}
.legend button.off{opacity:.38}
.legend i{width:14px;height:3px;border-radius:2px;display:inline-block}
.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;box-shadow:0 6px 20px rgba(15,23,42,.15);
padding:8px 10px;font-size:12px;min-width:190px;display:none;z-index:5}
.tip .row{display:flex;justify-content:space-between;gap:12px}
.tip i{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px}
.accts{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.acct{border-top:4px solid var(--c)}
.acct.combo{grid-column:1/-1;background:linear-gradient(0deg,var(--card),var(--card)),var(--card)}
.acct-h{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px}
.acct-h b{font-size:16px}
code{font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--soft);padding:1px 6px;border-radius:5px;color:var(--muted)}
.chip{font-size:12px;padding:1px 8px;border-radius:999px;background:var(--accent-soft);color:var(--accent);white-space:nowrap}
.chip.gray{background:var(--soft);color:var(--muted)}
.one{margin:6px 0 10px;color:var(--muted)}
.nums{display:flex;flex-wrap:wrap;gap:6px 22px;align-items:flex-end}
.nums .lbl{font-size:12px;color:var(--muted)}
.nums .big{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.25}
.nums .mid{font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}
.state{margin-top:10px;padding:8px 10px;border-radius:8px;background:var(--soft);font-size:13px}
.state .k{color:var(--muted);margin-right:4px}
.state div+div{margin-top:4px}
details{margin-top:10px}
summary{cursor:pointer;color:var(--accent);font-size:13px;user-select:none}
details ul{margin:8px 0;padding-left:20px}
details li{margin:3px 0}
.why{margin:8px 0;padding:8px 10px;border-left:3px solid var(--c);background:var(--soft);border-radius:0 8px 8px 0;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th{color:var(--muted);font-weight:600;background:var(--soft);white-space:nowrap}
td.l,th.l{text-align:left}
tbody tr:hover{background:var(--soft)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
.bt{margin-top:6px}
.bt td,.bt th{padding:5px 8px}
.small{font-size:12px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.mkt .top{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:6px}
.book{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}
.book div{border-radius:8px;padding:8px 10px}
.book .bid{background:var(--ok-soft)}.book .ask{background:#fef2f2}
@media (prefers-color-scheme:dark){.book .ask{background:#2a1416}}
.book .p{font-size:20px;font-weight:650;font-variant-numeric:tabular-nums}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:13px}
.kv span:nth-child(odd){color:var(--muted)}
.kv span:nth-child(even){text-align:right;font-variant-numeric:tabular-nums}
.verdict{margin-top:10px;padding:8px 10px;border-radius:8px;font-size:13px}
.verdict.yes{background:var(--ok-soft)}.verdict.no{background:var(--warn-soft)}
.filter{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
select{font:inherit;font-size:13px;padding:4px 8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--text)}
.method td,.method th{text-align:left}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.cols2 ul{margin:6px 0 0;padding-left:20px}.cols2 li{margin:4px 0}
footer{margin-top:36px;color:var(--muted);font-size:12px;text-align:center}
.empty{color:var(--muted);text-align:center;padding:18px}
@media (max-width:980px){.kpis{grid-template-columns:repeat(3,1fr)}}
@media (max-width:760px){.intro,.grid2,.cols2{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}h1{font-size:20px}
.accts{grid-template-columns:1fr}.kpi .val{font-size:18px}}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div>
    <h1>加密货币策略 · 实时模拟盘</h1>
    <p class="sub">第一阶段在 7 年历史数据上回测选出的策略，现在用 Deribit 交易所的实时行情 7×24 小时自动交易（模拟资金）</p>
  </div>
  <div class="links">
    <span class="pill" id="status"><i></i><span id="status-text">正在读取数据…</span></span>
    <a class="btn primary" id="gh" href="__GITHUB__" target="_blank" rel="noopener">第一阶段研究与代码（GitHub）</a>
  </div>
</header>

<div class="intro">
  <div class="card"><b>① 它在做什么</b><p>研究阶段挑出的 7 个策略、外加 2 个“买入持有”对照组，每个账户 100 万美元模拟资金，在美国服务器上按事先写好的规则自动下单，没有人工干预。</p></div>
  <div class="card"><b>② 价格从哪里来</b><p>每笔成交都用下单那一刻 Deribit 交易所真实的<b style="display:inline">买一/卖一价</b>；按真实费率扣手续费；永续合约按真实的每小时资金费率记账；期权到期按 Deribit 官方结算价结算。</p></div>
  <div class="card"><b>③ 为什么要做这个</b><p>回测再仔细，也是在“已经知道的历史”上做的。模拟盘只用未来的、当时还不知道的行情，是对策略真正的样本外检验。全部成交和净值都可下载核对。</p></div>
</div>

<div class="kpis" id="kpis"></div>

<h2>净值走势</h2>
<p class="hint">每 5 分钟记一次各账户净值（含未平仓头寸按标记价估值）。点击图例可以隐藏/显示某条线；鼠标移到图上看具体数值。</p>
<div class="card">
  <div class="chart-head">
    <div class="seg" id="range"><button data-r="all" class="on">全部</button><button data-r="7d">近 7 天</button><button data-r="24h">近 24 小时</button></div>
    <span class="small">纵轴：累计收益率；横轴：北京时间</span>
  </div>
  <div id="chart"></div>
  <div class="legend" id="legend"></div>
</div>

<h2>策略账户</h2>
<p class="hint">每张卡片：策略在做什么、现在持有什么、下一步什么时候做什么，以及它在第一阶段回测里的成绩。点“规则、原理和回测”展开细节。</p>
<div class="accts" id="accts"></div>

<h2>实时行情（Deribit）</h2>
<p class="hint">模拟盘每 30 秒读取一次 Deribit 公开行情。下单时，买入用卖一价、卖出用买一价，和真实吃单成交一样。</p>
<div class="grid2" id="market"></div>
<div class="card" style="margin-top:12px">
  <h3 id="q-title">期权实时报价</h3>
  <p class="small" id="q-note" style="margin:-4px 0 8px"></p>
  <div class="tw"><table id="quotes"></table></div>
</div>

<h2>当前持仓</h2>
<p class="hint">市值按 Deribit 标记价（期权）、标记价（永续）和指数价（现货）计算。</p>
<div class="card tw"><table id="positions"></table></div>

<h2>成交记录</h2>
<p class="hint">每一笔都记下了当时用的实时报价，可以和 Deribit 的历史行情核对。完整记录可下载 CSV。</p>
<div class="card">
  <div class="filter"><label class="small" for="f-acct">账户</label><select id="f-acct"><option value="">全部</option></select>
  <span class="small" id="f-count"></span></div>
  <div class="tw"><table id="trades"></table></div>
</div>

<h2>资金费</h2>
<p class="hint">永续合约每小时结算一次资金费：费率为正时多头付给空头。资金费率套利账户靠它赚钱，趋势账户（做多）要付它。</p>
<div class="grid2">
  <div class="card tw"><h3>各账户累计</h3><table id="fund-sum"></table></div>
  <div class="card tw"><h3>最近记录</h3><table id="fund-recent"></table></div>
</div>

<h2>方法说明</h2>
<div class="card tw method">
  <h3>成交和费用怎么算（与 Deribit 规则一致）</h3>
  <table>
    <thead><tr><th>品种</th><th>成交价</th><th>手续费</th></tr></thead>
    <tbody>
    <tr><td>BTC / ETH 永续合约</td><td>买入按当时卖一价，卖出按买一价</td><td>成交金额 × 0.05%（吃单费率）</td></tr>
    <tr><td>BTC / ETH 期权</td><td>买入按卖一价，卖出按买一价（报价以币计，乘以指数价换成美元）</td><td>min(0.03% × 指数价, 12.5% × 权利金) × 张数</td></tr>
    <tr><td>期权到期</td><td>周五 08:00 UTC 按 Deribit 官方结算价（到期前 30 分钟指数均价）现金结算内在价值</td><td>min(0.015% × 结算价, 12.5% × 内在价值) × 张数</td></tr>
    <tr><td>现货（套利、买入持有）</td><td>Deribit 指数价（多家现货交易所加权）</td><td>0.10%</td></tr>
    <tr><td>资金费</td><td>每个整点按 Deribit 公布的该小时费率 × 指数价 × 持仓数量记账</td><td>—</td></tr>
    </tbody>
  </table>
</div>
<div class="cols2" style="margin-top:12px">
  <div class="card"><h3>和真实交易还差什么</h3><ul>
    <li>只按最优一档价格成交，默认一档挂单量够大；期权单笔金额较大时，真实成交可能要吃掉好几档。</li>
    <li>每 30 秒检查一次，成交按检查那一刻的报价，不考虑排队和网络延迟。</li>
    <li>现货腿用 Deribit 指数价模拟，实际需要在现货交易所买入。</li>
    <li>保证金按美元简化处理，没有模拟 Deribit 的组合保证金和强平。</li>
    <li>运行几天到几周的结果只能展示流程，统计上说明不了好坏；判断策略主要看第一阶段的多年回测。</li>
  </ul></div>
  <div class="card"><h3>第一阶段研究（历史回测）</h3><ul>
    <li>在 2019–2026 年的数据上评估了 <b>1,632 个策略变体</b>（16 个策略家族），覆盖毫秒、秒、分钟、小时、日、周 6 个时间维度。</li>
    <li>回测引擎：<b>QuantConnect LEAN</b>（开源的机构级回测引擎，官方 Docker 镜像）；数据：<b>Deribit 官方历史数据</b>（期权和永续逐笔成交、结算价、资金费率）。</li>
    <li>参数只用 2019–2023 年挑选，2024–2026 年只检验一次；107 次 LEAN 回测的 33 万笔期权成交逐笔核验过。</li>
    <li>结论：大部分策略扣完真实成本都亏；这里运行的是少数在两段时间都赚钱的。</li>
    <li>完整报告、代码和结果：<a id="gh2" href="__GITHUB__" target="_blank" rel="noopener">GitHub 仓库</a>。</li>
  </ul></div>
</div>

<h2>导出数据</h2>
<div class="links"><a class="btn" href="equity.csv">净值（每 5 分钟）</a><a class="btn" href="trades.csv">全部成交</a><a class="btn" href="positions.csv">当前持仓</a><a class="btn" href="funding.csv">资金费明细</a><a class="btn" href="state.json">页面数据 JSON</a></div>

<h2>运行日志</h2>
<p class="hint">程序自己写的最近 40 条记录（UTC 时间）。</p>
<div class="card tw"><table id="log"></table></div>

<footer>
  模拟交易，不涉及真实资金，也不构成投资建议 · 行情来自 Deribit 公共 API · 页面只读，每 30 秒自动更新 ·
  <a href="__GITHUB__" target="_blank" rel="noopener">代码与报告</a>
</footer>
</div>

<script>
"use strict";
const ORDER = ["BTC_TSMOM28","ETH_TSMOM112","BTC_CARRY","ETH_CARRY","BTC_STRADDLE_DH","BTC_PUT30_HAR","ETH_STRANGLE25_HAR","BTC_HOLD","ETH_HOLD"];
const COL = {COMBO:"var(--combo)",BTC_TSMOM28:"#2563eb",ETH_TSMOM112:"#7c3aed",BTC_CARRY:"#0d9488",ETH_CARRY:"#65a30d",
  BTC_STRADDLE_DH:"#ea580c",BTC_PUT30_HAR:"#db2777",ETH_STRANGLE25_HAR:"#ca8a04",BTC_HOLD:"#64748b",ETH_HOLD:"#94a3b8"};
const META = {
 COMBO:{cn:"组合：回测里最好的搭配",cat:"组合 · 等额资金",
  one:"下面 3 个卖期权账户 + BTC 资金费率套利，各占四分之一资金。四者的日收益几乎不相关（相关系数 0–0.19），合在一起更平稳。",
  rules:["组合净值 = 四个账户净值的平均（相当于把 100 万平均分成四份）。","四个账户各自按自己的规则交易，组合本身不另外下单。"],
  why:"卖期权是替别人承担大涨大跌的风险，资金费率套利是收加杠杆做多的人付的钱，两类收入来源不同。在第一阶段的全部策略里，这个组合在两段时间都跑赢国债，回撤最小。",
  bt:[["2020–2023（样本内）","+6.0%","0.7","9.0%"],["2024–2026（样本外）","+7.9%","0.6","7.5%"]],
  note:"注意：这 3 个期权策略是看过两段结果之后挑的，2024–2026 年的数字偏乐观。这个模拟盘就是对它真正的前向检验。同期国债约 1.8%（2019–2023）和 4.4%（2024–2026）。"},
 BTC_TSMOM28:{cn:"BTC 趋势跟踪",cat:"趋势跟踪 · 日线",
  one:"过去 28 天 BTC 涨了就做多，没涨就空仓；行情越剧烈，仓位越小。",
  rules:["每天 00:05 UTC（北京时间 08:05）用前一天的日收盘价，计算过去 28 天的涨跌幅。","涨跌幅 > 0 就做多 BTC 永续合约，否则空仓（不做空）。",
   "仓位 = 40% ÷ 过去 30 天的年化波动率，最多 2 倍净值。","每天按最新的净值和波动率调整仓位。"],
  why:"时间序列动量：价格趋势往往会延续一段时间（Moskowitz、Ooi、Pedersen，2012），学术研究发现加密货币的动量效应尤其强（Liu、Tsyvinski，2021）。按波动率调仓位，是 CTA 基金的常规做法。",
  bt:[["2019–2023（样本内）","+75%","2.00","36%"],["2024–2026（样本外）","+12.6%","0.26","31%"]],
  note:"从 52 个趋势规则变体（动量、均线、通道突破、RSI 等）里只按 2019–2023 年挑出；2024–2026 年行情来回震荡，效果明显衰减。同期 BTC 买入持有：+50%（回撤 77%）、+19%（回撤 54%）。"},
 ETH_TSMOM112:{cn:"ETH 趋势跟踪",cat:"趋势跟踪 · 日线",
  one:"过去 112 天 ETH 涨了就做多，没涨就空仓；行情越剧烈，仓位越小。",
  rules:["每天 00:05 UTC（北京时间 08:05）用前一天的日收盘价，计算过去 112 天的涨跌幅。","涨跌幅 > 0 就做多 ETH 永续合约，否则空仓（不做空）。",
   "仓位 = 40% ÷ 过去 30 天的年化波动率，最多 2 倍净值。"],
  why:"同左：时间序列动量。ETH 在样本内表现最好的回看期更长（112 天）。",
  bt:[["2020–2023（样本内）","+60%","1.58","29%"],["2024–2026（样本外）","+7.8%","0.12","43%"]],
  note:"同样只按样本内挑选。同期 ETH 买入持有：+93%（回撤 79%）、−1.3%（回撤 68%）。"},
 BTC_CARRY:{cn:"BTC 资金费率套利",cat:"套利 · 每小时记账",
  one:"同时买入 BTC 现货、做空同样数量的 BTC 永续合约：价格涨跌两边抵消，只赚永续合约的资金费。",
  rules:["开始时建仓一次：买入约 83 万美元的 BTC 现货，同时卖出同样数量的 BTC 永续（剩下的资金作保证金缓冲）。","之后一直持有，不再交易。",
   "每个整点按 Deribit 公布的该小时资金费率记账：费率为正时，空头收钱。"],
  why:"永续合约没有到期日，交易所用资金费让它的价格贴近现货。加密市场里借钱做多的人多，资金费大多数时间为正，由多头付给空头。",
  bt:[["2019–2023（样本内）","+7.1%","—","3.3%"],["2024–2026（样本外）","+5.2%","—","0.3%"]],
  note:"净值波动很小，是因为资金费本身很平稳；真正的风险是交易所违约、暴涨时空单要追加保证金、资金费长期为负。2025–2026 年收益已接近国债。"},
 ETH_CARRY:{cn:"ETH 资金费率套利",cat:"套利 · 每小时记账",
  one:"同时买入 ETH 现货、做空同样数量的 ETH 永续合约，只赚资金费。",
  rules:["开始时建仓一次：买入约 83 万美元的 ETH 现货，同时卖出同样数量的 ETH 永续。","之后一直持有；每个整点按真实资金费率记账。"],
  why:"同 BTC 资金费率套利。",
  bt:[["2019–2023（样本内）","+5.7%","—","5.6%"],["2024–2026（样本外）","+3.1%","—","0.6%"]],
  note:"ETH 的资金费近两年明显下降，2025–2026 年低于国债。"},
 BTC_STRADDLE_DH:{cn:"卖 BTC 平值跨式 + 每日对冲",cat:"卖期权 · 周度",
  one:"每周五卖出一周后到期的 BTC 平值看涨和看跌期权，收权利金；每天用永续合约把价格方向的风险对冲掉。",
  rules:["周五 09:00 UTC（北京 17:00）卖出下周五 08:00 到期、行权价最接近当前价格的看涨和看跌期权各一份；每份的名义金额 = 账户净值（最多 25 BTC）。",
   "开仓后立即、以及之后每天 08:00 UTC，用 BTC 永续把组合的 Delta（对价格方向的敞口）调回 0。","到期按 Deribit 官方结算价现金结算，平掉永续对冲；下个周五再开新仓。"],
  why:"卖期权相当于卖保险：期权价格里隐含的波动率，长期看略高于之后实际发生的波动（波动率风险溢价，见 Lucic & Sepp 2024 对 Deribit 的研究）。每天对冲后，赚的就是两者之差。",
  bt:[["2019–2023（样本内）","+3.2%","0.16","24.5%"],["2024–2026（样本外）","+7.6%","0.27","17.1%"]],
  note:"这个结构是在全样本的市场研究里先发现的，所以 2024–2026 对它不算完全没看过的数据。极端行情的一周可能亏掉净值的 20% 以上。"},
 BTC_PUT30_HAR:{cn:"卖 BTC 看跌期权（波动率过滤）",cat:"卖期权 · 周度",
  one:"只在期权“偏贵”的那些周，卖出一份虚值 BTC 看跌期权赚权利金；其余时间空仓。",
  rules:["每周一 09:30 UTC（北京 17:30），取当周五到期的平值期权隐含波动率（IV），和 HAR 模型预测的未来 7 天波动率比较。",
   "IV ≥ 1.1 × HAR 预测时，卖出当周五到期、Delta 约 0.30 的看跌期权（名义金额 = 账户净值，最多 25 BTC）；否则这周不交易。","不对冲，周五 08:00 UTC 按官方结算价结算。"],
  why:"IV 明显高于模型预测，说明期权相对“贵”，这时卖出的期望收益更高。HAR-RV 是 Corsi（2009）提出的波动率预测模型：用过去 1 天、7 天、30 天的已实现波动率（5 分钟收益）做线性回归，这里滚动用最近 365 天拟合。",
  bt:[["2019–2023（样本内）","+4.3%","0.25","18.6%"],["2024–2026（样本外）","+8.2%","0.41","14.0%"]],
  note:"从 1,120 个卖期权变体里只按 2019–2023 年挑出（要求 t ≥ 2、至少 60% 的年份赚钱）。BTC 一周内暴跌时会亏得比较多。"},
 ETH_STRANGLE25_HAR:{cn:"卖 ETH 宽跨式（波动率过滤）",cat:"卖期权 · 周度",
  one:"只在期权“偏贵”的那些周，卖出 ETH 虚值看涨和看跌期权各一份赚权利金；其余时间空仓。",
  rules:["每周一 09:30 UTC 用同样的方法比较 ETH 平值 IV 和 HAR 预测。",
   "IV ≥ 1.1 × HAR 预测时，卖出当周五到期、Delta 约 0.25 的看涨和看跌期权各一份（每份名义金额 = 账户净值，最多 400 ETH）；否则这周不交易。","不对冲，周五 08:00 UTC 结算。"],
  why:"同上：只在期权偏贵时卖。宽跨式同时卖看涨和看跌，ETH 只要这一周不大涨也不大跌就赚钱。",
  bt:[["2020–2023（样本内）","+6.6%","1.14","3.4%"],["2024–2026（样本外）","+8.9%","0.44","13.8%"]],
  note:"在 1,120 个变体里 ETH 样本内排名第一。但它的样本外证据最弱：不经过 LEAN 的快速筛选里，2024–2026 年是 −0.7%/年（开仓 29 周），LEAN 全成本回测是 +2.4%/年（开仓 42 周），两种方法取隐含波动率的方式不同。ETH 一周内大涨或大跌都会亏。"},
 BTC_HOLD:{cn:"对照：BTC 买入持有",cat:"对照组",one:"开始时按指数价买入 BTC 现货，一直不动，用来和上面的策略比较。",
  rules:["开始时买入一次，之后不交易。"],why:"任何主动策略都应该先和“什么都不做”比较。",
  bt:[["2019–2023","+50%","0.70","77%"],["2024–2026","+19%","0.31","54%"]],note:"收益高，但回撤也极大。"},
 ETH_HOLD:{cn:"对照：ETH 买入持有",cat:"对照组",one:"开始时按指数价买入 ETH 现货，一直不动，用来和上面的策略比较。",
  rules:["开始时买入一次，之后不交易。"],why:"任何主动策略都应该先和“什么都不做”比较。",
  bt:[["2020–2023","+93%","1.02","79%"],["2024–2026","−1.3%","−0.08","68%"]],note:"2024–2026 年 ETH 基本没涨，回撤却接近七成。"}
};

let S = null, RANGE = "all", HIDDEN = new Set(), LOADED = 0;
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const ok = v => v !== null && v !== undefined && !Number.isNaN(v);
const fx = (v, d) => ok(v) ? Number(v).toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d}) : "—";
const usd = (v, d = 0) => ok(v) ? (v < 0 ? "−$" : "$") + fx(Math.abs(v), d) : "—";
const sgnUsd = (v, d = 0) => ok(v) ? (v > 0 ? "+" : v < 0 ? "−" : "") + "$" + fx(Math.abs(v), d) : "—";
const pct = (v, d = 2) => ok(v) ? (v > 0 ? "+" : v < 0 ? "−" : "") + fx(Math.abs(v * 100), d) + "%" : "—";
const cls = v => ok(v) ? (v > 1e-12 ? "pos" : v < -1e-12 ? "neg" : "") : "";
const tUTC = s => new Date(s.replace(" ", "T") + "Z");
const fmtBJ = d => new Date(d.getTime() + 8 * 3600e3).toISOString().slice(5, 16).replace("T", " ");
const fmtUTC = d => d.toISOString().slice(5, 16).replace("T", " ");
const both = s => { const d = tUTC(s); return `${fmtUTC(d)} UTC<br><span class="small">北京 ${fmtBJ(d)}</span>`; };
const name = a => (META[a] || {cn: a}).cn;
function dur(ms) {
  const s = Math.max(0, Math.floor(ms / 1000)), d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  return d ? `${d} 天 ${h} 小时` : h ? `${h} 小时 ${m} 分` : `${m} 分 ${s % 60} 秒`;
}

async function load() {
  try {
    const r = await fetch("state.json?_=" + Date.now(), {cache: "no-store"});
    if (!r.ok) throw new Error("HTTP " + r.status);
    S = await r.json();
    LOADED = Date.now();
    render();
  } catch (e) {
    $("#status").classList.add("bad");
    $("#status-text").textContent = "暂时读不到数据，稍后自动重试";
  }
}

function comboSeries() {
  const c = S.combo, t = S.eq.t;
  return t.map((_, i) => {
    let s = 0;
    for (const a of c) { const v = (S.eq.s[a] || [])[i]; if (!ok(v)) return null; s += v; }
    return s / c.length;
  });
}

function comboStats() {
  const acc = Object.fromEntries(S.accounts.map(a => [a.name, a]));
  const m = S.combo.map(a => acc[a]);
  const eq = m.reduce((s, a) => s + a.equity, 0) / m.length;
  const prev = m.reduce((s, a) => s + a.equity / (1 + a.ret24h), 0) / m.length;
  let peak = S.start_equity, mdd = 0;
  for (const v of comboSeries().concat([eq])) { if (!ok(v)) continue; peak = Math.max(peak, v); mdd = Math.max(mdd, 1 - v / peak); }
  return {name: "COMBO", equity: eq, ret: eq / S.start_equity - 1, ret24h: eq / prev - 1, mdd,
          trades: m.reduce((s, a) => s + a.trades, 0), fees: m.reduce((s, a) => s + a.fees, 0), funding: m.reduce((s, a) => s + a.funding, 0)};
}

function render() {
  const gen = tUTC(S.generated), age = Date.now() - gen.getTime();
  const st = $("#status");
  st.classList.toggle("bad", age > 180e3);
  $("#status-text").textContent = age > 180e3 ? `数据已 ${dur(age)} 没有更新` : `运行中 · 行情时间 ${fmtBJ(gen)}（北京）`;
  if (S.github) for (const id of ["#gh", "#gh2"]) $(id).href = S.github;
  renderKpis(); drawChart(); renderAccounts(); renderMarket(); renderPositions(); renderTrades(); renderFunding(); renderLog();
}

function nextEvent() {
  let best = null;
  for (const a of S.accounts) {
    if (!a.next_ts || a.name.includes("CARRY")) continue;
    const t = tUTC(a.next_ts).getTime();
    if (!best || t < best.t) best = {t, a};
  }
  return best;
}

function renderKpis() {
  const c = comboStats(), started = tUTC(S.started);
  const fees = S.accounts.reduce((s, a) => s + a.fees, 0), fund = S.accounts.reduce((s, a) => s + a.funding, 0);
  const nx = nextEvent();
  const k = [
    ["已运行", dur(Date.now() - started.getTime()), `开始于 ${fmtBJ(started)}（北京）`, "small"],
    ["组合累计收益", `<span class="${cls(c.ret)}">${pct(c.ret)}</span>`, `净值 ${usd(c.equity)} · 近 24h ${pct(c.ret24h)}`],
    ["成交笔数", fx(S.n_trades, 0), "9 个账户合计"],
    ["已付手续费", usd(fees), "按 Deribit 真实费率"],
    ["资金费净收入", `<span class="${cls(fund)}">${sgnUsd(fund)}</span>`, "正 = 收到，负 = 付出"],
    ["下一次交易", nx ? `<span class="cd" data-t="${nx.t}">${dur(nx.t - Date.now())}</span>` : "—",
      nx ? `${esc(name(nx.a.name))}：${fmtBJ(new Date(nx.t))}（北京）` : "", "small"],
  ];
  $("#kpis").innerHTML = k.map(([l, v, f, sz]) =>
    `<div class="card kpi"><div class="lbl">${l}</div><div class="val ${sz || ""}">${v}</div><div class="foot">${f}</div></div>`).join("");
}

function niceTicks(lo, hi, n) {
  const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), e = raw / mag;
  const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.round(v / step) * step);
  return {ticks: out, step};
}

function drawChart() {
  const box = $("#chart");
  if (!S || !S.eq || S.eq.t.length < 2) { box.innerHTML = '<div class="empty">净值记录还不够画图，几分钟后再看。</div>'; $("#legend").innerHTML = ""; return; }
  const T = S.eq.t.map(s => tUTC(s).getTime()), tEnd = T[T.length - 1];
  const span = RANGE === "24h" ? 864e5 : RANGE === "7d" ? 7 * 864e5 : Infinity;
  let i0 = 0; while (i0 < T.length - 2 && tEnd - T[i0] > span) i0++;
  const series = [{k: "COMBO", vals: comboSeries(), w: 3}].concat(ORDER.map(a => ({k: a, vals: S.eq.s[a] || [], w: 1.7, dash: a.endsWith("HOLD") ? "5 4" : ""})));
  for (const s of series) s.r = s.vals.map(v => ok(v) ? (v / S.start_equity - 1) * 100 : null);
  const vis = series.filter(s => !HIDDEN.has(s.k));
  let lo = Infinity, hi = -Infinity;
  for (const s of vis) for (let i = i0; i < T.length; i++) { const v = s.r[i]; if (ok(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  if (!isFinite(lo)) { lo = -1; hi = 1; }
  lo = Math.min(lo, 0); hi = Math.max(hi, 0);
  if (hi - lo < 0.2) { const m = (hi + lo) / 2; lo = m - 0.1; hi = m + 0.1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const W = Math.max(320, box.clientWidth || 800), H = W < 600 ? 250 : 340, L = 52, R = 14, Tp = 12, B = 30;
  const X = t => L + (W - L - R) * (t - T[i0]) / Math.max(1, tEnd - T[i0]);
  const Y = v => Tp + (H - Tp - B) * (hi - v) / (hi - lo);
  const {ticks, step} = niceTicks(lo, hi, 5), dec = step < 0.1 ? 2 : step < 1 ? 1 : 0;
  let g = "";
  for (const v of ticks) {
    g += `<line x1="${L}" x2="${W - R}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" style="stroke:var(--line)" ${Math.abs(v) < 1e-9 ? 'stroke-width="1.6"' : ""}/>`;
    g += `<text x="${L - 8}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end" font-size="11" style="fill:var(--muted)">${(v > 0 ? "+" : "") + v.toFixed(dec)}%</text>`;
  }
  const nx = W < 600 ? 3 : 6;
  for (let j = 0; j <= nx; j++) {
    const t = T[i0] + (tEnd - T[i0]) * j / nx, x = X(t);
    g += `<text x="${x.toFixed(1)}" y="${H - 9}" text-anchor="${j === 0 ? "start" : j === nx ? "end" : "middle"}" font-size="11" style="fill:var(--muted)">${fmtBJ(new Date(t))}</text>`;
  }
  let lines = "";
  for (const s of vis) {
    let d = "", pen = false;
    for (let i = i0; i < T.length; i++) {
      const v = s.r[i];
      if (!ok(v)) { pen = false; continue; }
      d += (pen ? "L" : "M") + X(T[i]).toFixed(1) + " " + Y(v).toFixed(1);
      pen = true;
    }
    lines += `<path d="${d}" fill="none" style="stroke:${COL[s.k]}" stroke-width="${s.w}" stroke-linejoin="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`;
  }
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="各账户累计收益率曲线">${g}${lines}
    <line id="xh" x1="0" x2="0" y1="${Tp}" y2="${H - B}" style="stroke:var(--muted)" stroke-dasharray="3 3" visibility="hidden"/>
    <rect x="${L}" y="${Tp}" width="${W - L - R}" height="${H - Tp - B}" fill="transparent" id="hit"/></svg><div class="tip" id="tip"></div>`;
  const svg = box.querySelector("svg"), tip = $("#tip"), xh = $("#xh");
  const show = cx => {
    const r = svg.getBoundingClientRect(), x = (cx - r.left) * W / r.width;
    const t = T[i0] + (x - L) / (W - L - R) * (tEnd - T[i0]);
    let a = i0, b = T.length - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (T[m] < t) a = m; else b = m; }
    const i = Math.abs(T[a] - t) < Math.abs(T[b] - t) ? a : b;
    xh.setAttribute("x1", X(T[i])); xh.setAttribute("x2", X(T[i])); xh.setAttribute("visibility", "visible");
    const rows = vis.filter(s => ok(s.r[i])).sort((p, q) => q.r[i] - p.r[i]);
    tip.innerHTML = `<div class="small" style="margin-bottom:4px">${fmtBJ(new Date(T[i]))}（北京）</div>` + rows.map(s =>
      `<div class="row"><span><i style="background:${COL[s.k]}"></i>${esc(name(s.k))}</span><b class="${cls(s.r[i])}">${(s.r[i] > 0 ? "+" : "") + s.r[i].toFixed(2)}%</b></div>`).join("");
    tip.style.display = "block";
    const px = X(T[i]) * r.width / W, tw = tip.offsetWidth;
    tip.style.left = Math.max(0, Math.min(r.width - tw, px + 14 + tw > r.width ? px - tw - 14 : px + 14)) + "px";
    tip.style.top = "8px";
  };
  const hide = () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); };
  svg.addEventListener("mousemove", e => show(e.clientX));
  svg.addEventListener("mouseleave", hide);
  svg.addEventListener("touchstart", e => show(e.touches[0].clientX), {passive: true});
  svg.addEventListener("touchmove", e => show(e.touches[0].clientX), {passive: true});
  svg.addEventListener("touchend", hide);
  $("#legend").innerHTML = series.map(s => `<button data-k="${s.k}" class="${HIDDEN.has(s.k) ? "off" : ""}"><i style="background:${COL[s.k]}"></i>${esc(name(s.k))}</button>`).join("");
}

function btTable(m) {
  return `<table class="bt"><thead><tr><th class="l">回测区间</th><th>年化收益</th><th>夏普</th><th>最大回撤</th></tr></thead><tbody>` +
    m.bt.map(r => `<tr><td class="l">${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("") + `</tbody></table>`;
}

function card(a, combo) {
  const m = META[a.name], cat = m.cat.startsWith("对照") ? "chip gray" : "chip";
  const nxt = a.next_ts ? (() => { const d = tUTC(a.next_ts); return `<div><span class="k">下一步：</span>${fmtUTC(d)} UTC（北京 ${fmtBJ(d)}），${esc(a.next_what)}
      · 还有 <span class="cd" data-t="${d.getTime()}">${dur(d.getTime() - Date.now())}</span></div>`; })() : "";
  const now = combo ? `<div><span class="k">成员：</span>${S.combo.map(x => esc(name(x))).join("、")}</div>` : `<div><span class="k">现在：</span>${esc(a.now)}</div>`;
  return `<div class="card acct${combo ? " combo" : ""}" style="--c:${COL[a.name]}">
    <div class="acct-h"><b>${esc(m.cn)}</b>${combo ? "" : `<code>${a.name}</code>`}<span class="${cat}">${m.cat}</span></div>
    <div class="one">${m.one}</div>
    <div class="nums">
      <div><div class="lbl">当前净值</div><div class="big">${usd(a.equity)}</div></div>
      <div><div class="lbl">累计收益</div><div class="big ${cls(a.ret)}">${pct(a.ret)}</div></div>
      <div><div class="lbl">近 24 小时</div><div class="mid ${cls(a.ret24h)}">${pct(a.ret24h)}</div></div>
      <div><div class="lbl">最大回撤</div><div class="mid">${pct(a.mdd).replace("+", "")}</div></div>
      <div><div class="lbl">成交</div><div class="mid">${a.trades} 笔</div></div>
      <div><div class="lbl">手续费</div><div class="mid">${usd(a.fees)}</div></div>
      ${Math.abs(a.funding) > 0.005 ? `<div><div class="lbl">资金费</div><div class="mid ${cls(a.funding)}">${sgnUsd(a.funding)}</div></div>` : ""}
    </div>
    <div class="state">${now}${nxt}</div>
    <details><summary>规则、原理和回测结果</summary>
      <ul>${m.rules.map(r => `<li>${r}</li>`).join("")}</ul>
      <div class="why"><b>为什么可能赚钱：</b>${m.why}</div>
      ${btTable(m)}
      <p class="small" style="margin:6px 0 0">${m.note}</p>
    </details></div>`;
}

function renderAccounts() {
  const open = new Set([...document.querySelectorAll("#accts details[open]")].map(d => d.closest(".acct").dataset.k));
  const acc = Object.fromEntries(S.accounts.map(a => [a.name, a]));
  const html = [card(comboStats(), true)].concat(ORDER.filter(k => acc[k]).map(k => card(acc[k], false)));
  $("#accts").innerHTML = html.join("");
  [...document.querySelectorAll("#accts .acct")].forEach((el, i) => {
    el.dataset.k = i === 0 ? "COMBO" : ORDER.filter(k => acc[k])[i - 1];
    if (open.has(el.dataset.k)) el.querySelector("details").open = true;
  });
}

function renderMarket() {
  const out = [];
  for (const c of ["BTC", "ETH"]) {
    const m = S.market[c];
    if (!m || !ok(m.bid)) { out.push(`<div class="card mkt"><h3>${c}</h3><div class="empty">行情暂缺</div></div>`); continue; }
    const ratio = ok(m.atm_iv) && ok(m.har) ? m.atm_iv / m.har : null;
    const who = c === "BTC" ? "BTC 卖看跌" : "ETH 宽跨";
    const v = ratio === null ? "" : ratio >= 1.1
      ? `<div class="verdict yes">IV / HAR = <b>${ratio.toFixed(2)}</b> ≥ 1.10：如果现在是周一 09:30，“${who}”账户<b>会</b>开仓。</div>`
      : `<div class="verdict no">IV / HAR = <b>${ratio.toFixed(2)}</b> &lt; 1.10：如果现在是周一 09:30，“${who}”账户<b>不会</b>开仓（期权不够贵）。</div>`;
    out.push(`<div class="card mkt">
      <div class="top"><h3>${c} 永续合约 <span class="small">${c}-PERPETUAL</span></h3><span class="small">24h ${pct(ok(m.change_24h) ? m.change_24h / 100 : null)}</span></div>
      <div class="book"><div class="bid"><div class="small">买一（卖出时成交价）</div><div class="p">${fx(m.bid, 2)}</div><div class="small">挂单 ${fx(m.bid_size, 0)} USD</div></div>
        <div class="ask"><div class="small">卖一（买入时成交价）</div><div class="p">${fx(m.ask, 2)}</div><div class="small">挂单 ${fx(m.ask_size, 0)} USD</div></div></div>
      <div class="kv">
        <span>买卖价差</span><span>${fx(m.ask - m.bid, 2)}（${fx(m.spread_bp, 2)} 个基点）</span>
        <span>指数价（现货加权）</span><span>${fx(m.index, 2)}</span>
        <span>标记价</span><span>${fx(m.mark, 2)}</span>
        <span>资金费率（8 小时）</span><span>${ok(m.funding_8h) ? pct(m.funding_8h, 4) : "—"}（年化 ${pct(m.funding_ann, 1)}）</span>
        <span>24 小时成交额</span><span>${ok(m.volume_usd) ? "$" + fx(m.volume_usd / 1e6, 0) + " 百万" : "—"}</span>
        <span>平值期权隐含波动率（${m.exp ? m.exp.slice(5, 10) : "—"} 到期）</span><span>${pct(m.atm_iv, 1).replace("+", "")}</span>
        <span>HAR 模型预测波动率（未来 7 天）</span><span>${pct(m.har, 1).replace("+", "")}</span>
      </div>${v}</div>`);
  }
  $("#market").innerHTML = out.join("");
  const rows = [];
  for (const c of ["BTC", "ETH"]) for (const q of (S.market[c] || {}).quotes || []) rows.push([c, q]);
  const any = (S.market.BTC || {}).exp || (S.market.ETH || {}).exp;
  $("#q-title").textContent = any ? `期权实时报价（最近一个周五 ${any.slice(5, 10)} 08:00 UTC 到期）` : "期权实时报价";
  $("#q-note").textContent = "这些就是期权策略开仓时会用到的报价：卖出按买一价成交。报价以币计（如 0.0123 BTC），乘以指数价换成美元。" +
    (S.market.BTC && S.market.BTC.quotes_ts ? `报价时间 ${S.market.BTC.quotes_ts.slice(5, 16)} UTC。` : "");
  $("#quotes").innerHTML = rows.length ? `<thead><tr><th class="l">合约</th><th class="l">说明</th><th>行权价</th><th>买一（币）</th><th>卖一（币）</th><th>买一（美元）</th><th>卖一（美元）</th><th>价差 / 中间价</th><th>隐含波动率</th><th>Delta</th></tr></thead><tbody>` +
    rows.map(([c, q]) => { const mid = (q.bid + q.ask) / 2;
      return `<tr><td class="l"><code>${esc(q.name)}</code></td><td class="l">${esc(q.label)}</td><td>${fx(q.K, 0)}</td><td>${q.bid || "—"}</td><td>${q.ask || "—"}</td>
      <td>${usd(q.bid_usd, 2)}</td><td>${usd(q.ask_usd, 2)}</td><td>${mid > 0 && q.bid > 0 ? pct((q.ask - q.bid) / mid, 1).replace("+", "") : "—"}</td><td>${pct(q.iv, 1).replace("+", "")}</td><td>${fx(q.delta, 2)}</td></tr>`; }).join("") + "</tbody>"
    : `<tbody><tr><td class="empty">暂无报价</td></tr></tbody>`;
}

function renderPositions() {
  const rows = [];
  for (const k of ORDER) { const a = S.accounts.find(x => x.name === k); if (a) for (const p of a.positions) rows.push([a, p]); }
  $("#positions").innerHTML = `<thead><tr><th class="l">账户</th><th class="l">合约</th><th class="l">说明</th><th>数量</th><th>开仓均价</th><th>现价</th><th>市值</th><th>浮动盈亏</th></tr></thead><tbody>` +
    (rows.length ? rows.map(([a, p]) => `<tr><td class="l">${esc(name(a.name))}</td><td class="l"><code>${esc(p.inst)}</code></td><td class="l">${esc(p.desc)}</td>
      <td class="num">${fx(p.qty, 4)}</td><td>${fx(p.avg, 2)}</td><td>${fx(p.price, 2)}</td><td>${usd(p.value)}</td><td class="${cls(p.upnl)}">${sgnUsd(p.upnl)}</td></tr>`).join("")
      : `<tr><td colspan="8" class="empty">暂无持仓</td></tr>`) + "</tbody>";
}

function renderTrades() {
  const sel = $("#f-acct"), cur = sel.value;
  if (sel.options.length <= 1) sel.innerHTML = `<option value="">全部</option>` + ORDER.map(k => `<option value="${k}">${esc(name(k))}</option>`).join("");
  sel.value = cur;
  const list = S.trades.filter(t => !sel.value || t.acct === sel.value);
  $("#f-count").textContent = `共 ${S.n_trades} 笔${S.trades.length < S.n_trades ? `，这里显示最近 ${S.trades.length} 笔` : ""}`;
  $("#trades").innerHTML = `<thead><tr><th class="l">时间</th><th class="l">账户</th><th class="l">合约</th><th class="l">方向</th><th>数量</th><th>成交价（美元）</th><th class="l">当时的实时报价</th><th>手续费</th><th class="l">原因</th></tr></thead><tbody>` +
    (list.length ? list.map(t => `<tr><td class="l">${both(t.ts)}</td><td class="l">${esc(name(t.acct))}</td><td class="l"><code>${esc(t.inst)}</code><br><span class="small">${esc(t.desc)}</span></td>
      <td class="l ${t.qty > 0 ? "pos" : "neg"}">${t.qty > 0 ? "买入" : "卖出"}</td><td class="num">${fx(Math.abs(t.qty), 4)}</td><td>${fx(t.price, 2)}</td>
      <td class="l small">${esc(t.quote) || "—"}</td><td>${fx(t.fee, 2)}</td><td class="l">${esc(t.tag)}</td></tr>`).join("")
      : `<tr><td colspan="9" class="empty">没有成交</td></tr>`) + "</tbody>";
}

function renderFunding() {
  const has = S.accounts.filter(a => a.funding_hours > 0);
  $("#fund-sum").innerHTML = `<thead><tr><th class="l">账户</th><th>记账小时数</th><th>累计</th></tr></thead><tbody>` +
    (has.length ? has.map(a => `<tr><td class="l">${esc(name(a.name))}</td><td>${a.funding_hours}</td><td class="${cls(a.funding)}">${sgnUsd(a.funding, 2)}</td></tr>`).join("")
      : `<tr><td colspan="3" class="empty">暂无</td></tr>`) + "</tbody>";
  $("#fund-recent").innerHTML = `<thead><tr><th class="l">整点（UTC）</th><th class="l">账户</th><th>该小时费率</th><th>金额</th></tr></thead><tbody>` +
    (S.funding.length ? S.funding.slice(0, 18).map(f => `<tr><td class="l">${f.ts.slice(5, 16)}</td><td class="l">${esc(name(f.acct))}</td><td>${pct(f.rate, 5)}</td><td class="${cls(f.amount)}">${sgnUsd(f.amount, 2)}</td></tr>`).join("")
      : `<tr><td colspan="4" class="empty">暂无</td></tr>`) + "</tbody>";
}

function renderLog() {
  $("#log").innerHTML = `<thead><tr><th class="l">时间（UTC）</th><th class="l">内容</th></tr></thead><tbody>` +
    S.log.map(l => `<tr><td class="l" style="white-space:nowrap">${esc(l.ts.slice(5))}</td><td class="l">${esc(l.msg)}</td></tr>`).join("") + "</tbody>";
}

function tick() {
  const now = Date.now();
  for (const el of document.querySelectorAll(".cd")) el.textContent = dur(+el.dataset.t - now);
  if (S && LOADED) {
    const age = now - tUTC(S.generated).getTime();
    if (age > 180e3) { $("#status").classList.add("bad"); $("#status-text").textContent = `数据已 ${dur(age)} 没有更新`; }
  }
}

$("#range").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  RANGE = b.dataset.r;
  for (const x of document.querySelectorAll("#range button")) x.classList.toggle("on", x === b);
  drawChart();
});
$("#legend").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  const k = b.dataset.k; HIDDEN.has(k) ? HIDDEN.delete(k) : HIDDEN.add(k);
  drawChart();
});
$("#f-acct").addEventListener("change", renderTrades);
let rz; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => S && drawChart(), 150); });
load(); setInterval(load, 30000); setInterval(tick, 1000);
</script>
</body>
</html>
"""
_PAGE_RAW = PAGE.replace("__GITHUB__", GITHUB).encode("utf-8")
PAGE_BYTES = (_PAGE_RAW, gzip.compress(_PAGE_RAW, 6))

STORE = ENGINE = None


def main():
    global STORE, ENGINE
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "web":
        serve_public()
        return
    STORE, market = Store(DB), Market()
    trader = Trader(STORE, market)
    ENGINE = trader.e
    if mode == "selftest":
        t = now_utc()
        market.refresh_perp()
        print("perp:", market.perp)
        for c in ("BTC", "ETH"):
            market.refresh_options(c, max_age=0)
            print(c, "options listed:", len(market.options[c]))
            fri = weekly_expiry(market, c, (t + timedelta(days=(4 - t.weekday()) % 7 or 7)).replace(hour=8, minute=0, second=0, microsecond=0))
            print(c, "next Friday expiry:", fri, "ATM pair:", atm_pair(market, c, fri) if fri else None)
            print(c, "0.30-delta put:", by_delta(market, c, fri, False, 0.30, t) if fri else None)
            print(c, "HAR forecast:", har_forecast(STORE, market, c))
            cl = market.daily_closes(c, 150)
            print(c, "daily closes:", len(cl), "last", cl[-1], "28d ret", cl[-1] / cl[-29] - 1, "112d ret", cl[-1] / cl[-113] - 1)
            print(c, "funding last hour:", market.funding_last_hour(c, t.replace(minute=0, second=0, microsecond=0)))
            print(c, "delivery 2026-10-16:", market.delivery_price(c, datetime(2026, 10, 16, tzinfo=UTC)))
        return
    pub = Publisher(trader)
    if mode == "export":                               # write the dashboard files once from the database, no trading
        market.refresh_perp()
        pub.publish(now_utc())
        print("wrote", sorted(os.listdir(PUBLIC)), "last error:", pub.last_err)
        return
    STORE.log(f"paper trader started (db {DB}, dashboard files in {PUBLIC})")
    while True:
        t = now_utc()
        try:
            trader.cycle(t)
        except Exception as ex:
            STORE.log(f"cycle error: {ex!r}\n{traceback.format_exc()[-1500:]}")
        if market.perp:
            pub.publish(t)
        time.sleep(max(5.0, 30.0 - (time.time() % 30.0)))


if __name__ == "__main__":
    main()
