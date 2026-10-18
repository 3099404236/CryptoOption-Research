# region imports
from AlgorithmImports import *
# endregion
"""
Intraday / "high-frequency" Deribit option strategies (V2).

Options are minute bars (Deribit options trade a few times per hour per strike, so nothing finer is
meaningful); the hedge instrument (the real perpetual) runs at second or minute resolution
(parameter `res`). Each strategy trades the nearest Friday weekly expiry at least 36 hours away and
holds to Deribit cash settlement unless it has its own exit rule. Parameters are fixed in advance
(no tuning on the test window). One unit = `unit_notional` x equity of notional per leg.

Strategies (parameter `strategy`):
  GAMMA_SCALPING              long ATM straddle, perpetual delta hedge when |delta| >= 4 % of a unit
  SABR_DELTA_HEDGED_STRANGLE  short 0.20-delta strangle hedged with the SABR (Bartlett) delta; SABR
                              (beta = 1) is re-fitted hourly to the expiry's out-of-the-money smile
  SHORT_STRANGLE_STOP_HEDGE   short 0.20-delta strangle; hold +/-1 unit of perpetual when spot is
                              more than 1.5 % from entry, flatten when back within 0.5 % (stop-loss hedge)
  AS_MARKET_MAKING            Avellaneda-Stoikov (2008) quotes on the ATM call and put: reservation
                              price r = mid - q*gamma*sigma^2*H, spread gamma*sigma^2*H + 2/gamma*ln(1+gamma/kappa),
                              gamma/kappa calibrated so that a full inventory shifts quotes by one market
                              half-spread and the quoted width equals the market width; fills only when
                              a real Deribit trade prints through the quote (25 % of printed volume);
                              hourly perpetual delta hedge
  NBP_FLOW_STRADDLE           net buying pressure (Alexander, Deng, Feng & Wan, JFM 2023): vega-weighted
                              signed ATM option flow over the last 60 min; z > 2 versus the trailing
                              168 hourly sums -> buy ATM straddle, exit after 4 h
  POST_SPIKE_PREMIUM_SELL     after a >1.5 % 60-minute move sell a 0.20-delta strangle; exit after 6 h
                              or when the loss exceeds 1.5x the premium received (honest name of the
                              original "negative gamma squeeze harvest")
  PARITY_CONVERSION_ARB       conversion / reversal against the perpetual at bid/ask, net of option,
                              perpetual and delivery fees and expected funding; both options must have
                              printed a real trade in the last 10 min; all legs at once, held to settlement
  TREND_RATIO_SPREAD_STOP     direction fixed ex ante by the trailing 24 h return: 1x2 put (down) or
                              call (up) ratio spread; closed when spot is 4 % beyond the short strike
  TREND_BUTTERFLY             direction by trailing 24 h return: symmetric put or call butterfly (w ~ 4 %)
  BUY_AND_HOLD                1x notional long underlying (spot proxy, no funding)
"""
import math
import os
from collections import deque
from datetime import datetime, timedelta

import numpy as np

from deribit_common import DeribitVenue, sabr_bartlett_delta, sabr_calibrate

MODES = ["GAMMA_SCALPING", "SABR_DELTA_HEDGED_STRANGLE", "SHORT_STRANGLE_STOP_HEDGE", "AS_MARKET_MAKING",
         "NBP_FLOW_STRADDLE", "POST_SPIKE_PREMIUM_SELL", "PARITY_CONVERSION_ARB", "TREND_RATIO_SPREAD_STOP",
         "TREND_BUTTERFLY", "BUY_AND_HOLD"]


class DeribitFlow(PythonData):
    """Per-minute signed option order flow written by build_lean_data.py (Deribit aggressor side)."""

    def GetSource(self, config, date, isLive):
        coin = config.Symbol.Value.upper().replace("FLOW", "").lower()
        path = os.path.join(Globals.DataFolder, "alternative", "deribit", "flow", coin, f"{date:%Y%m%d}.csv")
        return SubscriptionDataSource(path, SubscriptionTransportMedium.LocalFile)

    def Reader(self, config, line, date, isLive):
        if not line or not line[0].isdigit():
            return None
        p = line.split(",")
        d = DeribitFlow()
        d.Symbol = config.Symbol
        d.Time = datetime(date.year, date.month, date.day) + timedelta(milliseconds=int(p[0]))
        d.EndTime = d.Time + timedelta(minutes=1)
        d.Value = float(p[2])                 # ATM vega-weighted net buying pressure (USD per vol point)
        return d


class DeribitIntraday(QCAlgorithm):
    def Initialize(self):
        self.SetTimeZone(TimeZones.Utc)
        par = lambda k, d: self.GetParameter(k) or d
        self.coin = par("coin", "ETH")
        self.mode = par("strategy", "GAMMA_SCALPING")
        if self.mode not in MODES:
            raise ValueError(f"unknown strategy {self.mode}")
        s, e = par("start", "2026-10-11"), par("end", "2026-10-16")
        self.SetStartDate(*map(int, s.split("-")))
        self.SetEndDate(*map(int, e.split("-")))
        self.SetCash(float(par("cash", "1000000")))
        res = Resolution.Second if par("res", "minute") == "second" else Resolution.Minute
        self.unit_frac = float(par("unit_notional", "0.25"))
        self.cap = float(par("max_coins", "25" if self.coin == "BTC" else "400"))
        le = par("last_entry", "")
        self.last_entry = datetime.strptime(le, "%Y-%m-%d %H:%M") if le else None   # no new positions after
        self.venue = DeribitVenue(self, self.coin, underlying_resolution=res)
        if self.mode == "BUY_AND_HOLD":
            self.venue.funding = {}            # spot benchmark: no perpetual funding
        self.u = self.venue.u
        self.SetBenchmark(self.u)
        self.SetWarmUp(timedelta(days=8))      # trailing returns / order-flow history, no trading
        self.check_every = timedelta(seconds=45 if res == Resolution.Second else 60)
        self.last_check = datetime.min
        self._warming = True
        self.prices = deque(maxlen=2 * 24 * 60)     # 1-minute closes for return-based signals
        self._last_min = None
        self.book, self.book_expiry, self.book_day = [], None, None
        self.legs, self.unit, self.entry = [], 0.0, {}
        self.blocked_expiry = None                  # set after a stop-out: no re-entry this expiry
        self.sabr = None
        self.flow_minutes = deque()                 # (time, value) of the last 60 minutes
        self.flow_hours = deque(maxlen=168)         # completed hourly sums
        self._flow_hour, self._flow_hour_sum = None, 0.0
        self.last_trade_time = {}
        self.mm_last_quote = datetime.min
        self.mm_quotes = {}
        self.stats = {"cycles": 0, "hedges": 0, "signals": 0, "skipped": {}, "mm_fills": 0, "arb_trades": 0,
                      "arb_locked_edge_usd": 0.0, "stops": 0}
        if self.mode == "NBP_FLOW_STRADDLE":
            self.flow_sym = self.AddData(DeribitFlow, f"{self.coin}FLOW", Resolution.Minute).Symbol
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Friday), self.TimeRules.At(8, 5), self.after_settlement)
        if self.mode == "SABR_DELTA_HEDGED_STRANGLE":
            self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.fit_sabr)
        if self.mode == "AS_MARKET_MAKING":
            self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.mm_hedge)

    # ------------------------------------------------------------------ expiry book
    def target_expiry(self):
        for e in self.venue.expiries(min_hours=36, max_days=9):
            if e.weekday() == 4:
                return e
        return None

    def refresh_book(self):
        """Keep the target expiry's strikes within +/-20 % subscribed (refreshed daily)."""
        exp = self.book_expiry if (self.legs and self.book_expiry) else self.target_expiry()
        day = self.Time.date()
        if exp is None or (exp == self.book_expiry and day == self.book_day):
            return
        spot = self.venue.spot()
        book = []
        for right in ("call", "put"):
            for k in self.venue.strikes(exp, right):
                if abs(k / spot - 1) <= 0.20:
                    sym = self.venue.symbol(exp, k, right)
                    self.venue.add(sym)
                    book.append(sym)
        self.book, self.book_expiry, self.book_day = book, exp, day
        self.venue.drop_unused(keep=book + [s for s, _ in self.legs])

    def info(self, syms=None):
        """quote, implied vol and delta for the given contracts (default: whole book)."""
        out = {}
        for s in (self.book if syms is None else syms):
            q = self.venue.quote(s)
            iv = self.venue.iv(s) if q else None
            if q and iv:
                out[s] = (q, iv, self.venue.delta(s, iv))
        return out

    def near_money(self, width):
        spot = self.venue.spot()
        return [s for s in self.book if abs(float(s.ID.StrikePrice) / spot - 1) <= width]

    def by_delta(self, info, right, target):
        best = None
        for s, (_, _, d) in info.items():
            if (s.ID.OptionRight == OptionRight.Call) == (right == "call"):
                g = abs(abs(d) - target)
                if best is None or g < best[0]:
                    best = (g, s)
        return best[1] if best and best[0] < 0.08 else None

    def atm(self, syms, right):
        spot, best = self.venue.spot(), None
        for s in syms:
            if (s.ID.OptionRight == OptionRight.Call) == (right == "call"):
                g = abs(float(s.ID.StrikePrice) - spot)
                if best is None or g < best[0]:
                    best = (g, s)
        return best[1] if best else None

    def strike(self, syms, right, k):
        for s in syms:
            if float(s.ID.StrikePrice) == k and (s.ID.OptionRight == OptionRight.Call) == (right == "call"):
                return s
        return None

    def atm_pair(self):
        c = self.atm(self.book, "call")
        p = self.strike(self.book, "put", float(c.ID.StrikePrice)) if c else None
        if not (c and p and self.venue.quote(c) and self.venue.quote(p)):
            return None, None
        return c, p

    # ------------------------------------------------------------------ main loop
    def OnData(self, data):
        t = self.Time
        self.venue.on_data(t)
        if self.u in data.Bars:
            m = t.replace(second=0, microsecond=0)
            if m != self._last_min:
                self._last_min = m
                self.prices.append(float(data.Bars[self.u].Close))
        # Option data are minute bars, so a real (not fill-forwarded) option print can only arrive at second 0.
        # With the 1-second perpetual LEAN fill-forwards every option bar every second; scanning them all each
        # second took ~60 % of the run time. Same logic, only evaluated when a real option bar can exist.
        if t.second == 0:
            for sym, bar in data.Bars.items():
                if sym.SecurityType == SecurityType.Option and bar.Volume > 0 and not bar.IsFillForward:
                    self.last_trade_time[sym] = t
            if self.mode == "AS_MARKET_MAKING" and not self.IsWarmingUp:
                self.mm_match(data)
        if self.mode == "NBP_FLOW_STRADDLE" and data.ContainsKey(self.flow_sym):
            self._add_flow(data[self.flow_sym].EndTime, float(data[self.flow_sym].Value))
        if self._warming:                       # IsWarmingUp never turns back on: stop asking the engine once off
            if self.IsWarmingUp:
                return
            self._warming = False
        if t - self.last_check < self.check_every:
            return
        self.last_check = t
        if self.mode == "BUY_AND_HOLD":
            if not self.Portfolio[self.u].Invested and self.venue.spot() > 0:
                self.MarketOrder(self.u, self.venue.round_perp(
                    float(self.Portfolio.TotalPortfolioValue) / self.venue.spot() * 0.999))
            return
        self.refresh_book()
        if self.book_expiry is not None and self.book_expiry == self.blocked_expiry:
            return
        getattr(self, "step_" + self.mode.lower())()

    def unit_size(self):
        return self.venue.coins_for_notional(self.unit_frac, self.cap)

    def can_enter(self):
        return self.last_entry is None or self.Time < self.last_entry

    def open_legs(self, legs, tag):
        if not self.can_enter():
            return self.skip("after last_entry")
        legs = [(s, q) for s, q in legs if s is not None and q != 0]
        for s, q in legs:
            self.MarketOrder(s, q, tag=tag)
        self.legs = legs
        self.stats["cycles"] += 1
        self.entry = {"time": self.Time, "spot": self.venue.spot(), "value": self.legs_value()}

    def legs_value(self):
        v = 0.0
        for s, _ in self.legs:
            qt = self.venue.quote(s)
            if qt:
                v += float(self.Portfolio[s].Quantity) * qt[2]
        return v

    def close_all(self, tag):
        for s, _ in self.legs:
            h = self.Portfolio[s]
            if h.Invested:
                self.MarketOrder(s, -h.Quantity, tag=tag)
        if self.Portfolio[self.u].Invested:
            self.MarketOrder(self.u, -self.Portfolio[self.u].Quantity, tag=tag)
        self.legs = []

    def hedge_to(self, target_perp, band):
        perp = float(self.Portfolio[self.u].Quantity)
        if abs(target_perp - perp) >= band:
            q = self.venue.round_perp(target_perp - perp)
            if q != 0:
                self.MarketOrder(self.u, q, tag="delta hedge")
                self.stats["hedges"] += 1

    def skip(self, why):
        self.stats["skipped"][why] = self.stats["skipped"].get(why, 0) + 1

    # ------------------------------------------------------------------ strategies
    def step_gamma_scalping(self):
        if not self.legs:
            c, p = self.atm_pair()
            q = self.unit_size()
            if not (c and q):
                return self.skip("no ATM pair")
            self.unit = q
            self.open_legs([(c, q), (p, q)], "long straddle")
            return
        self.hedge_to(-self.venue.option_delta_exposure(), 0.04 * self.unit)

    def fit_sabr(self):
        if self.IsWarmingUp or not self.legs:
            return
        F = self.venue.spot()
        pts = {}
        for s, (_, iv, _) in self.info(self.near_money(0.20)).items():
            k = float(s.ID.StrikePrice)
            otm = (s.ID.OptionRight == OptionRight.Call and k >= F) or (s.ID.OptionRight == OptionRight.Put and k < F)
            if otm:
                pts[k] = iv
        if len(pts) >= 4:
            ks = sorted(pts)
            try:
                self.sabr = sabr_calibrate(F, self.venue.tau(self.legs[0][0]), ks, [pts[k] for k in ks])
            except Exception as ex:
                self.Debug(f"SABR fit failed: {ex}")

    def step_sabr_delta_hedged_strangle(self):
        if not self.legs:
            info = self.info(self.near_money(0.20))
            c, p = self.by_delta(info, "call", 0.20), self.by_delta(info, "put", 0.20)
            q = self.unit_size()
            if not (c and p and q):
                return self.skip("no 0.20-delta pair")
            self.unit = q
            self.open_legs([(c, -q), (p, -q)], "short strangle")
            self.fit_sabr()
            return
        if not self.sabr:
            return
        F = self.venue.spot()
        d = sum(float(self.Portfolio[s].Quantity) * sabr_bartlett_delta(
            F, float(s.ID.StrikePrice), self.venue.tau(s), self.sabr, s.ID.OptionRight == OptionRight.Call)
            for s, _ in self.legs if self.Portfolio[s].Invested)
        self.hedge_to(-d, 0.08 * self.unit)

    def step_short_strangle_stop_hedge(self):
        if not self.legs:
            info = self.info(self.near_money(0.20))
            c, p = self.by_delta(info, "call", 0.20), self.by_delta(info, "put", 0.20)
            q = self.unit_size()
            if not (c and p and q):
                return self.skip("no 0.20-delta pair")
            self.unit = q
            self.open_legs([(c, -q), (p, -q)], "short strangle")
            return
        move = self.venue.spot() / self.entry["spot"] - 1
        lot = 0.5 * self.venue.perp_lot
        if move > 0.015:
            self.hedge_to(self.unit, lot)
        elif move < -0.015:
            self.hedge_to(-self.unit, lot)
        elif abs(move) < 0.005:
            self.hedge_to(0.0, lot)

    # --- market making (Avellaneda & Stoikov 2008, parameters calibrated to the observed market width)
    # Quotes are kept in the algorithm and matched once per minute against the real Deribit prints of
    # that minute: our ask is lifted if a trade printed at/above it, our bid is hit if a trade printed
    # at/below it, for at most 25 % of the printed volume (no queue priority assumed beyond that).
    def mm_match(self, data):
        if not self.mm_quotes:
            return
        lot = self.venue.opt_lot
        for s, qt in list(self.mm_quotes.items()):
            if s not in data.Bars:
                continue
            bar = data.Bars[s]
            if bar.IsFillForward or bar.Volume <= 0:
                continue
            avail = math.floor(float(bar.Volume) * 0.25 / lot) * lot
            inv = float(self.Portfolio[s].Quantity)
            if qt.get("ask") and float(bar.High) >= qt["ask"] and inv > -qt["cap"] and avail > 0:
                q = min(qt["size"], avail, inv + qt["cap"]) if qt.get("unwind_only") is None else min(qt["size"], avail, max(inv, 0.0))
                q = math.floor(q / lot) * lot
                if q > 0:
                    self.MarketOrder(s, -q, tag=f"MMFILL@{qt['ask']:.4f} ask")
                    self.stats["mm_fills"] += 1
            inv = float(self.Portfolio[s].Quantity)
            if qt.get("bid") and float(bar.Low) <= qt["bid"] and inv < qt["cap"] and avail > 0:
                q = min(qt["size"], avail, qt["cap"] - inv) if qt.get("unwind_only") is None else min(qt["size"], avail, max(-inv, 0.0))
                q = math.floor(q / lot) * lot
                if q > 0:
                    self.MarketOrder(s, q, tag=f"MMFILL@{qt['bid']:.4f} bid")
                    self.stats["mm_fills"] += 1

    def step_as_market_making(self):
        if not self.can_enter():
            self.mm_quotes = {}
            return
        base = self.unit_size()
        lot = self.venue.opt_lot
        q_quote = math.floor(0.1 * base / lot) * lot
        if q_quote <= 0:
            return
        self.unit = base
        inv_cap = 5 * q_quote
        filled_since = any(float(self.Portfolio[s].Quantity) != q for s, q in self.legs)
        if self.Time - self.mm_last_quote < timedelta(minutes=15) and not filled_since:
            return
        self.mm_last_quote = self.Time
        targets = [s for s in (self.atm(self.book, "call"), self.atm(self.book, "put")) if s is not None]
        held = [s for s, _ in self.legs if self.Portfolio[s].Invested]
        quotes = {}
        for s in set(targets + held):
            inf = self.info([s]).get(s)
            if not inf:
                continue
            (bid, ask, mid), iv, delta = inf
            h_mkt = 0.5 * (ask - bid)
            if h_mkt <= 0:
                continue
            tau = self.venue.tau(s)
            if tau < 1.0 / (365.0 * 24.0):                             # no quoting in the last hour to expiry
                continue
            H = min(tau, 1.0 / 365.0)
            var = (abs(delta) * self.venue.spot() * iv) ** 2 * H        # option price variance over H (USD^2)
            if var <= 1e-12:
                continue
            gamma = h_mkt / (inv_cap * var)                            # full inventory shifts r by h_mkt
            kappa = 2.0 / max(2.0 * h_mkt - gamma * var, 1e-9)         # quoted width = market width
            spread = gamma * var + (2.0 / gamma) * math.log(1.0 + gamma / kappa)
            inv = float(self.Portfolio[s].Quantity)
            r = mid - inv * gamma * var
            qb, qa = round(r - spread / 2, 4), round(r + spread / 2, 4)
            if s in targets:
                quotes[s] = {"bid": qb if qb > 0 else None, "ask": qa, "size": q_quote, "cap": inv_cap}
            elif inv != 0:                                             # no longer ATM: only work inventory off
                quotes[s] = {"bid": qb if (inv < 0 and qb > 0) else None, "ask": qa if inv > 0 else None,
                             "size": q_quote, "cap": inv_cap, "unwind_only": True}
        self.mm_quotes = quotes
        self.legs = [(s, float(self.Portfolio[s].Quantity)) for s in set(targets + held)]

    def mm_hedge(self):
        if not self.IsWarmingUp and self.legs:
            self.hedge_to(-self.venue.option_delta_exposure(), 0.05 * max(self.unit, 1e-9))

    # --- order-flow signal
    def _add_flow(self, t, v):
        hour = t.replace(minute=0, second=0, microsecond=0)
        if self._flow_hour is not None and hour != self._flow_hour:
            self.flow_hours.append(self._flow_hour_sum)
            self._flow_hour_sum = 0.0
        self._flow_hour = hour
        self._flow_hour_sum += v
        self.flow_minutes.append((t, v))
        while self.flow_minutes and self.flow_minutes[0][0] <= t - timedelta(minutes=60):
            self.flow_minutes.popleft()

    def step_nbp_flow_straddle(self):
        if self.legs:
            if self.Time - self.entry["time"] >= timedelta(hours=4):
                self.close_all("NBP exit after 4h")
            return
        if len(self.flow_hours) < 48:
            return
        hist = np.array(self.flow_hours)
        sd = hist.std()
        if sd <= 0:
            return
        z = (sum(v for _, v in self.flow_minutes) - hist.mean()) / sd
        if z > 2.0:
            c, p = self.atm_pair()
            q = self.unit_size()
            if c and q:
                self.stats["signals"] += 1
                self.unit = q
                self.open_legs([(c, q), (p, q)], f"NBP z={z:.1f} long straddle")

    # --- volatility-spike premium selling
    def step_post_spike_premium_sell(self):
        if self.legs:
            pnl = self.legs_value() - self.entry["value"]
            premium = -self.entry["value"]
            if self.Time - self.entry["time"] >= timedelta(hours=6) or (premium > 0 and pnl < -1.5 * premium):
                if pnl < -1.5 * premium:
                    self.stats["stops"] += 1
                self.close_all("spike-sell exit")
            return
        if len(self.prices) < 61:
            return
        move = self.prices[-1] / self.prices[-61] - 1
        if abs(move) > 0.015:
            info = self.info(self.near_money(0.20))
            c, p = self.by_delta(info, "call", 0.20), self.by_delta(info, "put", 0.20)
            q = self.unit_size()
            if c and p and q:
                self.stats["signals"] += 1
                self.unit = q
                self.open_legs([(c, -q), (p, -q)], f"spike {move:+.2%} short strangle")

    # --- parity arbitrage
    def step_parity_conversion_arb(self):
        if self.legs or not self.can_enter():
            return                                   # locked until settlement
        perp = self.Securities[self.u]
        pb, pa = float(perp.BidPrice), float(perp.AskPrice)
        spot = self.venue.spot()
        if pa <= 0:
            return
        fresh = lambda s: s in self.last_trade_time and self.Time - self.last_trade_time[s] <= timedelta(minutes=10)
        cands = [s for s in self.near_money(0.05) if fresh(s)]
        best = None
        for c in [s for s in cands if s.ID.OptionRight == OptionRight.Call]:
            k = float(c.ID.StrikePrice)
            p = self.strike(cands, "put", k)
            if p is None:
                continue
            qc, qp = self.venue.quote(c), self.venue.quote(p)
            if not (qc and qp and qc[0] > 0 and qp[0] > 0):
                continue
            (cb, ca, _), (pbid, pask, _) = qc, qp
            hours = self.venue.tau(c) * 365 * 24
            fee = 2 * min(0.0003 * spot, 0.125 * max(cb, pask)) + 0.0005 * spot + 0.00015 * spot
            fund = abs(self.venue.funding.get(self.venue.now_ms() // 3_600_000, 0.0)) * spot * hours
            conv = (cb - pask + k) - pa - fee - fund      # sell call, buy put, buy perpetual
            rev = pb - (ca - pbid + k) - fee - fund       # buy call, sell put, sell perpetual
            for edge, kind in ((conv, "conversion"), (rev, "reversal")):
                if edge > 0.0005 * spot and (best is None or edge > best[0]):
                    best = (edge, kind, c, p)
        if not best:
            return
        q = self.unit_size()
        if q <= 0:
            return
        edge, kind, c, p = best
        sign = 1 if kind == "conversion" else -1
        self.open_legs([(c, -sign * q), (p, sign * q)], f"{kind} edge={edge:.2f}")
        self.MarketOrder(self.u, self.venue.round_perp(sign * q), tag=f"{kind} perpetual leg")
        self.stats["arb_trades"] += 1
        self.stats["arb_locked_edge_usd"] += edge * q

    # --- directional structures (direction fixed by the trailing 24 h return, known at entry)
    def trend_up(self):
        return self.prices[-1] >= self.prices[-1441]

    def step_trend_ratio_spread_stop(self):
        if self.legs:
            short = [s for s, q in self.legs if q < 0][0]
            k, spot = float(short.ID.StrikePrice), self.venue.spot()
            if (short.ID.OptionRight == OptionRight.Put and spot < 0.96 * k) or \
                    (short.ID.OptionRight == OptionRight.Call and spot > 1.04 * k):
                self.close_all("ratio spread 4% stop")
                self.stats["stops"] += 1
                self.blocked_expiry = self.book_expiry
            return
        if len(self.prices) < 1441:
            return
        right = "call" if self.trend_up() else "put"
        info = self.info(self.near_money(0.20))
        a, b = self.atm(list(info), right), self.by_delta(info, right, 0.25)
        q = self.unit_size()
        if a and b and q and a != b:
            self.unit = q
            self.open_legs([(a, q), (b, -2 * q)], f"{right} ratio 1x2")

    def step_trend_butterfly(self):
        if self.legs or len(self.prices) < 1441:
            return
        right = "call" if self.trend_up() else "put"
        syms = [s for s in self.book if (s.ID.OptionRight == OptionRight.Call) == (right == "call") and self.venue.quote(s)]
        a = self.atm(syms, right)
        q = self.unit_size()
        if not (a and q):
            return
        k0 = float(a.ID.StrikePrice)
        sgn = 1 if right == "call" else -1
        ks = sorted(float(s.ID.StrikePrice) for s in syms)
        k1 = min(ks, key=lambda k: abs(k - k0 * (1 + sgn * 0.04)))
        b, c = self.strike(syms, right, k1), self.strike(syms, right, k1 + (k1 - k0))
        if k1 != k0 and b and c:
            self.open_legs([(a, q), (b, -2 * q), (c, q)], f"{right} butterfly")

    # ------------------------------------------------------------------ expiry
    def after_settlement(self):
        """Friday 08:05 UTC: weekly options were cash-settled at 08:00; flatten hedges, start a new cycle."""
        for s, _ in self.legs:
            for o in self.Transactions.GetOpenOrders(s):
                self.Transactions.CancelOrder(o.Id)
        # flatten the perpetual only when no option is left (a position on a later expiry keeps its hedge)
        open_options = [s for s, h in self.Portfolio.items() if h.Invested and s.SecurityType == SecurityType.Option]
        if self.mode != "BUY_AND_HOLD" and self.Portfolio[self.u].Invested and not open_options:
            self.MarketOrder(self.u, -self.Portfolio[self.u].Quantity, tag="flatten after settlement")
        left = [s.Value for s, _ in self.legs if self.Portfolio[s].Invested and s.ID.Date.date() <= self.Time.date()]
        if left:
            self.Error(f"legs still open after settlement: {left}")
        self.legs = [(s, q) for s, q in self.legs if self.Portfolio[s].Invested]
        self.sabr, self.book_expiry, self.blocked_expiry = None, None, None
        self.mm_quotes = {}

    def OnEndOfAlgorithm(self):
        v = self.venue.summary()
        self.Log(f"SUMMARY mode={self.mode} coin={self.coin} {self.stats} fees={v['fees_paid']} "
                 f"funding={v['funding_paid']} settlements={v['settlements']}")
        self.ObjectStore.Save(f"v2_intraday_{self.coin}_{self.mode}_summary",
                              str({**v, **self.stats, "mode": self.mode, "coin": self.coin}))
