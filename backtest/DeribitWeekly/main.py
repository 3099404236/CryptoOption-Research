# region imports
from AlgorithmImports import *
# endregion
"""
Weekly Deribit option structures (V2): BTC 2019-2026 / ETH 2020-2026, minute data.

Every week (Monday 09:30 UTC, retried Tue/Wed if no tradable quotes) a structure is opened on the
Friday weekly expiry and held to Deribit's cash settlement on Friday 08:00 UTC.
Strikes are chosen by Black-Scholes delta computed from each contract's own implied volatility.
Size: one "unit" = `unit_notional` x equity of underlying notional per leg (default 25 %), capped.

Strategies (parameter `strategy`):
  SHORT_STRANGLE           sell 0.20-delta call and put (naked)
  IRON_CONDOR              sell 0.20-delta call/put, buy 0.07-delta wings
  IRON_CONDOR_HEDGED       IRON_CONDOR + hourly perpetual delta hedge when |delta| > 0.10 unit
  BULL_PUT_SPREAD          sell 0.25-delta put, buy 0.10-delta put
  BEAR_CALL_SPREAD         sell 0.25-delta call, buy 0.10-delta call
  CALENDAR_SPREAD          sell weekly ATM call, buy monthly (21-50 d) call at the same strike;
                           monthly leg sold at the bid after the weekly settles
  CALL_RATIO_SPREAD        buy 1 ATM call, sell 2 x 0.25-delta calls
  CALL_RATIO_SPREAD_HEDGED CALL_RATIO_SPREAD + buy perpetual when option delta < -0.15 unit,
                           unwind when it recovers above -0.05 unit (checked hourly)
  CALL_BUTTERFLY           +1 ATM / -2 K+w / +1 K+2w calls, w ~ 4 % of spot  (what the original "BWB" was)
  BROKEN_WING_BUTTERFLY    +1 ATM / -2 K+w / +1 K+3w calls (true broken wing: skip-strike upper wing)
  LONG_STRADDLE            buy ATM call and put
  HAR_VRP                  HAR-RV forecast of next-week realised vol (rolling OLS) vs weekly ATM IV:
                           IV > 1.10 x forecast -> IRON_CONDOR; IV < 0.90 x forecast -> LONG_STRADDLE
  TREND_SKEWED_CONDOR      condor; above 20-day SMA the call side moves to 0.10/0.04 delta and the
                           put side to 0.25/0.10 delta, otherwise symmetric 0.20/0.07
"""
import math
from collections import deque

import numpy as np

from deribit_common import DeribitVenue

MODES = ["SHORT_STRANGLE", "IRON_CONDOR", "IRON_CONDOR_HEDGED", "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD",
         "CALENDAR_SPREAD", "CALL_RATIO_SPREAD", "CALL_RATIO_SPREAD_HEDGED", "CALL_BUTTERFLY",
         "BROKEN_WING_BUTTERFLY", "LONG_STRADDLE", "HAR_VRP", "TREND_SKEWED_CONDOR"]


class DeribitWeekly(QCAlgorithm):
    def Initialize(self):
        self.SetTimeZone(TimeZones.Utc)
        par = lambda k, d: self.GetParameter(k) or d
        self.coin = par("coin", "BTC")
        self.mode = par("strategy", "IRON_CONDOR")
        if self.mode not in MODES:
            raise ValueError(f"unknown strategy {self.mode}")
        s, e = par("start", "2019-01-01"), par("end", "2026-10-15")
        self.SetStartDate(*map(int, s.split("-")))
        self.SetEndDate(*map(int, e.split("-")))
        self.SetCash(float(par("cash", "1000000")))
        self.unit_frac = float(par("unit_notional", "0.25"))
        self.cap = float(par("max_coins", "25" if self.coin == "BTC" else "400"))
        self.venue = DeribitVenue(self, self.coin)
        self.u = self.venue.u
        self.SetBenchmark(self.u)
        self.SetWarmUp(timedelta(days=30))

        self.sma20 = self.SMA(self.u, 20, Resolution.Daily)
        self.rv_days = deque(maxlen=800)       # daily realised variance from 5-minute log returns
        self._rv_acc, self._rv_day, self._last5 = 0.0, None, None
        cons = TradeBarConsolidator(timedelta(minutes=5))
        cons.DataConsolidated += self._on_5min
        self.SubscriptionManager.AddConsolidator(self.u, cons)

        self.pending = None                    # dict while waiting for candidate quotes
        self.week_done = set()                 # expiry dates already traded or abandoned
        self.legs = []                         # (symbol, qty) of the open structure
        self.far_leg = None
        self.unit = 0.0
        self.stats = {"cycles": 0, "skipped": {}, "hedge_trades": 0}
        for dow in (DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday):
            self.Schedule.On(self.DateRules.Every(dow), self.TimeRules.At(9, 30), self.open_cycle)
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Friday), self.TimeRules.At(8, 5), self.after_settlement)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.hedge)

    # ------------------------------------------------------------------ realised volatility (HAR)
    def _on_5min(self, _, bar):
        day = bar.EndTime.date()
        if self._rv_day is None:
            self._rv_day = day
        if day != self._rv_day:
            self.rv_days.append(self._rv_acc)
            self._rv_acc, self._rv_day = 0.0, day
        if self._last5 and bar.Close > 0:
            self._rv_acc += math.log(float(bar.Close) / self._last5) ** 2
        self._last5 = float(bar.Close)

    def har_forecast_vol(self):
        """Direct HAR: mean RV over the next 7 days regressed on daily/weekly/monthly RV (rolling 365 d)."""
        rv = np.array(self.rv_days)
        if len(rv) < 30 + 7 + 120:
            return None
        X, y = [], []
        for t in range(29, len(rv) - 7):
            X.append([1.0, rv[t], rv[t - 6:t + 1].mean(), rv[t - 29:t + 1].mean()])
            y.append(rv[t + 1:t + 8].mean())
        X, y = np.array(X[-365:]), np.array(y[-365:])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        t = len(rv) - 1
        f = beta @ [1.0, rv[t], rv[t - 6:t + 1].mean(), rv[t - 29:t + 1].mean()]
        m = rv[t - 29:t + 1].mean()
        f = min(max(f, 0.2 * m), 5.0 * m)
        return math.sqrt(365.0 * f)

    # ------------------------------------------------------------------ weekly cycle
    def target_expiry(self):
        for e in self.venue.expiries(min_hours=36, max_days=7):
            if e.weekday() == 4:
                return e
        return None

    def open_cycle(self):
        if self.IsWarmingUp or self.pending or self.legs:
            return
        exp = self.target_expiry()
        if exp is None:
            return self._skip("no Friday expiry listed")
        if exp in self.week_done:
            return
        expiries = [exp]
        far = None
        if self.mode == "CALENDAR_SPREAD":
            fars = [x for x in self.venue.expiries(min_hours=21 * 24, max_days=50)
                    if x.weekday() == 4 and (x + timedelta(days=7)).month != x.month]
            if not fars:
                self.week_done.add(exp)
                return self._skip("no monthly expiry 21-50 days out")
            far = fars[0]
            expiries.append(far)
        spot = self.venue.spot()
        cands = []
        for x in expiries:
            for right in ("call", "put"):
                for k in self.venue.strikes(x, right):
                    if abs(k / spot - 1) <= 0.25:
                        sym = self.venue.symbol(x, k, right)
                        self.venue.add(sym)
                        cands.append(sym)
        self.pending = {"expiry": exp, "far": far, "cands": cands, "ready": self.Time + timedelta(minutes=2)}

    def OnData(self, data):
        self.venue.on_data()
        if self.pending and self.Time >= self.pending["ready"]:
            p, self.pending = self.pending, None
            try:
                self.deploy(p)
            finally:
                self.venue.drop_unused(keep=[s for s, _ in self.legs] + ([self.far_leg] if self.far_leg else []))

    def _quotes(self, cands):
        out = {}
        for s in cands:
            q = self.venue.quote(s)
            iv = self.venue.iv(s) if q else None
            if q and iv:
                out[s] = (q, iv, self.venue.delta(s, iv))
        return out

    def _by_delta(self, info, expiry, right, target):
        best = None
        for s, (_, _, d) in info.items():
            if s.ID.Date == expiry and (s.ID.OptionRight == OptionRight.Call) == (right == "call"):
                gap = abs(abs(d) - target)
                if best is None or gap < best[0]:
                    best = (gap, s)
        return best[1] if best and best[0] < 0.08 else None

    def _atm(self, info, expiry, right):
        spot, best = self.venue.spot(), None
        for s in info:
            if s.ID.Date == expiry and (s.ID.OptionRight == OptionRight.Call) == (right == "call"):
                gap = abs(float(s.ID.StrikePrice) - spot)
                if best is None or gap < best[0]:
                    best = (gap, s)
        return best[1] if best else None

    def _strike(self, info, expiry, right, strike):
        for s in info:
            if s.ID.Date == expiry and float(s.ID.StrikePrice) == strike and \
                    (s.ID.OptionRight == OptionRight.Call) == (right == "call"):
                return s
        return None

    def deploy(self, p):
        exp, mode = p["expiry"], self.mode
        info = self._quotes(p["cands"])
        q = self.venue.coins_for_notional(self.unit_frac, self.cap)
        if q <= 0 or not info:
            return self._skip("no quotes or zero size", retry=True, exp=exp)
        legs = []

        def condor(c_short, c_long, p_short, p_long):
            cs, cl = self._by_delta(info, exp, "call", c_short), self._by_delta(info, exp, "call", c_long)
            ps, pl = self._by_delta(info, exp, "put", p_short), self._by_delta(info, exp, "put", p_long)
            if None in (cs, cl, ps, pl) or cs == cl or ps == pl:
                return None
            return [(cs, -q), (cl, q), (ps, -q), (pl, q)]

        if mode == "SHORT_STRANGLE":
            c, pu = self._by_delta(info, exp, "call", 0.20), self._by_delta(info, exp, "put", 0.20)
            legs = [(c, -q), (pu, -q)] if c and pu else None
        elif mode in ("IRON_CONDOR", "IRON_CONDOR_HEDGED"):
            legs = condor(0.20, 0.07, 0.20, 0.07)
        elif mode == "TREND_SKEWED_CONDOR":
            bull = self.sma20.IsReady and self.venue.spot() > self.sma20.Current.Value
            legs = condor(0.10, 0.04, 0.25, 0.10) if bull else condor(0.20, 0.07, 0.20, 0.07)
        elif mode == "BULL_PUT_SPREAD":
            a, b = self._by_delta(info, exp, "put", 0.25), self._by_delta(info, exp, "put", 0.10)
            legs = [(a, -q), (b, q)] if a and b and a != b else None
        elif mode == "BEAR_CALL_SPREAD":
            a, b = self._by_delta(info, exp, "call", 0.25), self._by_delta(info, exp, "call", 0.10)
            legs = [(a, -q), (b, q)] if a and b and a != b else None
        elif mode == "LONG_STRADDLE":
            c = self._atm(info, exp, "call")
            pu = self._strike(info, exp, "put", float(c.ID.StrikePrice)) if c else None
            legs = [(c, q), (pu, q)] if c and pu else None
        elif mode in ("CALL_RATIO_SPREAD", "CALL_RATIO_SPREAD_HEDGED"):
            a, b = self._atm(info, exp, "call"), self._by_delta(info, exp, "call", 0.25)
            legs = [(a, q), (b, -2 * q)] if a and b and b.ID.StrikePrice > a.ID.StrikePrice else None
        elif mode in ("CALL_BUTTERFLY", "BROKEN_WING_BUTTERFLY"):
            legs = self._butterfly(info, exp, q, broken=(mode == "BROKEN_WING_BUTTERFLY"))
        elif mode == "CALENDAR_SPREAD":
            near = self._atm(info, exp, "call")
            far = self._strike(info, p["far"], "call", float(near.ID.StrikePrice)) if near else None
            if near and far:
                legs = [(near, -q), (far, q)]
                self.far_leg = far
        elif mode == "HAR_VRP":
            fv = self.har_forecast_vol()
            c = self._atm(info, exp, "call")
            pu = self._strike(info, exp, "put", float(c.ID.StrikePrice)) if c else None
            if fv is None or not (c and pu):
                self.week_done.add(exp)
                return self._skip("HAR not ready" if fv is None else "no ATM pair")
            atm_iv = 0.5 * (info[c][1] + info[pu][1])
            if atm_iv > 1.10 * fv:
                legs = condor(0.20, 0.07, 0.20, 0.07)
            elif atm_iv < 0.90 * fv:
                legs = [(c, q), (pu, q)]
            else:
                self.week_done.add(exp)
                return self._skip("IV within 10% of HAR forecast (no trade)")
            self.Log(f"HAR {exp:%Y-%m-%d} atm_iv={atm_iv:.3f} har={fv:.3f} -> {'SELL' if atm_iv > fv else 'BUY'}")
        if not legs:
            return self._skip("strikes not available", retry=True, exp=exp)
        for s, qty in legs:
            self.MarketOrder(s, qty, tag=f"{mode} open")
        self.legs, self.unit = legs, q
        self.week_done.add(exp)
        self.stats["cycles"] += 1

    def _butterfly(self, info, exp, q, broken):
        a = self._atm(info, exp, "call")
        if not a:
            return None
        k0 = float(a.ID.StrikePrice)
        ks = sorted({float(s.ID.StrikePrice) for s in info if s.ID.Date == exp and s.ID.OptionRight == OptionRight.Call})
        k1 = min(ks, key=lambda k: abs(k - k0 * 1.04))
        w = k1 - k0
        k2 = k1 + (2 * w if broken else w)
        b, c = self._strike(info, exp, "call", k1), self._strike(info, exp, "call", k2)
        if w <= 0 or not (b and c):
            return None
        return [(a, q), (b, -2 * q), (c, q)]

    def _skip(self, why, retry=False, exp=None):
        if not retry and exp:
            self.week_done.add(exp)
        if retry and exp and self.Time.weekday() >= 2:   # Wednesday attempt failed: give up this week
            self.week_done.add(exp)
        self.stats["skipped"][why] = self.stats["skipped"].get(why, 0) + 1

    # ------------------------------------------------------------------ hedging & expiry
    def hedge(self):
        if self.IsWarmingUp or not self.legs or self.mode not in ("IRON_CONDOR_HEDGED", "CALL_RATIO_SPREAD_HEDGED"):
            return
        opt_delta = self.venue.option_delta_exposure()
        perp = float(self.Portfolio[self.u].Quantity)
        if self.mode == "IRON_CONDOR_HEDGED":
            gap = -opt_delta - perp
            if abs(opt_delta + perp) > 0.10 * self.unit:
                self._trade_perp(gap)
        else:
            if opt_delta < -0.15 * self.unit:
                self._trade_perp(-opt_delta - perp)
            elif perp > 0 and opt_delta > -0.05 * self.unit:
                self._trade_perp(-perp)

    def _trade_perp(self, qty):
        qty = self.venue.round_perp(qty)
        if qty != 0:
            self.MarketOrder(self.u, qty, tag="delta hedge")
            self.stats["hedge_trades"] += 1

    def after_settlement(self):
        """Friday 08:05: the weekly legs were cash-settled at 08:00; close leftovers."""
        if self.far_leg and self.Portfolio[self.far_leg].Invested:
            self.MarketOrder(self.far_leg, -self.Portfolio[self.far_leg].Quantity, tag="calendar far leg close")
        self.far_leg = None
        open_options = [s for s, h in self.Portfolio.items() if h.Invested and s.SecurityType == SecurityType.Option
                        and s.ID.Date.date() > self.Time.date()]
        if self.Portfolio[self.u].Invested and not open_options:
            self.MarketOrder(self.u, -self.Portfolio[self.u].Quantity, tag="flatten hedge")
        self.legs = [(s, q) for s, q in self.legs if self.Portfolio[s].Invested]
        if self.legs:
            self.Error(f"unexpected open legs after settlement: {[s.Value for s, _ in self.legs]}")
        self.legs = []

    def OnEndOfAlgorithm(self):
        v = self.venue.summary()
        self.Log(f"SUMMARY mode={self.mode} coin={self.coin} cycles={self.stats['cycles']} "
                 f"hedge_trades={self.stats['hedge_trades']} skipped={self.stats['skipped']} "
                 f"fees={v['fees_paid']} funding={v['funding_paid']} settlements={v['settlements']}")
        self.ObjectStore.Save(f"v2_weekly_{self.coin}_{self.mode}_summary",
                              str({**v, **self.stats, "mode": self.mode, "coin": self.coin}))
