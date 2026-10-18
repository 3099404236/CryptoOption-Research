# region imports
from AlgorithmImports import *
# endregion
"""
Option-selling research strategies (V2 research round): Deribit BTC 2019-2026 / ETH 2020-2026, minute data.

Each week one structure is sold on the next Friday expiry and held to Deribit's 08:00 UTC cash settlement.
Every parameter is fixed for a run; the parameter choice is made on 2019-2023 only (2024-2026 is the
out-of-sample test). Same venue model as the other V2 projects (fees, quotes, settlement, funding).

Parameters:
  structure  straddle    short ATM call + put
             strangle25  short 0.25-delta call + put      strangle35  short 0.35-delta call + put
             put30       short 0.30-delta put             put40       short 0.40-delta put
             ironfly10   short ATM straddle + long 0.10-delta wings   ironfly05  ... 0.05-delta wings
  entry      fri  Friday 09:00 UTC (right after the weekly settlement), 7-day option
             mon  Monday 09:30 UTC, 4-day option
             (two more attempts one and two hours later if the strikes have no quotes yet)
  hedge      none | daily (08:00 UTC) | h4 (every 4 hours) |
             band10 / band25 (checked hourly: re-hedge when the perpetual differs from the option delta
             by more than 10 % / 25 % of a unit); the first hedge is placed one minute after entry
  filter     none | harX.XX: sell only when the ATM implied vol >= X.XX x the HAR forecast of next-week
             realised vol (direct 7-day HAR, rolling 365-day OLS, 5-minute realised variance)
Size: one unit = `unit_notional` x equity of notional per leg (default 25 %), capped at `max_coins`.
"""
import math
from collections import deque

import numpy as np

from deribit_common import DeribitVenue

STRUCTURES = {
    "straddle":   [("call", "atm", -1), ("put", "atm", -1)],
    "strangle25": [("call", 0.25, -1), ("put", 0.25, -1)],
    "strangle35": [("call", 0.35, -1), ("put", 0.35, -1)],
    "put30":      [("put", 0.30, -1)],
    "put40":      [("put", 0.40, -1)],
    "ironfly10":  [("call", "atm", -1), ("put", "atm", -1), ("call", 0.10, 1), ("put", 0.10, 1)],
    "ironfly05":  [("call", "atm", -1), ("put", "atm", -1), ("call", 0.05, 1), ("put", 0.05, 1)],
}
HEDGES = ("none", "daily", "h4", "band10", "band25")


class DeribitVolSell(QCAlgorithm):
    def Initialize(self):
        self.SetTimeZone(TimeZones.Utc)
        par = lambda k, d: self.GetParameter(k) or d
        self.coin = par("coin", "BTC")
        self.mode = par("strategy", "VOL_SELL")
        self.structure = par("structure", "straddle")
        self.entry = par("entry", "fri")
        self.hedge_mode = par("hedge", "daily")
        self.filter = par("filter", "none")
        if self.structure not in STRUCTURES or self.entry not in ("fri", "mon") or self.hedge_mode not in HEDGES \
                or not (self.filter == "none" or self.filter.startswith("har")):
            raise ValueError(f"bad parameters {self.structure}/{self.entry}/{self.hedge_mode}/{self.filter}")
        self.har_ratio = float(self.filter[3:]) if self.filter.startswith("har") else None
        s, e = par("start", "2019-01-01"), par("end", "2026-10-15")
        self.SetStartDate(*map(int, s.split("-")))
        self.SetEndDate(*map(int, e.split("-")))
        self.SetCash(float(par("cash", "1000000")))
        self.unit_frac = float(par("unit_notional", "0.25"))
        self.cap = float(par("max_coins", "25" if self.coin == "BTC" else "400"))
        self.venue = DeribitVenue(self, self.coin)
        self.u = self.venue.u
        self.SetBenchmark(self.u)
        self.SetWarmUp(timedelta(days=200 if self.har_ratio else 10))

        self.rv_days = deque(maxlen=800)       # daily realised variance from 5-minute log returns
        self._rv_acc, self._rv_day, self._last5 = 0.0, None, None
        cons = TradeBarConsolidator(timedelta(minutes=5))
        cons.DataConsolidated += self._on_5min
        self.SubscriptionManager.AddConsolidator(self.u, cons)

        self.pending = None
        self.hedge_at = None
        self.week_done = set()
        self.legs, self.unit, self.expiry = [], 0.0, None
        self.stats = {"cycles": 0, "hedge_trades": 0, "skipped": {}}
        if self.entry == "fri":
            day, times = DayOfWeek.Friday, [(9, 0), (10, 0), (11, 0)]
        else:
            day, times = DayOfWeek.Monday, [(9, 30), (10, 30), (11, 30)]
        for i, (h, m) in enumerate(times):
            self.Schedule.On(self.DateRules.Every(day), self.TimeRules.At(h, m),
                             lambda last=(i == len(times) - 1): self.open_cycle(last))
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Friday), self.TimeRules.At(8, 5), self.after_settlement)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.hedge_check)

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
        return math.sqrt(365.0 * min(max(f, 0.2 * m), 5.0 * m))

    # ------------------------------------------------------------------ weekly cycle
    def target_expiry(self):
        lo, hi = (6 * 24, 8) if self.entry == "fri" else (36, 5)
        for x in self.venue.expiries(min_hours=lo, max_days=hi):
            if x.weekday() == 4:
                return x
        return None

    def open_cycle(self, last_attempt):
        if self.IsWarmingUp or self.pending or self.legs:
            return
        exp = self.target_expiry()
        if exp is None:
            return self._skip("no Friday expiry listed", final=last_attempt)
        if exp in self.week_done:
            return
        spot, cands = self.venue.spot(), []
        for right in ("call", "put"):
            for k in self.venue.strikes(exp, right):
                if abs(k / spot - 1) <= 0.25:
                    sym = self.venue.symbol(exp, k, right)
                    self.venue.add(sym)
                    cands.append(sym)
        self.pending = {"expiry": exp, "cands": cands, "ready": self.Time + timedelta(minutes=2), "last": last_attempt}

    def OnData(self, data):
        self.venue.on_data()
        if self.pending and self.Time >= self.pending["ready"]:
            p, self.pending = self.pending, None
            try:
                self.deploy(p)
            finally:
                self.venue.drop_unused(keep=[s for s, _ in self.legs])
        if self.hedge_at and self.Time >= self.hedge_at:
            self.hedge_at = None
            self.rehedge(force=True)

    def _quotes(self, cands):
        out = {}
        for s in cands:
            q = self.venue.quote(s)
            iv = self.venue.iv(s) if q else None
            if q and iv:
                out[s] = (q, iv, self.venue.delta(s, iv))
        return out

    def _pick(self, info, exp, right, target, atm_k):
        is_call = right == "call"
        if target == "atm":
            for s in info:
                if s.ID.Date == exp and (s.ID.OptionRight == OptionRight.Call) == is_call and float(s.ID.StrikePrice) == atm_k:
                    return s
            return None
        best = None
        for s, (_, _, d) in info.items():
            if s.ID.Date == exp and (s.ID.OptionRight == OptionRight.Call) == is_call and 0.02 < abs(d) < 0.6:
                gap = abs(abs(d) - target)
                if best is None or gap < best[0]:
                    best = (gap, s)
        return best[1] if best and best[0] <= 0.06 else None

    def deploy(self, p):
        exp = p["expiry"]
        info = self._quotes(p["cands"])
        q = self.venue.coins_for_notional(self.unit_frac, self.cap)
        if q <= 0 or not info:
            return self._skip("no quotes or zero size", final=p["last"], exp=exp)
        spot = self.venue.spot()
        calls = {float(s.ID.StrikePrice) for s in info if s.ID.OptionRight == OptionRight.Call}
        puts = {float(s.ID.StrikePrice) for s in info if s.ID.OptionRight == OptionRight.Put}
        both = sorted(calls & puts, key=lambda k: abs(k - spot))
        atm_k = both[0] if both and abs(both[0] / spot - 1) < 0.03 else None
        spec = STRUCTURES[self.structure]
        if any(t == "atm" for _, t, _ in spec) and atm_k is None:
            return self._skip("no ATM call/put pair", final=p["last"], exp=exp)
        if self.har_ratio:
            fv = self.har_forecast_vol()
            if atm_k is None:
                return self._skip("no ATM pair for the IV filter", final=p["last"], exp=exp)
            ivs = [v[1] for s, v in info.items() if float(s.ID.StrikePrice) == atm_k]
            atm_iv = sum(ivs) / len(ivs)
            if fv is None:
                self.week_done.add(exp)
                return self._skip("HAR not ready", final=True)
            if atm_iv < self.har_ratio * fv:
                self.week_done.add(exp)
                return self._skip("IV below HAR threshold (no trade)", final=True)
        legs = []
        for right, target, sign in spec:
            s = self._pick(info, exp, right, target, atm_k)
            if s is None:
                return self._skip("strikes not available", final=p["last"], exp=exp)
            legs.append((s, sign * q))
        if len({s for s, _ in legs}) < len(legs):
            return self._skip("legs collapse onto the same contract", final=p["last"], exp=exp)
        for s, qty in legs:
            self.MarketOrder(s, qty, tag=f"VOL_SELL {self.structure} open")
        self.legs, self.unit, self.expiry = legs, q, exp
        self.week_done.add(exp)
        self.stats["cycles"] += 1
        if self.hedge_mode != "none":
            self.hedge_at = self.Time + timedelta(minutes=1)

    def _skip(self, why, final=False, exp=None):
        if final and exp:
            self.week_done.add(exp)
        self.stats["skipped"][why] = self.stats["skipped"].get(why, 0) + 1

    # ------------------------------------------------------------------ hedging & expiry
    def hedge_check(self):
        if self.IsWarmingUp or not self.legs or self.hedge_mode == "none":
            return
        if self.Time >= self.expiry + timedelta(hours=8):        # settled; after_settlement flattens
            return
        h = self.Time.hour
        if self.hedge_mode == "daily" and h != 8:
            return
        if self.hedge_mode == "h4" and h % 4 != 0:
            return
        self.rehedge(force=self.hedge_mode in ("daily", "h4"))

    def rehedge(self, force):
        target = -self.venue.option_delta_exposure()
        gap = target - float(self.Portfolio[self.u].Quantity)
        band = {"band10": 0.10, "band25": 0.25}.get(self.hedge_mode, 0.0) * self.unit
        if not force and abs(gap) <= band:
            return
        qty = self.venue.round_perp(gap)
        if qty != 0:
            self.MarketOrder(self.u, qty, tag="delta hedge")
            self.stats["hedge_trades"] += 1

    def after_settlement(self):
        open_options = [s for s, h in self.Portfolio.items() if h.Invested and s.SecurityType == SecurityType.Option
                        and s.ID.Date.date() > self.Time.date()]
        if self.Portfolio[self.u].Invested and not open_options:
            self.MarketOrder(self.u, -self.Portfolio[self.u].Quantity, tag="flatten hedge")
        if [s for s, _ in self.legs if self.Portfolio[s].Invested and s.ID.Date.date() <= self.Time.date()]:
            self.Error(f"unexpected open legs after settlement: {[s.Value for s, _ in self.legs]}")
        if not open_options:
            self.legs, self.expiry, self.hedge_at = [], None, None

    def OnEndOfAlgorithm(self):
        v = self.venue.summary()
        self.Log(f"SUMMARY mode={self.mode} coin={self.coin} structure={self.structure} entry={self.entry} "
                 f"hedge={self.hedge_mode} filter={self.filter} cycles={self.stats['cycles']} "
                 f"hedge_trades={self.stats['hedge_trades']} skipped={self.stats['skipped']} "
                 f"fees={v['fees_paid']} funding={v['funding_paid']} settlements={v['settlements']}")
        self.ObjectStore.Save(f"v2_volsell_{self.coin}_{self.structure}_{self.entry}_{self.hedge_mode}_{self.filter}",
                              str({**v, **self.stats}))
